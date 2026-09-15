# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Serial DSv4 conversion using DCP storage plans, without a resident GPTModel.

One global save plan is committed once. Write items are packed into file plans
with a target payload size, then written serially to bound per-file buffering.
The reader intersects logical regions with actual DCP chunk offsets, independent
of the TP/PP/EP rank numbers that originally wrote them.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX, _StoragePrefix, _write_item,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata, ChunkStorageMetadata, Metadata, MetadataIndex,
    TensorProperties, TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType, LoadPlan, ReadItem, SavePlan, TensorWriteData, WriteItem, WriteItemType,
)

from . import MappingError
from .artifacts import is_extra_state, is_mcore_mtp_key, mcore_shape_matches, resolve_iteration_dir
from .fp8_native import NativeFp8Target
from .hf_io import NativeQuantizedWeight, parse_size
from .precision import DENSE_DTYPES, HEADER_DTYPES, canonical_hf_name
from .world_one import require_world_one


def validate_regions(key, shape, regions):
    """Check bounds, overlaps and complete volume without allocating a bitmap."""
    if not shape or any(n <= 0 for n in shape):
        raise MappingError(f"Invalid global tensor shape for {key}: {shape}")
    boxes = []
    for offset, size in regions:
        if len(offset) != len(shape) or len(size) != len(shape):
            raise MappingError(f"Invalid region dimensionality: {key}")
        if any(o < 0 or n <= 0 or o + n > total for o, n, total in zip(offset, size, shape)):
            raise MappingError(f"Out-of-bounds region: {key}: {offset}/{size} in {shape}")
        boxes.append((tuple(offset), tuple(o + n for o, n in zip(offset, size))))
    boxes.sort()
    active = []
    for start, end in boxes:
        active = [(a, b) for a, b in active if b[0] > start[0]]
        for other_start, other_end in active:
            if all(max(a, c) < min(b, d) for a, b, c, d in zip(start, end, other_start, other_end)):
                raise MappingError(f"Overlapping checkpoint regions: {key}: {start}, {other_start}")
        active.append((start, end))
    if sum(math.prod(b - a for a, b in zip(start, end)) for start, end in boxes) != math.prod(shape):
        raise MappingError(f"Incomplete checkpoint region coverage: {key}")


def _groups(profile):
    groups = defaultdict(list)
    for piece in profile.pieces:
        groups[piece.key].append(piece)
    for key, pieces in groups.items():
        shape = pieces[0].global_shape
        if any(piece.global_shape != shape for piece in pieces):
            raise MappingError(f"Conflicting mapping shapes for {key}")
        validate_regions(key, shape, [(p.offset, p.shape) for p in pieces])
    return groups


def _storage_dtype(pieces, source, policy):
    types = {source.specs[p.hf].dtype for p in pieces}
    integers = {"I32": torch.int32, "I64": torch.int64}
    if types.intersection(integers):
        if len(types) != 1 or any(not p.hf.endswith("tid2eid") for p in pieces):
            raise MappingError("Integer weights are only supported for the DSv4 hash router")
        return integers[next(iter(types))]
    if policy.native_fp8 and all(source.specs[p.hf].kind in ("fp8", "mxfp4") for p in pieces):
        from .fp8_native import NativeFp8Target
        return NativeFp8Target
    dtypes = {HEADER_DTYPES[code] if code in HEADER_DTYPES and policy.preserve_source_dtypes
              else torch.float32 if code == "F32" else DENSE_DTYPES[policy.dtype]
              for code in types}
    if len(dtypes) != 1:
        raise MappingError(f"Cannot fuse different source dtypes into one DCP tensor without casting: "
                           f"{pieces[0].key}: {sorted(types)}; choose an explicit --dtype if appropriate")
    return next(iter(dtypes))


def _checkpoint_args(model_args, policy):
    if model_args is None:
        raise MappingError("MCore output requires translated model args from HF config")
    values = dict(model_args)
    from .config_mapping import MCORE_MODEL_FIELDS, MCORE_OUTPUT_OVERRIDES

    unknown = values.keys() - MCORE_MODEL_FIELDS - MCORE_OUTPUT_OVERRIDES.keys()
    if unknown:
        raise MappingError(f"Model-only checkpoint args contain non-model fields: {sorted(unknown)}")
    values.update(MCORE_OUTPUT_OVERRIDES)
    values.update(bf16=policy.dtype == "bf16", fp16=policy.dtype == "fp16",
                  params_dtype=DENSE_DTYPES[policy.dtype])
    return SimpleNamespace(**values)


class _LazySaveData:
    def __init__(self, source, lookup, dtypes, groups=None):
        self.source, self.lookup, self.dtypes = source, lookup, dtypes
        self.resolved = set()
        self._group_by_key = {key: list(pieces) for key, pieces in (groups or {}).items()}
        self._qt_cache = {}

    def resolve_data(self, item):
        self.resolved.add(item.index)
        if item.type == WriteItemType.BYTE_IO:
            payload = io.BytesIO()
            # MCore's strategy serializes a list of ShardedObject.data values.
            # None is a supported dense TE extra-state placeholder (no history).
            torch.save([None], payload)
            payload.seek(0)
            return payload
        piece = self.lookup[item.index]
        if self.dtypes[piece.key] is NativeFp8Target:
            return self._native_fp8_piece(piece)
        tensor = self.source.get_tensor(piece.hf)
        if piece.hf_index is not None:
            tensor = tensor[piece.hf_index:piece.hf_index + 1]
        tensor = tensor.reshape(piece.shape).to(device="cpu", dtype=self.dtypes[piece.key])
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise MappingError(f"Non-finite or overflowing tensor: {piece.hf}")
        # Isolate the storage of mHC slices and dequantized views for torch.save.
        return tensor.contiguous().clone()

    def _native_fp8_piece(self, piece):
        """Materialize one chunk as a TE Float8BlockwiseQTensor.

        Pieces are contiguous row ranges of the key's raw E4M3 payload (the
        fused fc1 w1|w3 rows, or the (expert, half) chunk of the shared routed
        expert tensor). Chunks are built from raw uint8/FP32 payload slices,
        so the only qt dispatches used are the verified constructor and
        ``aten.view`` (3-D expert chunks view the 2-D qt, TE keeps 2-D
        physical storage). Scale rows are sliced per 128-row block so fused
        pieces keep their independent block scales.
        """
        from .fp8_native import payload_for_key, qt_for_payload
        siblings = self._group_by_key[piece.key]
        rows, cols = piece.shape[-2:]
        if len(piece.shape) == 3:
            if piece.shape[0] != 1:
                raise MappingError(f"Native FP8 expert chunks must hold one expert: {piece.key}")
            # Global (experts, band, cols) layout: the payload flattens to
            # rows in (expert, band-row) order, so the row start is
            # expert * band + band offset (the fused w1|w3 band stride).
            lo = piece.offset[0] * piece.global_shape[1] + piece.offset[1]
            hi = lo + rows
        else:
            if any(sibling.hf_index is not None for sibling in siblings):
                raise MappingError(f"Native FP8 storage cannot split dense HF tensors: {piece.key}")
            lo, hi = piece.offset[0], piece.offset[0] + rows
        payload, scale_inv = self._key_payload(piece.key, siblings)
        data = payload[lo:hi]
        scales = scale_inv[lo // 128:hi // 128]
        return qt_for_payload(data.contiguous(), scales.contiguous(), piece.shape)

    def _key_payload(self, key, siblings):
        """Load and cache the raw (uint8 payload, FP32 scale_inv) for one key.

        Pieces of one key never share an HF tensor with another key (HF
        quantized weights are whole tensors), so the key-level payload covers
        every chunk of the group; hf_index pieces are stacked in index order.
        """
        payload = self._qt_cache.get(key)
        if payload is None:
            from .fp8_native import payload_for_key
            ordered = sorted(siblings, key=lambda piece: (piece.hf_index or 0, piece.offset))
            payload = payload_for_key(ordered, self.source)
            self._qt_cache[key] = payload
        return payload


def _file_plans(items, max_shard_size):
    """Pack physical chunks, not global tensor sizes, without loading weights."""
    plans, bucket, size = [], [], 0
    for item in items:
        if item.type == WriteItemType.BYTE_IO:
            # Conservative estimate for the serialized [None] extra-state blob.
            item_size = 4096
        else:
            data = item.tensor_data
            if data.properties.dtype is NativeFp8Target:
                # 1 byte of fp8 payload + ~1/4096 of scale factors per element.
                item_size = math.prod(data.chunk.sizes)
            else:
                item_size = math.prod(data.chunk.sizes) * data.properties.dtype.itemsize
        if bucket and size + item_size > max_shard_size:
            plans.append(SavePlan(bucket))
            bucket, size = [], 0
        bucket.append(item)
        size += item_size
        # A single oversized chunk stays intact in its own file. Serialization
        # overhead is additional; the limit is a payload target, not a hard cap.
        if size >= max_shard_size:
            plans.append(SavePlan(bucket))
            bucket, size = [], 0
    if bucket:
        plans.append(SavePlan(bucket))
    return plans


def _write_bucket(writer, plan, resolver):
    """Persist one file plan without DCP's CPU tensor loader.

    FileSystemWriter.write_data routes every tensor through _SerialCpuLoader,
    whose values() does tensor.detach().cpu(). TE Float8BlockwiseQTensors
    dequantize on that no-op transfer (no aten.cpu/_to_copy dispatch), silently
    replacing the fp8 payload with dense bf16. The file layout, byte offsets,
    fsync and WriteResult accounting below replicate _write_files_from_queue's
    serial branch; the loader step is simply skipped so each resolved object —
    qt included — is handed straight to _write_item (torch.save).
    """
    storage_plan: _StoragePrefix = plan.storage_data
    file_name = f"{storage_plan.prefix}0{DEFAULT_SUFFIX}"
    path = writer.fs.concat_path(writer.path, file_name)
    tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
    bytes_items = [item for item in plan.items if item.type == WriteItemType.BYTE_IO]
    write_results = []
    with writer.fs.create_stream(path, "wb") as stream:
        for item in bytes_items:
            data = resolver.resolve_data(item)
            write_results.append(
                _write_item(writer.transforms, stream, data, item, file_name,
                            writer.serialization_format))
        for item in tensor_items:
            data = resolver.resolve_data(item)
            if data.device.type != "cpu":
                data = data.to("cpu")
            write_results.append(
                _write_item(writer.transforms, stream, data, item, file_name,
                            writer.serialization_format))
        try:
            os.fsync(stream.fileno())
        except (AttributeError, OSError):
            os.sync()
    return write_results


def save_streaming_checkpoint(source, profile, policy, config, output_root, *, model_args,
                               max_shard_size="4GB"):
    require_world_one()
    max_shard_size = parse_size(max_shard_size) if isinstance(max_shard_size, str) else max_shard_size
    if not isinstance(max_shard_size, int) or isinstance(max_shard_size, bool) or max_shard_size <= 0:
        raise MappingError("max_shard_size must be a positive byte count")
    saved_args = _checkpoint_args(model_args, policy)
    ignored_bias = [name for name in source.specs if name.endswith(".ffn.gate.bias")
                    and model_args.get("moe_router_enable_expert_bias") is False]
    for name in ignored_bias:
        if bool((source.get_tensor(name) != 0).any()):
            raise MappingError(f"Cannot omit nonzero HF router bias with MCore expert bias disabled: {name}")
    root = Path(output_root)
    directory = root / "iter_0000000"
    directory.mkdir(parents=True, exist_ok=False)
    groups = _groups(profile)
    dtypes = {key: _storage_dtype(pieces, source, policy) for key, pieces in groups.items()}
    if any(dtype is NativeFp8Target for dtype in dtypes.values()):
        # Make the written checkpoint self-loadable under torch>=2.6
        # weights_only unpickling in THIS process too (the load side must
        # register the same allowlist — see fp8_native.register_native_fp8_safe_globals).
        from .fp8_native import register_native_fp8_safe_globals
        register_native_fp8_safe_globals()
    storage_metadata, items, lookup = {}, [], {}
    for key, pieces in groups.items():
        # Native FP8 chunks are stored as TE Float8BlockwiseQTensor objects; the
        # DCP metadata records their fake (dequantized) dtype, as mcore expects.
        properties = TensorProperties(
            dtype=torch.bfloat16 if dtypes[key] is NativeFp8Target else dtypes[key])
        chunks = []
        for i, piece in enumerate(pieces):
            chunk = ChunkStorageMetadata(torch.Size(piece.offset), torch.Size(piece.shape))
            chunks.append(chunk)
            index = MetadataIndex(key, piece.offset, i)
            items.append(WriteItem(index, WriteItemType.SHARD,
                                   tensor_data=TensorWriteData(chunk, properties, torch.Size(piece.global_shape))))
            lookup[index] = piece
        storage_metadata[key] = TensorStorageMetadata(properties, torch.Size(pieces[0].global_shape), chunks)
    extra_keys = sorted({p.extra_key for p in profile.pieces if p.extra_key})
    for key in extra_keys:
        storage_metadata[key] = BytesStorageMetadata()
        items.append(WriteItem(MetadataIndex(key), WriteItemType.BYTE_IO))
    # No high-level save wrapper: neither a real distributed group nor an
    # async worker/pinned copy of the whole state dictionary is needed.
    writer = FileSystemWriter(directory, single_file_per_rank=True, thread_count=1,
                              per_thread_copy_ahead=0, sync_files=True, overwrite=False)
    writer.set_up_storage_writer(is_coordinator=True)
    plans = [writer.prepare_local_plan(plan) for plan in _file_plans(items, max_shard_size)]
    # Allocate every file prefix in one call. These are storage buckets, not
    # distributed ranks; separately planning each bucket would reuse __0_0.distcp.
    plans = writer.prepare_global_plan(plans)
    resolver = _LazySaveData(source, lookup, dtypes, groups)
    results = []
    for plan in plans:
        written = _write_bucket(writer, plan, resolver)
        expected_file = {item.index for item in plan.items}
        if len(written) != len(expected_file) or {r.index for r in written} != expected_file:
            raise MappingError("DCP writer did not persist the complete file plan")
        results.extend(written)
    expected = {item.index for item in items}
    if resolver.resolved != expected or len(results) != len(expected) or {r.index for r in results} != expected:
        raise MappingError("DCP writer did not resolve and persist the complete global plan")
    storage_files = {r.storage_data.relative_path for r in results}
    if len(storage_files) != len(plans):
        raise MappingError("DCP file plans did not produce distinct storage files")
    source.validate_consumed()
    # Only after all tensor/scale and region checks pass is the global metadata
    # finalized. No repeated save() calls overwrite the same checkpoint.
    writer.finish(Metadata(storage_metadata), [results])
    common = {
        "args": saved_args,
        "checkpoint_version": 3.0, "iteration": 0,
        "content_metadata": {"non_homogeneous_layers": True, "singleton_local_shards": False,
                             "checkpoint_content": "model_only", "model_config_source": "hf_config",
                             "deepseek_v4_conversion": policy.report(),
                             "deepseek_v4_hf_config": dict(config),
                             "profile_revision": "dsv4-global-v1"},
    }
    torch.save(common, directory / "common.pt")
    marker = {"sharded_backend": "torch_dist", "sharded_backend_version": 1,
              "common_backend": "torch", "common_backend_version": 1}
    (directory / "metadata.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    (root / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    return {"iteration_dir": str(directory), "stored_tensor_count": len(groups),
            "removed_generated_hf_biases": ignored_bias,
            "stored_chunk_count": len(lookup), "extra_state_count": len(extra_keys),
            "storage_file_count": len(storage_files), "max_shard_size_bytes": max_shard_size,
            "largest_storage_file_bytes": max(
                ((directory / name).stat().st_size for name in storage_files), default=0)}


class _ReadDestination:
    def __init__(self, tensor):
        self.tensor = tensor

    def resolve_tensor(self, item):
        result = self.tensor
        for axis, (offset, size) in enumerate(zip(item.dest_offsets, item.lengths)):
            result = result.narrow(axis, offset, size)
        return result

    def commit_tensor(self, item, tensor):
        pass  # FileSystemReader has already copied into the destination view.


def hf_export_tensors(tensors, generated):
    """Stream actual weights, then materialize only planned neutral router biases."""
    names = set()
    for item in generated:
        name, shape = item["hf"], item["shape"]
        if (item["operation"] != "zero_disabled_router_bias" or item["value"] != 0
                or item["dtype"] != "F32" or not re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.gate\.bias", name)
                or canonical_hf_name(item["name"]) != name
                or len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0 or name in names):
            raise MappingError(f"Invalid generated HF router-bias target: {name}")
        names.add(name)
    for name, tensor in tensors:
        if name in names:
            raise MappingError(f"Generated HF target overlaps a stored source weight: {name}")
        yield name, tensor
    for item in generated:
        tensor = torch.zeros(tuple(item["shape"]), dtype=torch.float32, device="cpu")
        yield item["hf"], tensor
        del tensor


class StreamingCheckpointReader:
    def __init__(self, checkpoint_path, profile, *, skip_mtp=False):
        require_world_one()
        self.directory = resolve_iteration_dir(checkpoint_path)
        self.reader = FileSystemReader(self.directory)
        self.metadata = self.reader.read_metadata()
        self.reader.set_up_storage_reader(self.metadata, is_coordinator=True)
        self.profile, self.used = profile, set()
        self.skip_mtp = skip_mtp
        if skip_mtp and any(is_mcore_mtp_key(key) for key in profile.keys):
            raise MappingError("A plan that drops MTP must contain no MTP target regions")
        self.by_hf = defaultdict(list)
        self.groups = _groups(profile)
        for piece in profile.pieces:
            self.by_hf[piece.hf].append(piece)
        self._validate_metadata()
        self.tensor_dtypes()  # Reject implicit promotion when joining MCore pieces.
        self.native_keys, self.native_hf_names = self._detect_native_keys()

    def _detect_native_keys(self):
        """Find DCP keys whose first chunk is a TE Float8BlockwiseQTensor blob.

        These keys declare a dense bf16 metadata dtype but their storage holds
        serialized TE quantizer objects; reading them through DCP's tensor
        loader would either fail weights_only unpickling or silently
        dequantize. They are read directly as raw payload/scale pairs instead.
        """
        native_keys = set()
        for key in sorted(self.groups):
            entry = self.metadata.state_dict_metadata[key]
            if not isinstance(entry, TensorStorageMetadata) or not entry.chunks:
                continue
            chunk = entry.chunks[0]
            index = MetadataIndex(key, chunk.offsets, 0)
            info = self.metadata.storage_data.get(index)
            if info is None:
                continue
            with (self.directory / info.relative_path).open("rb") as stream:
                stream.seek(info.offset)
                head = stream.read(min(info.length, 4096))
            if b"Float8BlockwiseQTensor" not in head:
                continue
            if entry.properties.dtype != torch.bfloat16:
                raise MappingError(
                    f"Native quantized DCP storage must declare bf16 metadata dtype: {key}: {entry.properties.dtype}")
            native_keys.add(key)
        return native_keys, {piece.hf for piece in self.profile.pieces if piece.key in native_keys}

    def tensor_dtypes(self):
        dtypes = {}
        for name, pieces in self.by_hf.items():
            types = {self.metadata.state_dict_metadata[piece.key].properties.dtype for piece in pieces}
            if len(types) != 1:
                raise MappingError(f"Inconsistent source DCP dtypes for {name}: {sorted(map(str, types))}")
            dtypes[name] = next(iter(types))
        return dtypes

    def _validate_metadata(self):
        ignored = ("optimizer", "rng_state", "rerun_state_machine", "opt_param_scheduler",
                   "common_state", "num_floating_point_operations_so_far")

        def _is_ignored(key: str) -> bool:
            if any(key == prefix or key.startswith(prefix + ".") or key.startswith(prefix + "/")
                   for prefix in ignored):
                return True
            # Megatron training saves persist trainer rerun-state bytes as
            # rerun_state_machine_state/shard_<r>_<world> — trainer metadata,
            # not model tensors. Accept it like the other ignored prefixes.
            return key.startswith("rerun_state_machine_state/")

        unknown = []
        for key, entry in self.metadata.state_dict_metadata.items():
            if self.skip_mtp and is_mcore_mtp_key(key):
                continue
            if key not in self.groups:
                if not is_extra_state(key) and not _is_ignored(key):
                    unknown.append(key)
                continue
            if not isinstance(entry, TensorStorageMetadata):
                raise MappingError(f"Expected DCP tensor: {key}")
        if unknown or self.groups.keys() - self.metadata.state_dict_metadata.keys():
            raise MappingError(f"DCP schema differs from dsv4-global-v1: unknown={unknown[:20]}, "
                               f"missing={sorted(self.groups.keys() - self.metadata.state_dict_metadata.keys())[:20]}")
        file_sizes = {}
        for key, pieces in self.groups.items():
            entry = self.metadata.state_dict_metadata[key]
            expected, actual = pieces[0].global_shape, tuple(entry.size)
            if not mcore_shape_matches(key, actual, expected):
                raise MappingError(f"DCP global shape mismatch: {key}: {actual} != {expected}")
            dtype = entry.properties.dtype
            if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64):
                raise MappingError(f"Unsupported raw DCP precision {dtype}: {key}; "
                                   "this adapter requires standard dense model tensors")
            integer = dtype in (torch.int32, torch.int64)
            if integer != all(p.hf.endswith("tid2eid") for p in pieces):
                raise MappingError(f"Invalid DCP weight/router dtype: {key}: {dtype}")
            validate_regions(key, actual, [(tuple(c.offsets), tuple(c.sizes)) for c in entry.chunks])
            for i, chunk in enumerate(entry.chunks):
                index = MetadataIndex(key, chunk.offsets, i)
                info = self.metadata.storage_data.get(index)
                if info is None:
                    raise MappingError(f"Missing DCP storage entry: {index}")
                path = (self.directory / info.relative_path).resolve()
                if path not in file_sizes:
                    if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
                        raise MappingError(f"Missing or external DCP shard: {info.relative_path}")
                    file_sizes[path] = path.stat().st_size
                if info.offset < 0 or info.length <= 0 or info.offset + info.length > file_sizes[path]:
                    raise MappingError(f"Invalid DCP byte range: {index}")

    def _load_qt(self, index):
        """Load one serialized TE Float8BlockwiseQTensor directly from its blob."""
        from .fp8_native import register_native_fp8_safe_globals
        register_native_fp8_safe_globals()  # idempotent; needed for weights_only unpickling
        info = self.metadata.storage_data.get(index)
        if info is None:
            raise MappingError(f"Missing DCP storage entry: {index}")
        with (self.directory / info.relative_path).open("rb") as stream:
            stream.seek(info.offset)
            payload = stream.read(info.length)
        obj = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
        if type(obj).__name__ != "Float8BlockwiseQTensor":
            raise MappingError(f"Native quantized blob holds unexpected object: {index}")
        return obj

    def _read_native_piece(self, piece):
        """Read one piece of a native quantized key as raw payload + scales.

        The qt stores the raw uint8 payload (2-D rows x cols, 3-D expert
        chunks keep a 2-D physical view) and FP32 scales per 128x128 block.
        Piece overlap math mirrors the dense path; multi-part pieces must
        share one column range and are concatenated along rows.
        """
        entry = self.metadata.state_dict_metadata[piece.key]
        parts = []
        covered = 0
        for i, chunk in enumerate(entry.chunks):
            start = tuple(max(a, b) for a, b in zip(piece.offset, chunk.offsets))
            end = tuple(min(a + n, b + m) for a, n, b, m in zip(
                piece.offset, piece.shape, chunk.offsets, chunk.sizes))
            lengths = tuple(b - a for a, b in zip(start, end))
            if any(n <= 0 for n in lengths):
                continue
            covered += math.prod(lengths)
            qt = self._load_qt(MetadataIndex(piece.key, chunk.offsets, i))
            data = qt._rowwise_data
            if data.ndim > 2:
                data = data.reshape(-1, data.shape[-1])
            scale = qt._rowwise_scale_inv
            if data.dtype != torch.uint8 or scale.dtype != torch.float32:
                raise MappingError(
                    f"Unexpected native payload dtypes: {piece.key}: {data.dtype}/{scale.dtype}")
            local = tuple(a - b for a, b in zip(start, chunk.offsets))
            if len(lengths) == 3:
                if lengths[0] != chunk.sizes[0]:
                    raise MappingError(
                        f"Native expert axis cannot be split across ranks: {piece.key}")
                lo = local[0] * chunk.sizes[1] + local[1]
                hi = lo + lengths[1]
                c0, c1 = local[2], local[2] + lengths[2]
            else:
                lo = local[0]
                hi = lo + lengths[0]
                c0, c1 = local[1], local[1] + lengths[1]
            if lo % 128 or c0 % 128:
                raise MappingError(
                    f"Native quantized region must be 128-aligned: {piece.key}: rows {lo}, cols {c0}")
            part_data = data[lo:hi, c0:c1].contiguous()
            part_scale = scale[lo // 128:hi // 128, c0 // 128:c1 // 128].contiguous()
            parts.append((start[0], part_data, part_scale))
        if covered != math.prod(piece.shape):
            raise MappingError(f"Incomplete read coverage for {piece.hf}")
        rows, cols = piece.shape[-2:]
        expected = (math.ceil(rows / 128), math.ceil(cols / 128))
        if len(parts) == 1:
            _, data, scale_inv = parts[0]
            if tuple(scale_inv.shape) != expected:
                raise MappingError(f"Native scale shape mismatch: {piece.hf}: {tuple(scale_inv.shape)} != {expected}")
            return NativeQuantizedWeight("fp8", data, scale_inv)
        if any(part[2].shape[1] != parts[0][2].shape[1] for part in parts[1:]):
            raise MappingError(f"Native pieces of {piece.hf} disagree on column range")
        parts.sort(key=lambda part: part[0])
        data = torch.cat([part[1] for part in parts], dim=0).contiguous()
        scale_inv = torch.cat([part[2] for part in parts], dim=0).contiguous()
        if tuple(scale_inv.shape) != expected:
            raise MappingError(f"Native scale shape mismatch: {piece.hf}: {tuple(scale_inv.shape)} != {expected}")
        return NativeQuantizedWeight("fp8", data, scale_inv)

    def _read_piece(self, piece):
        if piece.key in self.native_keys:
            return self._read_native_piece(piece)
        entry = self.metadata.state_dict_metadata[piece.key]
        tensor = torch.empty(piece.shape, dtype=entry.properties.dtype, device="cpu")
        requests, covered = [], 0
        for i, chunk in enumerate(entry.chunks):
            start = tuple(max(a, b) for a, b in zip(piece.offset, chunk.offsets))
            end = tuple(min(a + n, b + m) for a, n, b, m in zip(
                piece.offset, piece.shape, chunk.offsets, chunk.sizes))
            lengths = tuple(b - a for a, b in zip(start, end))
            if any(n <= 0 for n in lengths):
                continue
            covered += math.prod(lengths)
            requests.append(ReadItem(
                type=LoadItemType.TENSOR, dest_index=MetadataIndex(piece.hf),
                dest_offsets=torch.Size(a - b for a, b in zip(start, piece.offset)),
                storage_index=MetadataIndex(piece.key, chunk.offsets, i),
                storage_offsets=torch.Size(a - b for a, b in zip(start, chunk.offsets)),
                lengths=torch.Size(lengths),
            ))
        if covered != math.prod(piece.shape):
            raise MappingError(f"Incomplete read coverage for {piece.hf}")
        plan = self.reader.prepare_local_plan(LoadPlan(requests))
        plan = self.reader.prepare_global_plan([plan])[0]
        self.reader.read_data(plan, _ReadDestination(tensor)).wait()
        return tensor

    def tensors(self):
        for name, pieces in self.by_hf.items():
            if len(pieces) == 1:
                tensor = self._read_piece(pieces[0])
                if not isinstance(tensor, NativeQuantizedWeight):
                    tensor = tensor.reshape(self.profile.shapes[name])
                else:
                    if tuple(tensor.shape) != tuple(self.profile.shapes[name]):
                        raise MappingError(
                            f"Native quantized piece shape mismatch: {name}: "
                            f"{tuple(tensor.shape)} != {tuple(self.profile.shapes[name])}")
            else:
                pieces = sorted(pieces, key=lambda p: p.hf_index)
                results = [self._read_piece(piece) for piece in pieces]
                if all(isinstance(result, NativeQuantizedWeight) for result in results):
                    data = torch.cat([result.data for result in results], dim=0).contiguous()
                    scale_inv = torch.cat([result.scale_inv for result in results], dim=0).contiguous()
                    tensor = NativeQuantizedWeight("fp8", data, scale_inv)
                elif any(isinstance(result, NativeQuantizedWeight) for result in results):
                    raise MappingError(f"Mixed dense/quantized pieces for {name}")
                else:
                    tensor = torch.cat([result.reshape(-1) for result in results])
                    tensor = tensor.reshape(self.profile.shapes[name])
            self.used.add(name)
            yield name, tensor
            del tensor
        if self.used != set(self.profile.shapes):
            raise MappingError("Incomplete HF export coverage")

    def report(self):
        files = {info.relative_path for info in self.metadata.storage_data.values()}
        return {"source_storage_files": len(files),
                "source_tensor_chunks": sum(len(self.metadata.state_dict_metadata[k].chunks) for k in self.groups),
                "topology_resolution": "global key plus chunk offsets; source rank numbering is not required"}
