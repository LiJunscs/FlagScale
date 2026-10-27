# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Original HF payloads alongside a normal model-only MCore checkpoint.

The archive is not TE quantizer state and is never loaded by MCore training.
Explicit restoration requires unchanged decoded model bytes; no hashes are used.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .hf_io import RawTensor, SafetensorReader, ShardedSafetensorWriter
from .quantization import DENSE_DTYPES, SCALE_SUFFIXES, PrecisionError, PrecisionSource


ARCHIVE_DIRECTORY = "hf_quantized"


def _specs(source):
    return json.loads(json.dumps({key: asdict(spec) for key, spec in sorted(source.specs.items())
                                  if spec.kind != "dense"}))


def save_quantized_archive(source, config, iteration_dir, max_shard_size):
    model_dtype = next((name for name, dtype in DENSE_DTYPES.items() if dtype == source.dtype), None)
    if model_dtype is None:
        raise PrecisionError("Original quantization archives require a supported decoded model dtype")
    specs = _specs(source)
    if not specs:
        return {"quantized_tensor_count": 0, "stored_bytes": 0, "directory": None}
    directory = Path(iteration_dir) / ARCHIVE_DIRECTORY
    directory.mkdir(exist_ok=False)
    writer = ShardedSafetensorWriter(directory, max_shard_size)
    size = 0
    for spec in specs.values():
        for name in (spec["name"], spec["scale"]):
            writer.add(name, RawTensor(source.reader, name))
            size += source.reader.tensor_storage(name)[2]
    writer.finish()
    manifest = {"schema_version": 2, "kind": "hf_original_quantized_payloads",
                "model_dtype": model_dtype, "hf_config": config, "specs": specs}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"quantized_tensor_count": len(specs), "stored_bytes": size,
            "directory": ARCHIVE_DIRECTORY}


class QuantizedArchive:
    def __init__(self, iteration_dir, target_source, *, source_factory=PrecisionSource):
        directory = Path(iteration_dir) / ARCHIVE_DIRECTORY
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise PrecisionError("--restore-original-quantization requires hf_quantized/manifest.json "
                                 "in the source iteration; import the quantized HF source first")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise PrecisionError(f"Duplicate archive manifest field: {key}")
                result[key] = value
            return result

        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 2
                or manifest.get("kind") != "hf_original_quantized_payloads"
                or manifest.get("model_dtype") not in DENSE_DTYPES):
            raise PrecisionError("Unsupported original HF quantization archive")
        self.config = manifest["hf_config"]
        self.source = source_factory(SafetensorReader(directory), self.config,
                                      dtype=DENSE_DTYPES[manifest["model_dtype"]], device="cpu")
        expected = _specs(self.source)
        if (not expected or manifest.get("specs") != expected
                or self.source.keys != set(expected)):
            raise PrecisionError("Original quantization archive differs from its manifest")
        self.output_specs = {}
        for name, spec in self.source.specs.items():
            target = target_source.specs.get(name)
            if target is None or target.shape != spec.shape:
                raise PrecisionError(f"Original quantization archive differs from target names/shapes: {name}; "
                                     "omit --restore-original-quantization to export current source storage")
            # The reference supplies names; the archive owns packed dtype,
            # scale layout and quantization config.
            suffix = next(suffix for suffix in SCALE_SUFFIXES if spec.scale.endswith(suffix))
            self.output_specs[name] = replace(spec, name=target.name,
                                              scale=target.name.removesuffix(".weight") + suffix)
        self.restored = set()

    def restore(self, name, tensor):
        if name in self.restored:
            raise PrecisionError(f"Duplicate original payload restoration: {name}")
        spec = self.source.specs[name]
        if tensor.dtype != self.source.dtype or tuple(tensor.shape) != spec.shape:
            raise PrecisionError(f"restore requires unchanged model dtype/shape ({self.source.dtype}): {name}")
        original = self.source.get_tensor(name)
        # Byte views distinguish signed zero; never round current training
        # weights to pass verification. Compare bounded slices.
        current_bytes = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        original_bytes = original.reshape(-1).view(torch.uint8)
        for start in range(0, original_bytes.numel(), 8 * 1024**2):
            if not torch.equal(current_bytes[start:start + 8 * 1024**2],
                               original_bytes[start:start + 8 * 1024**2]):
                raise PrecisionError(f"MCore model tensor differs from archived HF dequantization: {name}; "
                                     "omit --restore-original-quantization to export current source storage")
        self.restored.add(name)
        return RawTensor(self.source.reader, spec.name), RawTensor(self.source.reader, spec.scale)

    def validate_consumed(self):
        self.source.validate_consumed()
        if self.restored != self.source.keys:
            raise PrecisionError("Incomplete original quantization restoration")

    def report(self):
        return {"directory": ARCHIVE_DIRECTORY, "restored_quantized_tensor_count": len(self.restored),
                "verification": "Decoded model dtype, shape and bytes; all restored tensors checked"}
