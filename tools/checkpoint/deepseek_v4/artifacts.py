# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Inspect existing config/index/common/DCP metadata without model construction."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .hf_config import read_hf_config
from . import MappingError
from .safetensors_index import read_safetensors_index
from .world_one import require_world_one


def resolve_iteration_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if (path / ".metadata").is_file():
        _validate_checkpoint_dir(path)
        return path
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text(encoding="utf-8").strip()
        if value == "release":
            candidate = path / "release"
        else:
            candidate = path / f"iter_{int(value):07d}"
        if (candidate / ".metadata").is_file():
            _validate_checkpoint_dir(candidate)
            return candidate
        raise MappingError(f"Tracker points to an incomplete checkpoint: {candidate}")
    candidates = sorted(
        item for item in path.glob("iter_*")
        if (item / ".metadata").is_file() and (item / "metadata.json").is_file()
    )
    if candidates:
        _validate_checkpoint_dir(candidates[-1])
        return candidates[-1]
    raise MappingError(f"No MCore torch_dist checkpoint found under {path}")


def _validate_checkpoint_dir(path: Path) -> None:
    marker = path / "metadata.json"
    if not marker.is_file():
        raise MappingError(f"Missing MCore completion marker: {marker}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("sharded_backend") != "torch_dist":
        raise MappingError(f"Expected MCore torch_dist backend in {marker}")


def is_extra_state(name: str) -> bool:
    return re.search(r"(?:^|\.)_extra_state\d*(?:/|$)", name) is not None


def is_mcore_mtp_key(name: str) -> bool:
    return re.match(r"^mtp\.layers\.\d+\.", name) is not None


def mcore_shape_matches(key, stored_shape, logical_shape):
    """Allow trailing vocabulary padding only on known token-indexed matrices.

    MCore gives its hash router actual_vocab_size=padded_vocab_size. Like
    embedding/output weights, tid2eid may therefore have more rows than the
    explicit source vocab_size exported to HF. All other axes remain exact.
    """
    stored, logical = tuple(stored_shape), tuple(logical_shape)
    if stored == logical:
        return True
    vocab = key in {"embedding.word_embeddings.weight", "output_layer.weight"}
    vocab |= re.fullmatch(r"decoder\.layers\.\d+\.mlp\.router\.tid2eid", key) is not None
    return (vocab and len(stored) == len(logical) == 2
            and stored[1:] == logical[1:] and stored[0] >= logical[0])


def _plain(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    if callable(value) and getattr(value, "__module__", None) and getattr(value, "__qualname__", None):
        return {"callable": value.__module__ + "." + value.__qualname__}
    raise TypeError("Runtime-only configuration value")


def read_hf_artifacts(path):
    directory = Path(path).expanduser().resolve()
    config = read_hf_config(directory)
    index = read_safetensors_index(directory / "model.safetensors.index.json")
    return {"directory": str(directory), "config": config,
            "weight_map": dict(sorted(index["weight_map"].items())),
            "index_metadata": index.get("metadata")}


def json_args(args):
    """Keep all serializable saved defaults; report runtime objects explicitly."""
    values, omitted = {}, []
    for name, value in args.items():
        try:
            values[name] = _plain(value)
        except (TypeError, ValueError):
            omitted.append(name)
    return values, sorted(omitted)


def read_mcore_artifacts(path):
    require_world_one()
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    directory = resolve_iteration_dir(path)
    # Common (non-sharded) state is read with the official reader, which is
    # format-transparent: standalone common.pt (legacy torch save) or the
    # common_state ShardedObject inside the DCP payload (fork torch_dist
    # save). .distcp model/optimizer tensor payloads are not read here.
    from megatron.core.dist_checkpointing.serialization import (
        load_common_state_dict,
    )

    common = load_common_state_dict(str(directory))
    if not isinstance(common, dict) or "args" not in common:
        raise MappingError("Existing MCore checkpoint must contain common state args")
    saved = common["args"]
    raw_args = dict(saved) if isinstance(saved, dict) else dict(vars(saved))
    values, omitted = json_args(raw_args)
    content = common.get("content_metadata") or {}
    if not isinstance(content, dict):
        raise MappingError("MCore content_metadata must be a mapping")
    if content.get("singleton_local_shards", False) or content.get("non_homogeneous_layers") is False:
        raise MappingError("Only non-homogeneous, non-singleton DSv4 DCP keys are supported")
    metadata = FileSystemReader(directory).read_metadata()
    ignored = ("optimizer", "rng_state", "rerun_state_machine", "opt_param_scheduler",
               "common_state", "num_floating_point_operations_so_far")
    schema = {
        name: {"shape": list(entry.size), "dtype": str(entry.properties.dtype)}
        for name, entry in sorted(metadata.state_dict_metadata.items())
        if isinstance(entry, TensorStorageMetadata)
        and not is_extra_state(name)
        and not any(name == prefix or name.startswith(prefix + ".") or name.startswith(prefix + "/")
                    for prefix in ignored)
    }
    if not schema:
        raise MappingError("No standard DSv4 MCore model tensor metadata was found")
    return {"directory": str(directory), "args": values, "omitted_runtime_args": omitted,
            "model_tensors": schema, "iteration": common.get("iteration")}


if __name__ == "__main__":
    checkpoint = "outputs/DeepSeek-V4-Flash/tp4cp1ep8_fused/iter_0000004"
    artifacts = read_mcore_artifacts(checkpoint)
