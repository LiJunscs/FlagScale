# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Inspect existing config/index/common/DCP metadata without model construction."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


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
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise MappingError("HF config must be an object")
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
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    directory = resolve_iteration_dir(path)
    # Our model-only outputs have standalone common.pt; avoid importing the
    # training runtime just to read them. Embedded common state uses MCore's reader.
    if (directory / "common.pt").is_file():
        import torch
        common = torch.load(directory / "common.pt", map_location="cpu", weights_only=False)
    else:
        from megatron.core.dist_checkpointing.serialization import load_common_state_dict
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
        raise MappingError("Only non-homogeneous, non-singleton DCP keys are supported")
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
        raise MappingError("No MCore model tensor metadata was found")
    return {"directory": str(directory), "args": values, "omitted_runtime_args": omitted,
            "model_tensors": schema, "iteration": common.get("iteration")}
