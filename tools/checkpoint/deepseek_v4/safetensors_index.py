# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Read exported keys from an index alone, using only the standard library."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors index key: {key}")
        result[key] = value
    return result


def read_safetensors_index(path):
    """Validate weight_map without opening shards or inferring shape/precision."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    if not isinstance(payload, dict):
        raise ValueError("Safetensors index must be a JSON object")
    weights = payload.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Safetensors index requires a nonempty weight_map object")
    for name, shard in weights.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard:
            raise ValueError("Safetensors weight_map must map nonempty keys to shard filenames")
        posix = PurePosixPath(shard.replace("\\", "/"))
        windows = PureWindowsPath(shard)
        if (posix.is_absolute() or windows.drive or windows.root or ".." in posix.parts
                or posix.suffix != ".safetensors" or "\x00" in shard):
            raise ValueError(f"Invalid relative safetensors shard path: {shard!r}")
    return payload
