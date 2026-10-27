# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Safetensors index/header inspection with no tensor or model library imports."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from tools.checkpoint.hf_mcore_dist.safetensors_index import read_safetensors_index


class HuggingFaceIOError(RuntimeError):
    pass


class SafetensorMetadataReader:
    """Read metadata and file sizes only; never read tensor payloads."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise HuggingFaceIOError(f"HF model directory does not exist: {self.model_dir}")
        index_path = self.model_dir / "model.safetensors.index.json"
        self.index_path = None
        self.index_metadata = None
        if index_path.is_file():
            index = read_safetensors_index(index_path)
            self.index_path = index_path
            self.index_metadata = index.get("metadata")
            self._weight_map = dict(index["weight_map"])
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if len(files) != 1:
                raise HuggingFaceIOError(
                    "Expected model.safetensors.index.json or exactly one safetensors file in "
                    f"{self.model_dir}"
                )
            if not files[0].resolve().is_relative_to(self.model_dir):
                raise HuggingFaceIOError(f"Shard escapes HF directory: {files[0].name}")
            header, _ = self._read_header(files[0])
            self._weight_map = {key: files[0].name for key in header}
        for filename in set(self._weight_map.values()):
            if not (self.model_dir / filename).resolve().is_relative_to(self.model_dir):
                raise HuggingFaceIOError(f"Shard escapes HF directory: {filename}")
        missing_files = sorted(filename for filename in set(self._weight_map.values())
                               if not (self.model_dir / filename).is_file())
        if missing_files:
            raise HuggingFaceIOError(f"Missing safetensors shards: {missing_files[:20]}")
        self._metadata = {}
        self._data_starts = {}
        for filename in sorted(set(self._weight_map.values())):
            header, start = self._read_header(self.model_dir / filename)
            self._data_starts[filename] = start
            for name, entry in header.items():
                if self._weight_map.get(name) != filename:
                    raise HuggingFaceIOError(f"Unindexed or duplicate tensor: {name}")
                self._metadata[name] = entry
        missing = self.keys - self._metadata.keys()
        if missing:
            raise HuggingFaceIOError(f"Index keys absent from shards: {sorted(missing)[:20]}")

    @staticmethod
    def _read_header(path: Path) -> tuple[dict, int]:
        # Preserve storage dtype names even when installed tensor libraries do
        # not implement them. Numerical interpretation belongs to precision.py.
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise HuggingFaceIOError(f"Duplicate safetensors header key: {key}")
                result[key] = value
            return result

        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise HuggingFaceIOError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            file_size = path.stat().st_size
            if length > 100_000_000 or length + 8 > file_size:
                raise HuggingFaceIOError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length), object_pairs_hook=unique_object)
        if not isinstance(header, dict):
            raise HuggingFaceIOError(f"Safetensors header must be an object: {path}")
        header.pop("__metadata__", None)
        data_size = file_size - 8 - length
        spans = []
        byte_sizes = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8,
                      "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1,
                      "F8_E8M0": 1, "F8_E8M0FNU": 1}
        for name, entry in header.items():
            if not isinstance(entry, dict):
                raise HuggingFaceIOError(f"Invalid tensor metadata: {name}")
            offsets, shape, dtype = (entry.get(field) for field in ("data_offsets", "shape", "dtype"))
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise HuggingFaceIOError(f"Invalid data offsets for {name}")
            begin, end = offsets
            if not (0 <= begin <= end <= data_size):
                raise HuggingFaceIOError(f"Invalid data offsets for {name}")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise HuggingFaceIOError(f"Invalid shape for {name}")
            if not isinstance(dtype, str) or not dtype:
                raise HuggingFaceIOError(f"Invalid storage dtype for {name}")
            size = byte_sizes.get(dtype)
            if size is not None and end - begin != size * math.prod(shape):
                raise HuggingFaceIOError(f"Invalid tensor byte count: {name}")
            if end > begin:
                spans.append((begin, end, name))
        cursor = 0
        for begin, end, name in sorted(spans):
            if begin != cursor:
                raise HuggingFaceIOError(f"Overlapping or incomplete safetensors storage: {name}")
            cursor = end
        if cursor != data_size:
            raise HuggingFaceIOError(f"Unindexed safetensors storage: {path}")
        return header, 8 + length

    @property
    def keys(self) -> set[str]:
        return set(self._weight_map)

    @property
    def weight_map(self) -> dict[str, str]:
        return dict(self._weight_map)

    def tensor_dtypes(self) -> dict[str, str]:
        return {name: entry["dtype"].upper() for name, entry in self._metadata.items()}

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(entry["shape"]) for name, entry in self._metadata.items()}

    def tensor_storage(self, name: str) -> tuple[str, tuple[int, ...], int]:
        entry = self._metadata[name]
        begin, end = entry["data_offsets"]
        return entry["dtype"], tuple(entry["shape"]), end - begin
