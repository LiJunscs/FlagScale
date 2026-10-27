# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile executable mappings, then feed the existing streaming storage backend."""
from __future__ import annotations

import fnmatch
from dataclasses import asdict

import torch

from . import MappingError
from .mapping import expand_rules
from .operators import transform
from .plan import Piece, Profile
from .precision import DENSE_DTYPES, HEADER_DTYPES, PrecisionPolicy, TensorSpec

STORAGE_DTYPES = {**HEADER_DTYPES, "I32": torch.int32, "I64": torch.int64}
DTYPE_CODES = {str(value): key for key, value in STORAGE_DTYPES.items()}


def converted_dtype(code, requested):
    if code not in STORAGE_DTYPES:
        raise MappingError(f"Table mappings currently require dense storage, got {code}; "
                           "decode the source encoding before applying tensor layouts")
    if requested == "auto" or code in ("F32", "I32", "I64"):
        return code
    return DTYPE_CODES[str(DENSE_DTYPES[requested])]


def compile_mapping(mapping, specs, configs, *, forward, dtype="auto"):
    source_side, target_side = ("hf", "mcore") if forward else ("mcore", "hf")
    patterns = mapping.get("ignore", {}).get(source_side, [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise MappingError("ignore.hf / ignore.mcore must be lists of glob patterns")
    ignored = {name for name in specs if any(fnmatch.fnmatchcase(name, p) for p in patterns)}
    available = specs.keys() - ignored
    nodes, outputs, used = [], {}, set()
    for row in expand_rules(mapping, available, configs, forward):
        names, targets = row[source_side], row[target_side]
        missing = set(names) - available
        if missing:
            raise MappingError(f"Rule {row['id']} has missing inputs: {sorted(missing)}")
        codes = [converted_dtype(specs[name].dtype, dtype) for name in names]
        inputs = [torch.empty(specs[name].shape, device="meta", dtype=STORAGE_DTYPES[code])
                  for name, code in zip(names, codes)]
        result = transform(inputs, row.get("op", "identity"), row["params"], forward)
        if len(result) != len(targets):
            raise MappingError(f"Rule {row['id']} produces {len(result)} tensors for {len(targets)} targets")
        extras = row.get("extra_state", [])
        if not isinstance(extras, list) or any(not isinstance(x, str) for x in extras):
            raise MappingError("extra_state must be a list of MCore storage keys")
        if len(extras) > len(row["mcore"]):
            raise MappingError("At most one extra-state key per MCore tensor is supported")
        node = {"id": row["id"], "inputs": names, "outputs": targets,
                "op": row.get("op", "identity"), "params": row["params"],
                "forward": forward, "input_dtypes": codes,
                "extra_state": extras if forward else []}
        for name, tensor in zip(targets, result):
            if name in outputs:
                raise MappingError(f"Multiple rules write target {name}")
            if not tensor.shape or any(n <= 0 for n in tensor.shape):
                raise MappingError(f"Target {name} needs a nonempty shape; reshape scalars explicitly")
            outputs[name] = TensorSpec(name, tuple(tensor.shape), DTYPE_CODES[str(tensor.dtype)])
        nodes.append(node)
        used.update(names)
    if available - used:
        raise MappingError(f"Unmapped source tensors (use explicit ignore rules): {sorted(available - used)[:20]}")
    if not outputs:
        raise MappingError("Mapping produced no tensors")
    return {"nodes": nodes, "outputs": outputs, "ignored": sorted(ignored)}


def storage_profile(specs, name="mapping", nodes=()):
    extra = {}
    for node in nodes:
        extra.update(zip(node["outputs"], node["extra_state"]))
    pieces = [Piece(key, key, key, spec.shape, (0,) * len(spec.shape), spec.shape,
                    extra_key=extra.get(key)) for key, spec in specs.items()]
    return Profile(pieces, {key: spec.shape for key, spec in specs.items()}, name)


class MappedSource:
    """Materialize one operation group at a time, without a whole-model cache."""
    def __init__(self, plan, get_tensor):
        self.specs = plan["outputs"]
        self.get_input = get_tensor
        self.nodes = {key: node for node in plan["nodes"] for key in node["outputs"]}
        self.consumed = set()

    def get_tensor(self, name):
        node = self.nodes[name]
        inputs = [self.get_input(key).to(dtype=STORAGE_DTYPES[code])
                  for key, code in zip(node["inputs"], node["input_dtypes"])]
        outputs = transform(inputs, node["op"], node["params"], node["forward"])
        result = outputs[node["outputs"].index(name)]
        spec = self.specs[name]
        if tuple(result.shape) != spec.shape or result.dtype != STORAGE_DTYPES[spec.dtype]:
            raise MappingError(f"Execution differs from prepared metadata: {name}")
        self.consumed.add(name)
        return result

    def validate_consumed(self):
        if self.consumed != self.specs.keys():
            raise MappingError("Not all mapped tensors were written")

    def tensors(self):
        for name in self.specs:
            tensor = self.get_tensor(name)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise MappingError(f"Non-finite/overflowing output: {name}")
            yield name, tensor
            del tensor
        self.validate_consumed()


def dense_policy(specs):
    sizes = {code: sum(torch.Size(s.shape).numel() for s in specs.values() if s.dtype == code)
             for code in HEADER_DTYPES}
    code = max(sizes, key=sizes.get)
    dtype = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}[code]
    return PrecisionPolicy(dtype, "cpu", False, "Execute declared layouts; model semantics are not validated")


def plan_report(plan):
    return {**plan, "outputs": {name: asdict(spec) for name, spec in plan["outputs"].items()}}
