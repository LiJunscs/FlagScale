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
from .precision import DENSE_DTYPES, HEADER_DTYPES, TensorSpec

STORAGE_DTYPES = {**HEADER_DTYPES, "I32": torch.int32, "I64": torch.int64}
DTYPE_CODES = {str(value): key for key, value in STORAGE_DTYPES.items()}


def converted_dtype(code, requested):
    if code not in STORAGE_DTYPES:
        raise MappingError(f"Table mappings currently require dense storage, got {code}; "
                           "decode the source encoding before applying tensor layouts")
    if requested == "auto" or code in ("F32", "I32", "I64"):
        return code
    return DTYPE_CODES[str(DENSE_DTYPES[requested])]


def compile_mapping(mapping, specs, configs, *, forward, dtype="auto", preserve=False):
    source_side, target_side = ("hf", "mcore") if forward else ("mcore", "hf")
    patterns = mapping.get("ignore", {}).get(source_side, [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise MappingError("ignore.hf / ignore.mcore must be lists of glob patterns")
    zero_checks = [name for name in specs if any(fnmatch.fnmatchcase(name, p)
                   for p in mapping.get("zero_checks", {}).get(source_side, []))]
    ignored = {name for name in specs if any(fnmatch.fnmatchcase(name, p) for p in patterns)} | set(zero_checks)
    available = specs.keys() - ignored
    nodes, outputs, used = [], {}, set()
    for row in expand_rules(mapping, available, configs, forward):
        names, targets = row[source_side], row[target_side]
        if row.get("generate"):
            spec = row["generate"]
            if spec.get("side", "hf") != target_side:
                continue
            if names:
                raise MappingError("Generated tensors must not declare source names")
            result = [torch.empty(spec["shape"], device="meta", dtype=STORAGE_DTYPES[spec["dtype"]])
                      for _ in targets]
            codes = []
        else:
            result = None
        missing = set(names) - available
        if missing:
            raise MappingError(f"Rule {row['id']} has missing inputs: {sorted(missing)}")
        if result is None:
            codes = [converted_dtype(specs[name].dtype, dtype) for name in names]
            inputs = [torch.empty(specs[name].shape, device="meta",
                                  dtype=torch.float32 if preserve else STORAGE_DTYPES[code])
                      for name, code in zip(names, codes)]
            result = transform(inputs, row.get("op", "identity"), row["params"], forward)
        if row.get("export_slice") and not forward:
            if row.get("op", "identity") != "identity":
                raise MappingError("export_slice currently applies to identity rules")
            axis, size = row["export_slice"]["axis"], row["export_slice"]["size"]
            if type(size) is not int or size <= 0 or size > result[0].shape[axis]:
                raise MappingError("Invalid export_slice size")
            result = [result[0].narrow(axis, 0, size)]
        if len(result) != len(targets):
            raise MappingError(f"Rule {row['id']} produces {len(result)} tensors for {len(targets)} targets")
        extras = row.get("extra_state", [])
        if not isinstance(extras, list) or any(not isinstance(x, str) for x in extras):
            raise MappingError("extra_state must be a list of MCore storage keys")
        node = {"id": row["id"], "inputs": names, "outputs": targets,
                "op": row.get("op", "identity"), "params": row["params"],
                "forward": forward, "input_dtypes": codes,
                "extra_state": extras, "hf": row["hf"], "mcore": row["mcore"],
                "hf_shapes": [list(specs[n].shape) for n in names] if forward else [list(t.shape) for t in result],
                "mcore_shapes": [list(t.shape) for t in result] if forward else [list(specs[n].shape) for n in names],
                "export_slice": row.get("export_slice"), "generate": row.get("generate")}
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
    return {"nodes": nodes, "outputs": outputs, "ignored": sorted(ignored), "zero_checks": zero_checks}



def plan_report(plan):
    return {**plan, "outputs": {name: asdict(spec) for name, spec in plan["outputs"].items()}}
