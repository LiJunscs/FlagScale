# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Common reversible layouts. Shape compatibility is not model equivalence."""
from __future__ import annotations

import torch

from . import MappingError


def transform(tensors, op, params, forward=True):
    allowed = {
        "identity": set(), "concat": {"axis", "sizes"}, "split": {"axis", "sizes"},
        "swiglu": set(), "stack": {"axis"}, "permute": {"dims"},
        "reshape": {"hf_shape", "mcore_shape"},
        "qkv": {"heads", "groups", "order"}, "expert_swiglu": set(),
    }
    if op not in allowed or not isinstance(params, dict) or params.keys() - allowed[op]:
        raise MappingError(f"Unsupported operator or parameters: {op}: {params}")
    if not tensors:
        raise MappingError("An operation needs input tensors")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise MappingError("Fusion requires one dtype; choose an explicit conversion dtype")

    def single():
        if len(tensors) != 1:
            raise MappingError(f"{op} requires exactly one tensor in this direction")
        return tensors[0]

    try:
        if op == "identity":
            return [single()]
        if op == "permute":
            tensor = single()
            dims = params["dims"]
            if sorted(dims) != list(range(tensor.ndim)):
                raise MappingError("dims must be a permutation of all axes")
            return [tensor.permute(dims if forward else [dims.index(i) for i in range(len(dims))])]
        if op == "reshape":
            return [single().reshape(params["mcore_shape" if forward else "hf_shape"])]
        if op in ("concat", "split", "swiglu"):
            join = forward != (op == "split")
            axis = params.get("axis", 0)
            if op == "swiglu":
                if join and (len(tensors) != 2 or tensors[0].shape != tensors[1].shape):
                    raise MappingError("swiglu requires two equally shaped tensors")
                if not join and single().shape[0] % 2:
                    raise MappingError("swiglu needs an even row count")
                sizes = [single().shape[0] // 2] * 2 if not join else None
            else:
                sizes = params.get("sizes")
                if not join and sizes is None:
                    raise MappingError(f"{op} reverse split requires explicit sizes")
            if sizes is not None:
                if not isinstance(sizes, list) or any(type(n) is not int or n <= 0 for n in sizes):
                    raise MappingError("Split sizes must be positive integers")
                if join and [t.shape[axis] for t in tensors] != sizes:
                    raise MappingError("Input dimensions differ from declared split sizes")
            return [torch.cat(tensors, dim=axis)] if join else list(torch.split(single(), sizes, dim=axis))
        if op == "stack":
            axis = params.get("axis", 0)
            return [torch.stack(tensors, dim=axis)] if forward else list(single().unbind(axis))
        if op == "expert_swiglu":
            if forward:
                if len(tensors) % 2:
                    raise MappingError("expert_swiglu requires gate/up pairs")
                return [torch.stack([transform(tensors[i:i + 2], "swiglu", {})[0]
                                     for i in range(0, len(tensors), 2)], dim=0)]
            return [part for expert in single().unbind(0)
                    for part in transform([expert], "swiglu", {}, False)]
        if op == "qkv":
            heads, groups = params["heads"], params["groups"]
            if (type(heads) is not int or type(groups) is not int
                    or heads <= 0 or groups <= 0 or heads % groups):
                raise MappingError("QKV heads must be a positive multiple of groups")
            order = params.get("order", ["q", "k", "v"])
            if sorted(order) != ["k", "q", "v"]:
                raise MappingError("QKV order must contain q, k, v exactly once")
            if forward:
                if len(tensors) != 3:
                    raise MappingError("qkv requires three tensors in HF order [q, k, v]")
                q, k, v = tensors
                if (q.shape[0] % heads or k.shape != v.shape or k.shape[0] % groups
                        or q.shape[0] // heads != k.shape[0] // groups):
                    raise MappingError("QKV dimensions cannot be grouped as requested")
                parts = dict(zip(("q", "k", "v"),
                                 [t.reshape(groups, t.shape[0] // groups, *t.shape[1:]) for t in tensors]))
                return [torch.cat([parts[key] for key in order], dim=1).flatten(0, 1)]
            tensor = single()
            width = heads + 2 * groups
            if tensor.shape[0] % width:
                raise MappingError("Fused QKV rows cannot be split as requested")
            dim = tensor.shape[0] // width
            sizes = {"q": heads // groups * dim, "k": dim, "v": dim}
            pieces = dict(zip(order, torch.split(
                tensor.reshape(groups, -1, *tensor.shape[1:]), [sizes[key] for key in order], dim=1)))
            return [pieces[key].flatten(0, 1) for key in ("q", "k", "v")]
    except (RuntimeError, ValueError, IndexError, KeyError, TypeError) as error:
        raise MappingError(f"Cannot execute {op}: {error}") from error
    raise MappingError(f"Unsupported operator: {op}")
