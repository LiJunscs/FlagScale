# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile common layouts into HF fragments and global DCP rectangles."""
from . import MappingError
from .plan import Piece, Profile
from .streaming import validate_regions


def compile_regions(plan, name="mapping"):
    pieces, shapes, physical, extras = [], {}, {}, []
    for node in plan["nodes"]:
        if node.get("generate"):
            continue
        hf, mc = node["hf"], node["mcore"]
        hs, ms = node["hf_shapes"], node["mcore_shapes"]
        shapes.update(zip(hf, map(tuple, hs)))
        extras.extend(node["extra_state"])
        op, params = node["op"], node["params"]
        if node.get("export_slice") and not node["forward"]:
            physical[mc[0]] = tuple(ms[0])
            ms = hs  # Identity only: logical DCP coverage excludes explicitly cropped rows.

        def add(i, j, shape=None, offset=None, source_offset=None, source_shape=None, axes=None):
            shape = tuple(ms[j] if shape is None else shape)
            fragment = tuple(hs[i] if source_shape is None else source_shape)
            if source_offset is not None and not any(source_offset) and fragment == tuple(hs[i]):
                source_offset = None
            pieces.append(Piece(
                hf[i], mc[j], mc[j], shape, tuple(offset or [0] * len(shape)), tuple(ms[j]),
                source_offset=tuple(source_offset) if source_offset is not None else None,
                source_shape=fragment, axes=tuple(axes) if axes is not None else None))

        if op in ("identity", "reshape", "permute"):
            add(0, 0, axes=params.get("dims") if op == "permute" else None)
        elif op in ("concat", "swiglu"):
            axis = params.get("axis", 0) % len(ms[0])
            start = 0
            for i, shape in enumerate(hs):
                offset = [0] * len(shape)
                offset[axis] = start
                add(i, 0, shape, offset)
                start += shape[axis]
        elif op == "split":
            axis = params.get("axis", 0) % len(hs[0])
            start = 0
            for j, shape in enumerate(ms):
                offset = [0] * len(shape)
                offset[axis] = start
                add(0, j, source_offset=offset, source_shape=shape)
                start += shape[axis]
        elif op == "stack":
            axis = params.get("axis", 0) % len(ms[0])
            for i, shape in enumerate(hs):
                shape = list(shape)
                shape.insert(axis, 1)
                offset = [0] * len(shape)
                offset[axis] = i
                add(i, 0, shape, offset)
        elif op == "expert_swiglu":
            for i, shape in enumerate(hs):
                add(i, 0, (1, *shape), (i // 2, (i % 2) * shape[0], *([0] * (len(shape) - 1))))
        elif op == "qkv":
            groups = params["groups"]
            order = params.get("order", ["q", "k", "v"])
            sizes = [shape[0] // groups for shape in hs]
            for group in range(groups):
                target_start = group * sum(sizes)
                for label in order:
                    i = ["q", "k", "v"].index(label)
                    shape = (sizes[i], *hs[i][1:])
                    add(i, 0, shape, (target_start, *([0] * (len(shape) - 1))),
                        (group * sizes[i], *([0] * (len(shape) - 1))), shape)
                    target_start += sizes[i]
        else:
            raise MappingError(f"No region compiler for {op}")
    profile = Profile(pieces, shapes, name, extra_keys=tuple(sorted(set(extras))), physical_shapes=physical)
    # Reverse writes must cover each HF output once, including arbitrary split axes.
    if plan["nodes"] and not plan["nodes"][0]["forward"]:
        for key, shape in shapes.items():
            validate_regions(key, shape, [(p.source_offset or (0,) * len(shape), p.source_shape)
                                         for p in pieces if p.hf == key])
    return profile


def validate_preserve(profile, specs):
    """Raw encodings can move as whole matrices; arbitrary scale reblocking is not implicit."""
    for piece in profile.pieces:
        spec = specs[piece.hf]
        if spec.kind == "dense":
            continue
        if (piece.source_offset is not None or piece.axes is not None
                or piece.shape[-2:] != spec.shape[-2:]
                or len(piece.shape) not in (2, 3)
                or (len(piece.shape) == 3 and piece.shape[0] != 1)):
            raise MappingError(f"preserve cannot slice/permute quantized matrix {piece.hf}; use dense mode")
