# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Instantiate known DSv4 shapes with serialized names from an existing index.

An HF reference supplies names and (optionally) storage types, never dimensions
for a different MCore model. Unknown module types are not inferred from shapes.
"""
from __future__ import annotations

import re
from dataclasses import replace

from .hf_config import derive_compress_ratios
from . import MappingError
from .precision import SCALE_SUFFIXES, TensorSpec, canonical_hf_name
from .registry import _expected_shapes, active_rules


def key_pattern(name):
    return re.sub(r"(?<=\.)\d+(?=\.)", "{n}", name)


def _renumber(original, canonical, wanted):
    old = re.findall(r"(?<=\.)\d+(?=\.)", canonical)
    new = re.findall(r"(?<=\.)\d+(?=\.)", wanted)
    if len(old) != len(new):
        raise MappingError(f"Cannot expand serialized key {original} to {wanted}")
    numbers = iter(new)
    return re.sub(r"(?<=\.)\d+(?=\.)", lambda _: next(numbers), original)


class SchemaSource:
    def __init__(self, specs):
        self.specs = specs

    @property
    def keys(self):
        return set(self.specs)

    @property
    def low_precision_kinds(self):
        return {spec.kind for spec in self.specs.values() if spec.kind != "dense"}


def index_specs(weight_map):
    specs, paired = {}, set()
    physical = set(weight_map)
    scales = {name for name in physical if name.endswith(SCALE_SUFFIXES)}
    for name in sorted(physical - scales):
        canonical = canonical_hf_name(name)
        if canonical in specs:
            raise MappingError(f"Duplicate HF semantic name: {canonical}")
        candidates = [name.removesuffix(".weight") + suffix for suffix in SCALE_SUFFIXES]
        matches = [candidate for candidate in candidates if candidate in scales]
        if len(matches) > 1:
            raise MappingError(f"Multiple scale keys for {name}")
        paired.update(matches)
        # Index-only metadata cannot establish a storage dtype. This placeholder
        # is used only for keys/shapes and dense export, never to decode payloads.
        specs[canonical] = TensorSpec(name, (), "I64" if canonical.endswith("tid2eid") else "BF16")
    if paired != scales:
        raise MappingError(f"Orphan scale keys: {sorted(scales - paired)[:20]}")
    return specs


def naming_specs(specs):
    """Reference precision does not constrain an MCore source or new layers."""
    return {name: replace(spec, dtype="I64" if name.endswith("tid2eid") else "BF16",
                          kind="dense", scale=None, scale_dtype=None)
            for name, spec in specs.items()}


def with_storage_dtypes(source, dtypes, *, native=None, preserved=None):
    """Bind logical output names to actual DCP dtypes, including generated bias.

    `native` optionally maps HF names to their original quantized HF template
    spec. Keys present there were stored as native quantized payloads in the
    MCore checkpoint (TE Float8BlockwiseQTensor) and keep that storage instead
    of a dense dtype: FP8-origin weights keep their original payload/scale
    layout, MXFP4-origin weights are re-stored as FP8 E4M3 with FP32 block
    scales (the mcore->hf direction never decodes MXFP4).

    `preserved` comes from the actual checkpoint's portable storage descriptor,
    retaining FP8/MXFP4 and scale dtypes while rebinding the output HF names.
    """
    codes = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32",
             "torch.int32": "I32", "torch.int64": "I64"}
    native, preserved = native or {}, preserved or {}
    if source.keys != set(dtypes):
        raise MappingError("Actual source dtype coverage differs from mapped HF targets")
    specs = {}
    for name, spec in source.specs.items():
        if name in preserved and preserved[name].kind != "dense":
            origin = preserved[name]
            if origin.kind not in {"fp8", "mxfp4"} or origin.shape != spec.shape:
                raise MappingError(f"Preserved source encoding/shape differs from target: {name}")
            suffix = next((suffix for suffix in SCALE_SUFFIXES if origin.scale.endswith(suffix)), None)
            if suffix is None:
                raise MappingError(f"Unknown preserved scale name: {origin.scale}")
            specs[name] = replace(spec, dtype=origin.dtype, kind=origin.kind,
                                  scale=spec.name.removesuffix(".weight") + suffix,
                                  scale_dtype=origin.scale_dtype)
        elif name in native:
            origin = native[name]
            if origin.kind not in ("fp8", "mxfp4"):
                raise MappingError(
                    f"Unsupported native quantized storage for {name}: kind={origin.kind}")
            if origin.kind == "fp8":
                specs[name] = replace(spec, dtype="F8_E4M3", kind="fp8",
                                      scale=origin.scale, scale_dtype=origin.scale_dtype)
            else:
                specs[name] = replace(spec, dtype="F8_E4M3", kind="fp8",
                                      scale=name.removesuffix(".weight") + ".weight_scale_inv",
                                      scale_dtype="F32")
        else:
            specs[name] = replace(spec, dtype=codes[str(dtypes[name])], kind="dense",
                                  scale=None, scale_dtype=None)
    return SchemaSource(specs)


def instantiate_schema(config, reference_specs, rules, *, exact=False):
    specs = {}
    patterns = {}
    for canonical, spec in reference_specs.items():
        patterns.setdefault(key_pattern(canonical), []).append((canonical, spec))
    for context, rule in active_rules(config, derive_compress_ratios(config), rules):
        _, prefix, _, _, ratio, _ = context
        shapes = _expected_shapes(rule["id"], config, ratio)
        if shapes is None:
            raise MappingError(f"No shape formula for rule {rule['id']}; add a semantic operator first")
        experts = int(config["n_routed_experts"]) if rule["op"].startswith("expert") else 1
        for expert in range(experts):
            names = [canonical_hf_name(prefix + key.format(expert=expert)) for key in rule["hf"]]
            for name, shape in zip(names, shapes):
                entry = reference_specs.get(name)
                if entry is None and not exact:
                    candidates = patterns.get(key_pattern(name), [])
                    # A reference with per-layer mixed storage is ambiguous when
                    # expanding to new layers. Require explicit headers/schema.
                    signatures = {(key_pattern(spec.name), spec.dtype, spec.kind,
                                   key_pattern(spec.scale) if spec.scale else None, spec.scale_dtype)
                                  for _, spec in candidates}
                    if len(signatures) > 1:
                        raise MappingError(f"Ambiguous storage/name templates for {name}")
                    if candidates:
                        canonical, sample = candidates[0]
                        entry = replace(sample, name=_renumber(sample.name, canonical, name),
                                        scale=_renumber(sample.scale, canonical, name) if sample.scale else None)
                if entry is None:
                    if rule.get("optional"):
                        continue
                    raise MappingError(f"Existing HF index has no key template for {name}")
                if name in specs:
                    raise MappingError(f"Overlapping mapping rules for {name}")
                specs[name] = replace(entry, shape=tuple(shape))
    if exact and set(specs) != set(reference_specs):
        raise MappingError(f"Unmapped existing HF keys: {sorted(set(reference_specs) - set(specs))[:20]}")
    return SchemaSource(specs)


def check_mcore_schema(profile, schema):
    if profile.keys != set(schema):
        raise MappingError(f"Existing MCore keys differ: missing={sorted(profile.keys - set(schema))[:20]}, "
                           f"unknown={sorted(set(schema) - profile.keys)[:20]}")
    for piece in profile.pieces:
        shape = tuple(schema[piece.key]["shape"])
        expected = piece.global_shape
        if not mcore_shape_matches(piece.key, shape, expected):
            raise MappingError(f"MCore args/metadata disagree for {piece.key}: {shape} != {expected}")
        dtype = schema[piece.key]["dtype"]
        allowed = {"torch.int32", "torch.int64"} if piece.hf.endswith("tid2eid") else {
            "torch.bfloat16", "torch.float16", "torch.float32"}
        if dtype not in allowed:
            raise MappingError(f"Unsupported native DCP storage dtype: {piece.key}: {dtype}")


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
