# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Minimal mapping files: whole-rule overrides, numeric templates and config aliases.

No model probing, architecture validation or evaluation of Python expressions.
An explicitly selected python:module:attribute can expose an existing repository's
mapping dict; export it to JSON to remove that import dependency.
"""
from __future__ import annotations

import importlib
import itertools
import json
import re
from copy import deepcopy
from pathlib import Path
from string import Formatter

from . import MappingError
from .config import resolve


def merge_rules(base, overrides):
    result = {}
    for rows in (base, overrides):
        seen = set()
        if not isinstance(rows, list):
            raise MappingError("rules must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise MappingError("Every rule requires a nonempty id")
            key = row["id"]
            if key in seen:
                raise MappingError(f"Duplicate rule id: {key}")
            seen.add(key)
            if row.get("disabled") is True:
                if set(row) != {"id", "disabled"} or key not in result:
                    raise MappingError(f"Invalid disabled rule: {key}")
                del result[key]
            else:
                result[key] = deepcopy(row)
    return list(result.values())


def load_mapping(location, overrides=(), *, _parents=(), _base=None):
    location = str(location)
    if location.startswith("python:"):
        _, module, attribute = location.split(":", 2)
        identity, directory = location, Path.cwd()
        if identity in _parents:
            raise MappingError(f"Cyclic mapping inheritance: {identity}")
        payload = deepcopy(getattr(importlib.import_module(module), attribute))
    else:
        path = Path(location).expanduser().resolve()
        identity, directory = str(path), path.parent
        if identity in _parents:
            raise MappingError(f"Cyclic mapping inheritance: {identity}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise MappingError("Mapping must be a schema_version: 1 object")
    allowed = {"schema_version", "extends", "profile", "rules", "config_map", "hf_config",
               "mcore_args", "ignore", "quantization", "naming", "adapter", "scopes", "zero_checks"}
    if payload.keys() - allowed:
        raise MappingError(f"Unknown mapping fields: {sorted(payload.keys() - allowed)}")
    result = deepcopy(_base) if _base is not None else {"schema_version": 1, "rules": []}
    if "extends" in payload:
        parent = payload["extends"]
        if not isinstance(parent, str):
            raise MappingError("extends must be one mapping path")
        if not parent.startswith("python:"):
            parent = str(directory / parent)
        parent_mapping = load_mapping(parent, _parents=(*_parents, identity))
        result = overlay(result, parent_mapping)
    result["rules"] = merge_rules(result["rules"], payload.get("rules", []))
    for key in ("config_map", "hf_config", "mcore_args", "ignore", "quantization", "naming", "scopes", "zero_checks"):
        value = payload.get(key, {})
        if not isinstance(value, dict):
            raise MappingError(f"{key} must be an object")
        result[key] = {**result.get(key, {}), **deepcopy(value)}
    for key in ("profile", "adapter"):
        if key in payload:
            result[key] = payload[key]
    for override in overrides:
        result = load_mapping(override, _base=result)
    return result


def overlay(base, patch):
    result = deepcopy(base)
    result["rules"] = merge_rules(base["rules"], patch["rules"])
    for key in ("config_map", "hf_config", "mcore_args", "ignore", "quantization", "naming", "scopes", "zero_checks"):
        result[key] = {**base.get(key, {}), **patch.get(key, {})}
    for key in ("profile", "adapter"):
        if key in patch:
            result[key] = patch[key]
    return result


def _pattern(template):
    pattern, seen = "", set()
    for literal, field, spec, conversion in Formatter().parse(template):
        pattern += re.escape(literal)
        if field is not None:
            if not field.isidentifier() or spec or conversion:
                raise MappingError(f"Only plain numeric placeholders are supported: {template}")
            pattern += f"(?P={field})" if field in seen else f"(?P<{field}>[0-9]+)"
            seen.add(field)
    return re.compile("^" + pattern + "$"), seen


def expand_rules(mapping, keys, configs, forward):
    """Enumerate layer ids from actual keys; only collected axes need a count."""
    source_side = "hf" if forward else "mcore"
    rows = []
    for rule in mapping["rules"]:
        if rule.get("scope") is not None and rule["scope"] not in mapping.get("scopes", {}):
            raise MappingError(f"Unknown scope: {rule['scope']}")
        scopes = mapping.get("scopes", {}).get(rule.get("scope"), [None])
        for scope in scopes:
            row = {k: v for k, v in rule.items() if k != "scope"}
            if scope is not None:
                for side in ("hf", "mcore"):
                    row[side] = [scope[side] + t for t in row[side]]
                row["id"] = rule["id"] + ":" + scope["hf"]
                row["extra_state"] = [scope["mcore"] + t for t in row.get("extra_state", [])]
                row["optional"] = row.get("optional", False) or scope.get("optional", False)
            rows.append(row)
    for raw in rows:
        allowed = {"id", "hf", "mcore", "op", "params", "collect", "optional", "extra_state", "export_slice", "generate"}
        if raw.keys() - allowed:
            raise MappingError(f"Unknown fields in rule {raw['id']}: {sorted(raw.keys() - allowed)}")
        if raw.get("generate"):
            row = {**raw, "params": {}, "generate": resolve(raw["generate"], configs)}
            yield row
            continue
        for side in ("hf", "mcore"):
            if (not isinstance(raw.get(side), list) or not raw[side]
                    or any(not isinstance(x, str) or not x for x in raw[side])):
                raise MappingError(f"Rule {raw['id']} requires nonempty {side} name templates")
        collect = resolve(raw.get("collect", {}), configs)
        if not isinstance(collect, dict) or any(
                not isinstance(key, str) or not key.isidentifier() or type(n) is not int or n <= 0
                for key, n in collect.items()):
            raise MappingError(f"Invalid collected axes: {raw['id']}")
        patterns = [_pattern(t)[0] for t in raw[source_side]]
        contexts = set()
        for key in keys:
            for pattern in patterns:
                match = pattern.fullmatch(key)
                if match:
                    contexts.add(tuple(sorted((k, v) for k, v in match.groupdict().items() if k not in collect)))
        if not contexts and not raw.get("optional", False):
            raise MappingError(f"Rule {raw['id']} matches no {source_side} tensors")
        for ctx in sorted(contexts, key=lambda ctx: tuple(int(v) for _, v in ctx)):
            row = {**raw, "params": resolve(raw.get("params", {}), configs)}
            for side in ("hf", "mcore"):
                names = []
                # Collected names are expert-major, preserving the listed gate/up order.
                axes = {k: n for k, n in collect.items()
                        if any(k in _pattern(t)[1] for t in raw[side])}
                for values in itertools.product(*(range(n) for n in axes.values())):
                    bindings = {**dict(ctx), **dict(zip(axes, values))}
                    try:
                        names.extend(t.format(**bindings) for t in raw[side])
                    except KeyError as error:
                        raise MappingError(f"Unbound target placeholder in {raw['id']}: {error}") from error
                if len(names) != len(set(names)):
                    raise MappingError(f"Duplicate expanded names: {raw['id']}")
                row[side] = names
            if "extra_state" in row:
                axes = {k: n for k, n in collect.items()
                        if any(k in _pattern(t)[1] for t in row["extra_state"])}
                row["extra_state"] = [t.format(**{**dict(ctx), **dict(zip(axes, values)),
                                                    **{k + "_count": v for k, v in collect.items()}})
                                      for values in itertools.product(*(range(n) for n in axes.values()))
                                      for t in row["extra_state"]]
            if "export_slice" in row:
                row["export_slice"] = resolve(row["export_slice"], configs)
            yield row
