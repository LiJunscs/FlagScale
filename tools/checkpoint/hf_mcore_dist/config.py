# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Shared config aliases, nested fields, defaults and small explicit expressions."""
from copy import deepcopy
import operator

from . import MappingError

MCORE_OUTPUT_OVERRIDES = dict(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
    expert_model_parallel_size=1, expert_tensor_parallel_size=1, context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None, world_size=1, data_parallel_size=1,
    rank=0, local_rank=0, sequence_parallel=False,
    no_save_optim=True, no_save_rng=True,
    iteration=0, ckpt_format="torch_dist", use_dist_ckpt=True, use_distributed_optimizer=False,
    fp8=None, fp4=None, fp8_param_gather=False, fp4_param_gather=False,
    fp8_param=False, fp4_param=False,
)
_MISSING = object()


def get_field(values, path, default=None):
    for part in path.split("."):
        if not isinstance(values, dict) or part not in values:
            return default
        values = values[part]
    return values


def set_field(values, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in values:
            values[part] = {}
        if not isinstance(values[part], dict):
            raise MappingError(f"Config field conflicts with nested target: {path}")
        values = values[part]
    values[parts[-1]] = deepcopy(value)


def get_alias(values, aliases, *, required=False):
    found = [(key, get_field(values, key)) for key in aliases if get_field(values, key) is not None]
    if found:
        if any(value != found[0][1] for _, value in found[1:]):
            raise MappingError(f"Conflicting config aliases: {found}")
        return found[0][1]
    if required:
        raise MappingError(f"Missing required config fields: {aliases}")
    return None


def set_alias(values, aliases, value):
    set_field(values, aliases[0], value)
    for key in aliases[1:]:
        if get_field(values, key, _MISSING) is not _MISSING:
            set_field(values, key, value)


def record(rows, hf, mc, op, before, after):
    rows.append({"hf": hf, "mcore": list(mc), "operation": op,
                 "template_value": deepcopy(before), "source_value": deepcopy(after)})


def resolve(value, configs):
    if isinstance(value, str) and value.startswith("$"):
        result = get_field(configs, value[1:], _MISSING)
        if result is _MISSING:
            raise MappingError(f"Missing mapping value: {value}")
        return deepcopy(result)
    if isinstance(value, list):
        return [resolve(item, configs) for item in value]
    if isinstance(value, dict):
        if set(value) == {"op", "args"}:
            args = [resolve(item, configs) for item in value["args"]]
            operations = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                          "div": operator.truediv, "floordiv": operator.floordiv,
                          "not": operator.not_, "int": int}
            if value["op"] not in operations:
                raise MappingError(f"Unknown config operation: {value['op']}")
            try:
                return operations[value["op"]](*args)
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise MappingError(f"Cannot evaluate config operation: {value}") from error
        return {key: resolve(item, configs) for key, item in value.items()}
    return value


def translate_fields(fields, source, forward, template=None):
    """Translate declared aliases/defaults; preserve field-level provenance."""
    target, rows = deepcopy(template or {}), []
    for field in fields:
        hf, aliases = field["hf"], field["mcore"]
        op = "copy"
        if forward:
            value = get_field(source, hf, _MISSING)
            if value is _MISSING:
                if field.get("required"):
                    raise MappingError(f"Missing required config field: {hf}")
                if "hf_default" not in field:
                    continue
                value, op = field["hf_default"], "fixed_profile_default"
        else:
            value = get_alias(source, aliases, required=field.get("required", False))
            if value is None and "reverse_default" in field:
                value = field["reverse_default"]
            elif value is None and field.get("normalize") != "int_or_zero":
                op = "nullable"
        if field.get("normalize") == "int_or_zero":
            value = int(value or 0)
        if forward:
            set_alias(target, aliases, value)
        elif value is not None or get_field(target, hf, _MISSING) is not _MISSING:
            set_field(target, hf, value)
        record(rows, hf, aliases, op, get_field(template or {}, hf), value)
    return target, rows


def translate_config(mapping, source, forward):
    fields = []
    for hf, mc in mapping.get("config_map", {}).items():
        spec = dict(mc) if isinstance(mc, dict) else {"mcore": mc}
        aliases = spec["mcore"]
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list) or not aliases or any(not isinstance(x, str) for x in aliases):
            raise MappingError(f"Invalid config aliases: {hf}")
        fields.append({**spec, "hf": hf, "mcore": aliases})
    target, _ = translate_fields(fields, source, forward)
    context = {"hf": source if forward else target, "mcore": target if forward else source}
    for key, value in mapping.get("mcore_args" if forward else "hf_config", {}).items():
        set_field(target, key, resolve(value, context))
    return target
