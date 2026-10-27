# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Optional naming aliases and compact serialized-name templates, without shape tables."""
import re
from dataclasses import replace

from . import MappingError
from .mapping import _pattern


def normalize_name(name, naming):
    name = name.removeprefix(naming.get("strip_prefix", ""))
    aliases = naming.get("segments", {})
    name = ".".join(aliases.get(part, part) for part in name.split("."))
    for before, after in naming.get("replace", {}).items():
        name = name.replace(before, after)
    return name


def _template(name):
    index = iter(range(1000))
    return re.sub(r"(?<=\.)\d+(?=\.)", lambda _: "{n" + str(next(index)) + "}", name)


def name_templates(specs):
    result = {}
    for name, spec in specs.items():
        key, value = _template(name), _template(spec.name)
        if key in result and result[key] != value:
            raise MappingError(f"Ambiguous serialized name template: {key}")
        result[key] = value
    return result


def bind_names(specs, templates):
    result = {}
    for name, spec in specs.items():
        template = _template(name)
        if template in templates:
            bindings = _pattern(template)[0].fullmatch(name).groupdict()
            output = templates[template].format(**bindings)
            scale = spec.scale
            if scale is not None:
                for suffix in (".weight_scale_inv", ".scale"):
                    if scale.endswith(suffix):
                        scale = output.removesuffix(".weight") + suffix
                        break
            spec = replace(spec, name=output, scale=scale)
        result[name] = spec
    return result
