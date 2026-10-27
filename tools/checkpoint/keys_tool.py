# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Inspect checkpoint configs and keys; never construct a model."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.checkpoint.hf_mcore_dcp.world_one import require_world_one


def _write_json(path, payload):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _reader_inventory(reader):
    shapes, dtypes = reader.tensor_shapes(), reader.tensor_dtypes()
    return {name: {"shape": list(shapes[name]), "dtype": dtypes[name]} for name in sorted(reader.keys)}

def probe_weights(args):
    """Inspect an existing HF export without importing or constructing models."""
    require_world_one()
    from tools.checkpoint.hf_mcore_dcp.safetensors_index import read_safetensors_index
    from tools.checkpoint.hf_mcore_dcp.safetensors_metadata import SafetensorMetadataReader

    directory = Path(args.hf_model).expanduser().resolve()
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("config.json must be a JSON object")
    index_only = getattr(args, "index_only", False)
    index_path = directory / "model.safetensors.index.json"
    if index_only:
        index = read_safetensors_index(index_path)
        weights = dict(sorted(index["weight_map"].items()))
        index_metadata = index.get("metadata")
        serialized = {name: {"shape": None, "dtype": None} for name in weights}
    else:
        reader = SafetensorMetadataReader(directory)
        weights = dict(sorted(reader.weight_map.items()))
        index_path, index_metadata = reader.index_path, reader.index_metadata
        serialized = _reader_inventory(reader)
    dtype_counts = defaultdict(int)
    for entry in serialized.values():
        if entry["dtype"] is not None:
            dtype_counts[entry["dtype"]] += 1
    payload = {
        "schema_version": 1, "profile": config.get("model_type"), "kind": "existing_export_inventory",
        "source_type": "weights_directory", "source_directory": str(directory),
        "inspection_mode": "index_only" if index_only else "headers",
        "config_path": str(config_path), "config": config,
        "profile_validation": "model semantics not validated",
        "source_index": str(index_path) if index_path is not None else None,
        "transformers_version": None,
        "config_transformers_version": config.get("transformers_version"),
        "serialized_keys": serialized, "weight_map": weights, "index_metadata": index_metadata,
        "key_count": len(weights), "shard_count": len(set(weights.values())),
        "storage_dtype_counts": dict(sorted(dtype_counts.items())),
        "unavailable_from_inspection": (["shape", "dtype"] if index_only else [])
            + ["runtime_storage_aliases", "module_classes"],
        "precision_validation": "Not performed; storage metadata and config do not validate quantization semantics",
    }
    _write_json(args.output, payload)
    print(f"Recorded {len(weights)} keys from {payload['inspection_mode']}: {Path(args.output).resolve()}")


def diff(args):
    previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    for snapshot in (previous, current):
        if snapshot.get("schema_version") != 1 or snapshot.get("kind") != "existing_export_inventory":
            raise ValueError("Only v1 HF key snapshots can be compared")
    before, after = previous["serialized_keys"], current["serialized_keys"]
    changed = {}
    compared = {"shape": 0, "dtype": 0}
    unknown = {"shape": 0, "dtype": 0}
    for key in sorted(before.keys() & after.keys()):
        different = False
        for field in compared:
            old, new = before[key].get(field), after[key].get(field)
            if old is None or new is None:
                unknown[field] += 1
            else:
                compared[field] += 1
                different = different or old != new
        if different:
            changed[key] = {"previous": before[key], "current": after[key]}
    old_config, new_config = previous.get("config"), current.get("config")
    payload = {"added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys()),
               "changed": changed, "compared_field_counts": compared, "unknown_field_counts": unknown,
               "same_config": old_config == new_config if old_config is not None and new_config is not None else None,
               "previous_transformers": previous.get("transformers_version"),
               "current_transformers": current.get("transformers_version")}
    if args.output:
        _write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def probe_mcore(args):
    from tools.checkpoint.hf_mcore_dcp.artifacts import read_mcore_artifacts
    _write_json(args.output, {"schema_version": 2, "kind": "mcore_config_inventory",
                              **read_mcore_artifacts(args.mcore_path)})


def mapping(args):
    from tools.checkpoint.hf_mcore_dcp.convert import build_parser, prepare
    command = "mcore-to-hf" if args.mcore_path else "hf-to-mcore"
    source = args.mcore_path or args.hf_model
    if not source:
        raise ValueError("Provide --hf-model or --mcore-path")
    argv = [command, "--source", source, "--output", args.output, "--mapping", args.mapping]
    if args.mcore_path and args.hf_model:
        argv += ["--hf-reference", args.hf_model]
    for override in args.mapping_override:
        argv += ["--mapping-override", override]
    payload = prepare(build_parser().parse_args(argv)).report
    _write_json(args.output, payload)
    print(f"Prepared {command} mapping: {Path(args.output).resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("probe-weights", allow_abbrev=False)
    weights.add_argument("--hf-model", required=True, help="Existing directory containing config.json and safetensors index")
    weights.add_argument("--index-only", action="store_true", help="Read config and index only, without shard files")
    weights.add_argument("--output", required=True)
    weights.set_defaults(handler=probe_weights)
    mcore = commands.add_parser("probe-mcore", allow_abbrev=False)
    mcore.add_argument("--mcore-path", required=True)
    mcore.add_argument("--output", required=True)
    mcore.set_defaults(handler=probe_mcore)
    mapping_parser = commands.add_parser("mapping", allow_abbrev=False)
    mapping_parser.add_argument("--hf-model")
    mapping_parser.add_argument("--mapping", required=True)
    mapping_parser.add_argument("--mcore-path", help="Actual MCore source for export; omit for HF-to-MCore model import")
    mapping_parser.add_argument("--mapping-override", action="append", default=[])
    mapping_parser.add_argument("--output", required=True)
    mapping_parser.set_defaults(handler=mapping)
    compare = commands.add_parser("diff", allow_abbrev=False)
    compare.add_argument("--previous", required=True)
    compare.add_argument("--current", required=True)
    compare.add_argument("--output")
    compare.set_defaults(handler=diff)
    args = parser.parse_args(argv)
    try:
        require_world_one()
        args.handler(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
