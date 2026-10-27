# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""HF <-> MCore DCP conversion using small, overridable mapping tables."""
from __future__ import annotations

import argparse
import fnmatch
from dataclasses import replace
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.checkpoint.hf_mcore_dist import MappingError
from tools.checkpoint.hf_mcore_dist.mapping import load_mapping
from tools.checkpoint.hf_mcore_dist.config import translate_config
from tools.checkpoint.hf_mcore_dist.world_one import require_world_one


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MappingError(f"Expected JSON object: {path}")
    return value


def convert(args):
    from tools.checkpoint.hf_mcore_dist.artifacts import read_mcore_artifacts, resolve_iteration_dir
    from tools.checkpoint.hf_mcore_dist.engine import (
        DTYPE_CODES, MappedSource, compile_mapping, dense_policy, plan_report, storage_profile,
    )
    from tools.checkpoint.hf_mcore_dist.hf_io import (
        SafetensorReader, ShardedSafetensorWriter, copy_hf_assets, parse_size,
    )
    from tools.checkpoint.hf_mcore_dist.precision import DENSE_DTYPES, TensorSpec
    from tools.checkpoint.hf_mcore_dist.streaming import StreamingCheckpointReader, save_streaming_checkpoint
    from tools.checkpoint.hf_mcore_dist.transaction import _staged_output

    from tools.checkpoint.hf_mcore_dist.quantization import PrecisionSource, resolve_precision

    require_world_one()
    forward = args.command == "hf-to-mcore"
    saved = {}
    if not forward:
        directory = resolve_iteration_dir(args.source)
        if (directory / "hf_export.json").is_file():
            saved = _json(directory / "hf_export.json")
    mapping_path = args.mapping
    if mapping_path:
        mapping = load_mapping(mapping_path, args.mapping_override)
    elif saved.get("mapping"):
        mapping = saved["mapping"]
        for path in args.mapping_override:
            mapping = load_mapping(path, _base=mapping)
    else:
        raise MappingError("Provide --mapping (or a converted checkpoint containing hf_export.json)")
    shard_size = parse_size(args.max_shard_size)
    if shard_size <= 0:
        raise MappingError("max-shard-size must be positive")
    if forward:
        hf = _json(Path(args.source) / "config.json")
        mc = translate_config(mapping, hf, True)
        reader = SafetensorReader(args.source)
        quantization = mapping.get("quantization", {})
        if quantization.keys() - {"mxfp4"}:
            raise MappingError("quantization supports explicit mxfp4 tensor glob patterns")
        patterns = quantization.get("mxfp4", [])
        if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
            raise MappingError("quantization.mxfp4 must be a list of tensor glob patterns")
        quant_source = PrecisionSource(reader, hf, mxfp4=lambda name: any(
            fnmatch.fnmatchcase(name, p) for p in patterns))
        input_policy = resolve_precision(quant_source, hf, args.dtype, "cpu", to_mcore=True,
                                         storage_mode="dense")
        quant_source.dtype = DENSE_DTYPES[input_policy.dtype]
        specs = {key: replace(spec, dtype=DTYPE_CODES[str(quant_source.dtype)], kind="dense",
                              scale=None, scale_dtype=None) if spec.kind != "dense" else spec
                 for key, spec in quant_source.specs.items()}
    else:
        artifacts = read_mcore_artifacts(args.source)
        mc = artifacts["args"]
        hf = dict(saved.get("hf_config", {}))
        if args.hf_reference:
            hf.update(_json(Path(args.hf_reference) / "config.json"))
        hf.update(translate_config(mapping, mc, False))
        specs = {key: TensorSpec(key, tuple(item["shape"]), DTYPE_CODES.get(item["dtype"], item["dtype"]))
                 for key, item in artifacts["model_tensors"].items()}
    plan = compile_mapping(mapping, specs, {"hf": hf, "mcore": mc}, forward=forward, dtype=args.dtype)
    profile_name = mapping.get("profile", "mapping")
    if not forward:
        selected = {key: specs[key] for node in plan["nodes"] for key in node["inputs"]}
        profile = storage_profile(selected, profile_name)
        reader = StreamingCheckpointReader(args.source, profile, ignore=plan["ignored"])
        if reader.native_keys:
            raise MappingError("Table layouts require dense DCP tensors; use the model adapter for quantized storage")
        pieces = {piece.key: piece for piece in profile.pieces}
        get_tensor = lambda key: reader._read_piece(pieces[key])
    else:
        get_tensor = quant_source.get_tensor
    source = MappedSource(plan, get_tensor)
    policy = dense_policy(source.specs)
    report = {"schema_version": 1, "direction": args.command, "profile": profile_name,
              "checkpoint_content": "model_only", "validation": "storage and executable shapes only",
              "model_semantics_validated": False, "precision": policy.report(),
              "mapping": mapping, "plan": plan_report(plan)}
    if forward:
        report["source_precision"] = input_policy.report()
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    inputs = [args.source] + ([args.hf_reference] if args.hf_reference else [])
    inputs += [p for p in [mapping_path, *args.mapping_override] if p and not p.startswith("python:")]
    with _staged_output(args.output, args.overwrite, inputs) as output:
        if forward:
            details = save_streaming_checkpoint(
                source, storage_profile(source.specs, profile_name, plan["nodes"]), policy, hf,
                output, model_args=mc, max_shard_size=shard_size)
            details["iteration_dir"] = str(Path(args.output).resolve() / "iter_0000000")
            # Templates and config only: no maintained per-tensor shape inventory.
            (output / "iter_0000000" / "hf_export.json").write_text(
                json.dumps({"schema_version": 1, "mapping": mapping, "hf_config": hf}, indent=2) + "\n")
            report["storage"] = details
        else:
            if args.hf_reference:
                copy_hf_assets(args.hf_reference, output)
            for key in ("quantization_config", "scale_fmt", "scale_dtype", "expert_dtype"):
                hf.pop(key, None)
            hf["torch_dtype"] = str(DENSE_DTYPES[policy.dtype]).removeprefix("torch.")
            if "dtype" in hf:
                hf["dtype"] = hf["torch_dtype"]
            (output / "config.json").write_text(json.dumps(hf, indent=2) + "\n")
            writer = ShardedSafetensorWriter(output, shard_size)
            for name, tensor in source.tensors():
                writer.add(name, tensor)
            writer.finish()
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.command} completed: {Path(args.output).resolve()}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # DSv4 declares its naming/MTP policy and uses the same storage/quantization backend.
    if argv[:1] == ["deepseek-v4"]:
        from tools.checkpoint.deepseek_v4.convert_deepseek_v4 import main as dsv4_main
        return dsv4_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("hf-to-mcore", "mcore-to-hf", "export-mapping"))
    parser.add_argument("--mapping", help="YAML/JSON file or python:module:attribute exposing a mapping dict")
    parser.add_argument("--mapping-override", action="append", default=[])
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-reference", help="Optional HF config and tokenizer/assets for export; no index required")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export-mapping":
            if not args.mapping:
                raise MappingError("export-mapping requires --mapping")
            mapping = load_mapping(args.mapping, args.mapping_override)
            if args.dry_run:
                print(json.dumps(mapping, indent=2))
            else:
                with Path(args.output).open("w" if args.overwrite else "x", encoding="utf-8") as stream:
                    stream.write(json.dumps(mapping, indent=2) + "\n")
        else:
            if not args.source:
                raise MappingError("Conversion requires --source")
            convert(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
