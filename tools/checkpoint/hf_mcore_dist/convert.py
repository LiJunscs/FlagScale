# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One rule-driven HF / MCore conversion pipeline, including preserved quantization."""
from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import sys
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.checkpoint.hf_mcore_dist import MappingError
from tools.checkpoint.hf_mcore_dist.config import translate_config
from tools.checkpoint.hf_mcore_dist.mapping import load_mapping
from tools.checkpoint.hf_mcore_dist.world_one import require_world_one


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MappingError(f"Expected JSON object: {path}")
    return value


def prepare(args):
    from tools.checkpoint.hf_mcore_dist.artifacts import read_mcore_artifacts, resolve_iteration_dir
    from tools.checkpoint.hf_mcore_dist.engine import DTYPE_CODES, compile_mapping, plan_report
    from tools.checkpoint.hf_mcore_dist.hf_io import SafetensorReader, parse_size
    from tools.checkpoint.hf_mcore_dist.naming import bind_names, name_templates, normalize_name
    from tools.checkpoint.hf_mcore_dist.precision import DENSE_DTYPES, TensorSpec
    from tools.checkpoint.hf_mcore_dist.quantization import PrecisionSource, resolve_precision
    from tools.checkpoint.hf_mcore_dist.regions import compile_regions, validate_preserve
    from tools.checkpoint.hf_mcore_dist.streaming import StreamingCheckpointReader, _groups, _storage_dtype

    require_world_one()
    forward = args.command == "hf-to-mcore"
    saved = {}
    if not forward:
        directory = resolve_iteration_dir(args.source)
        if (directory / "hf_export.json").is_file():
            saved = _json(directory / "hf_export.json")
    if args.mapping:
        mapping = load_mapping(args.mapping, args.mapping_override)
    elif saved.get("mapping"):
        mapping = saved["mapping"]
        for path in args.mapping_override:
            mapping = load_mapping(path, _base=mapping)
    else:
        raise MappingError("Provide --mapping or a checkpoint containing hf_export.json")
    shard_size = parse_size(args.max_shard_size)
    if shard_size <= 0:
        raise MappingError("max-shard-size must be positive")
    adapter = importlib.import_module(mapping["adapter"]) if mapping.get("adapter") else None
    naming = mapping.get("naming", {})
    canonical = lambda name: normalize_name(name, naming)
    patterns = mapping.get("quantization", {}).get("mxfp4", [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise MappingError("quantization.mxfp4 must be a list of tensor patterns")
    reader = None
    if forward:
        hf = _json(Path(args.source) / "config.json")
        ignored = mapping.get("ignore", {}).get("hf", [])
        original = PrecisionSource(SafetensorReader(args.source), hf, normalize=canonical,
                                   skip=lambda n: any(fnmatch.fnmatchcase(canonical(n), p) for p in ignored),
                                   mxfp4=lambda n: any(fnmatch.fnmatchcase(canonical(n), p) for p in patterns))
        if adapter:
            hf = adapter.normalize_config(hf, True, original.keys)
        mc = translate_config(mapping, hf, True)
        if adapter:
            mc = adapter.finish_config(mc, True)
        mode = args.conversion_mode or "dense"
        policy = resolve_precision(original, hf, args.dtype, args.device, to_mcore=True, storage_mode=mode)
        original.dtype, original.device = DENSE_DTYPES[policy.dtype], policy.device
        specs = {key: replace(spec, dtype=DTYPE_CODES[str(original.dtype)], kind="dense", scale=None,
                              scale_dtype=None) if spec.kind != "dense" else spec
                 for key, spec in original.specs.items()}
        templates = name_templates(original.specs)
    else:
        artifacts = read_mcore_artifacts(args.source)
        mc = artifacts["args"]
        if args.vocab_size is not None:
            mc["vocab_size"] = args.vocab_size
        if adapter:
            mc = adapter.normalize_config(mc, False, artifacts["model_tensors"])
        hf = dict(saved.get("hf_config", {}))
        templates = saved.get("names", {})
        if args.hf_reference:
            hf.update(_json(Path(args.hf_reference) / "config.json"))
            index = Path(args.hf_reference) / "model.safetensors.index.json"
            if index.is_file():
                templates = name_templates({canonical(n): TensorSpec(n, (), "BF16")
                                             for n in _json(index)["weight_map"]})
        hf.update(translate_config(mapping, mc, False))
        if adapter:
            hf = adapter.finish_config(hf, False)
        specs = {key: TensorSpec(key, tuple(item["shape"]), DTYPE_CODES.get(item["dtype"], item["dtype"]))
                 for key, item in artifacts["model_tensors"].items()}
    effective = adapter.prepare_mapping(mapping, {"hf": hf, "mcore": mc}, specs, forward) if adapter else mapping
    # Shape inference uses metadata only. Preserve may fuse different physical dtypes.
    plan = compile_mapping(effective, specs, {"hf": hf, "mcore": mc}, forward=forward,
                           dtype=args.dtype, preserve=forward and mode == "preserve")
    profile = compile_regions(plan, mapping.get("profile", "mapping"))
    if forward:
        source = copy(original)
        source.specs = {key: original.specs[key] for key in profile.shapes}
        source.physical_keys = {n for spec in source.specs.values() for n in (spec.name, spec.scale) if n}
        source.consumed = set()
        if mode == "preserve":
            validate_preserve(profile, source.specs)
        for pieces in _groups(profile).values():
            _storage_dtype(pieces, source, policy)
        zero_getter = original.get_tensor
    else:
        reader = StreamingCheckpointReader(args.source, profile, ignore=plan["ignored"])
        if reader.native_keys - reader.preserved_keys:
            raise MappingError("This entrypoint requires dense DCP or preserved HF payloads, not native TE objects")
        dtypes = reader.tensor_dtypes()
        output_specs = {name: replace(spec, dtype=DTYPE_CODES[str(dtypes[name])])
                        for name, spec in plan["outputs"].items() if name in profile.shapes}
        output_specs.update(reader.preserved_specs)
        output_specs.update({name: spec for name, spec in plan["outputs"].items() if name not in profile.shapes})
        source = SimpleNamespace(specs=bind_names(output_specs, templates), keys=set(output_specs),
                                 low_precision_kinds={s.kind for s in output_specs.values() if s.kind != "dense"})
        if reader.preserved_specs:
            for field in ("quantization_config", "scale_fmt", "scale_dtype", "expert_dtype"):
                hf.pop(field, None)
            hf.update(reader.preserved_config)
        mode = args.conversion_mode or ("preserve" if reader.preserved_specs else "dense")
        if mode == "dense" and source.low_precision_kinds:
            target = args.dtype if args.dtype != "auto" else "bf16"
            source.specs = {n: replace(s, dtype=DTYPE_CODES[str(DENSE_DTYPES[target])], kind="dense",
                                       scale=None, scale_dtype=None) if s.kind != "dense" else s
                            for n, s in source.specs.items()}
            source.low_precision_kinds = set()
        policy = resolve_precision(source, hf, args.dtype, args.device, to_mcore=False,
                                   storage_mode="preserve" if mode == "preserve" else None)
        zero_getter = None
    report = {"schema_version": 2, "direction": args.command, "profile": profile.name,
              "checkpoint_content": "model_only", "model_semantics_validated": False,
              "mapping": mapping, "configs": {"hf": hf, "mcore": mc},
              "plan": plan_report(plan), "regions": profile.report(),
              "precision": policy.report(), "value_checks_pending": plan["zero_checks"]}
    return SimpleNamespace(args=args, forward=forward, mapping=mapping, hf=hf, mc=mc, plan=plan,
                           profile=profile, source=source, reader=reader, policy=policy, report=report,
                           shard_size=shard_size, templates=templates, zero_getter=zero_getter, mode=mode)


def execute(context):
    import torch
    from tools.checkpoint.hf_mcore_dist.hf_io import ShardedSafetensorWriter, copy_hf_assets
    from tools.checkpoint.hf_mcore_dist.quantization import decode_stored_tensor, encode_hf_tensors, output_config
    from tools.checkpoint.hf_mcore_dist.precision import DENSE_DTYPES
    from tools.checkpoint.hf_mcore_dist.streaming import save_streaming_checkpoint
    from tools.checkpoint.hf_mcore_dist.transaction import _staged_output

    c, args = context, context.args
    inputs = [args.source] + ([args.hf_reference] if args.hf_reference else [])
    inputs += [p for p in [args.mapping, *args.mapping_override] if p and not p.startswith("python:")]
    for name in c.plan["zero_checks"]:
        if c.zero_getter is None or bool((c.zero_getter(name) != 0).any()):
            raise MappingError(f"Cannot omit a nonzero tensor: {name}")
    with _staged_output(args.output, args.overwrite, inputs) as output:
        if c.forward:
            details = save_streaming_checkpoint(c.source, c.profile, c.policy, c.hf, output,
                                                model_args=c.mc, max_shard_size=c.shard_size)
            details["iteration_dir"] = str(Path(args.output).resolve() / "iter_0000000")
            (output / "iter_0000000" / "hf_export.json").write_text(json.dumps({
                "schema_version": 2, "mapping": c.mapping, "hf_config": c.hf, "names": c.templates}, indent=2) + "\n")
            c.report["storage"] = details
        else:
            if args.hf_reference:
                copy_hf_assets(args.hf_reference, output)
            config = output_config(c.hf, c.policy)
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            def tensors():
                for name, tensor in c.reader.tensors():
                    if c.mode == "dense":
                        tensor = decode_stored_tensor(tensor, DENSE_DTYPES[c.policy.dtype])
                    yield name, tensor
                for node in c.plan["nodes"]:
                    if node.get("generate"):
                        from tools.checkpoint.hf_mcore_dist.engine import STORAGE_DTYPES
                        item = node["generate"]
                        for name in node["outputs"]:
                            yield name, torch.full(item["shape"], item["value"], dtype=STORAGE_DTYPES[item["dtype"]])
            writer = ShardedSafetensorWriter(output, c.shard_size)
            for name, tensor in encode_hf_tensors(tensors(), c.source, c.policy, c.hf):
                writer.add(name, tensor)
            writer.finish()
        c.report["value_checks_pending"] = []
        (output / "conversion_report.json").write_text(json.dumps(c.report, indent=2) + "\n")
    print(f"{args.command} completed: {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("hf-to-mcore", "mcore-to-hf", "export-mapping"))
    parser.add_argument("--mapping", help="YAML/JSON or python:module:attribute; required unless saved with source")
    parser.add_argument("--mapping-override", action="append", default=[])
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-reference", help="Optional config, naming index and tokenizer/assets; no weights required")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--conversion-mode", choices=("dense", "preserve"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, help="Explicit source unpadded vocabulary for MCore export")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
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
            context = prepare(args)
            if args.dry_run:
                print(json.dumps(context.report, indent=2))
            else:
                execute(context)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
