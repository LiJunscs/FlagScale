# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Model-only DSv4 HF <-> MCore streaming conversion, world size one."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.checkpoint.hf_mcore_dist.transaction import _staged_output

from tools.checkpoint.deepseek_v4.config_mapping import without_mtp
from tools.checkpoint.deepseek_v4.hf_config import (
    ConversionConfigurationError, derive_compress_ratios,
)
from tools.checkpoint.hf_mcore_dist.hf_io import SafetensorReader, parse_size, write_hf_checkpoint
from tools.checkpoint.deepseek_v4 import MappingError
from tools.checkpoint.deepseek_v4.mapping_plan import (
    load_mapping_plan, mcore_read_profile, prepare_mapping_plan, schemas_from_plan,
)
from tools.checkpoint.deepseek_v4.precision import (
    DENSE_DTYPES, PrecisionError, PrecisionSource,
    encode_hf_tensors, is_hf_mtp_key, output_config, resolve_precision,
)
from tools.checkpoint.deepseek_v4.registry import compile_profile
from tools.checkpoint.hf_mcore_dist.quantized_archive import QuantizedArchive, save_quantized_archive
from tools.checkpoint.deepseek_v4.storage import (
    StreamingCheckpointReader, hf_export_tensors, save_streaming_checkpoint,
)
from tools.checkpoint.hf_mcore_dist.streaming import _groups, _storage_dtype
from tools.checkpoint.hf_mcore_dist.world_one import require_world_one
from tools.checkpoint.deepseek_v4.weight_schema import with_storage_dtypes


def _plan(args, reference):
    headers = args.hf_headers
    vocab_override = getattr(args, "vocab_size", None)
    if args.command != "mcore-to-hf" and vocab_override is not None:
        raise MappingError("--vocab-size only applies to mcore-to-hf")
    if args.mapping_plan:
        plan = load_mapping_plan(args.mapping_plan, args.hf_model, reference,
                                 vocab_size_override=vocab_override)
        if headers and plan["hf"]["headers"] is None:
            raise MappingError("Regenerate the plan with --hf-headers to use observed storage types")
        return plan
    return prepare_mapping_plan(args.hf_model, reference, args.mapping_override, hf_headers=headers,
                                vocab_size_override=vocab_override)


def _prepare_conversion(args, plan, to_mcore):
    # Phase two: the complete config/key mapping already exists. Validate the
    # actual source headers/regions before any tensor-value read or output write.
    if to_mcore:
        config = plan["hf"]["effective_config"]
        hf_reader = SafetensorReader(args.hf_model)
        has_mtp = any(is_hf_mtp_key(key) for key in hf_reader.keys)
        if has_mtp and not plan["mtp_dropped"]:
            # Actual source validation is stricter than an index-only key plan.
            # Establish all main weights first so an MTP error cannot hide one.
            main_config = without_mtp(config)
            main_source = PrecisionSource(hf_reader, main_config, skip_mtp=True)
            compile_profile(main_source, main_config, derive_compress_ratios(main_config), rules=plan["rules"])
            try:
                source = PrecisionSource(hf_reader, config)
                compile_profile(source, config, derive_compress_ratios(config), rules=plan["rules"])
            except (MappingError, PrecisionError, ConversionConfigurationError):
                updated = prepare_mapping_plan(args.hf_model,
                                               hf_headers=True, rules=plan["rules"])
                if not updated["mtp_dropped"]:
                    raise
                plan.clear()
                plan.update(updated)
                config, source = plan["hf"]["effective_config"], main_source
        else:
            source = PrecisionSource(hf_reader, config, skip_mtp=plan["mtp_dropped"])
    else:
        config = plan["config_mapping"]["mcore_to_hf"]["target_config"]
        hf_storage, source = schemas_from_plan(plan)
    if to_mcore:
        profile, removed = mcore_read_profile(
            source, config, plan["config_mapping"]["hf_to_mcore"]["target_args"], plan["rules"])
        ignored_bias = [item["hf"] for item in removed]
        if ignored_bias != plan["weight_mapping"]["hf_to_mcore"]["ignored_zero_biases"]:
            raise MappingError("HF-only bias exclusions differ from the prepared plan; regenerate it")
    else:
        profile, generated = mcore_read_profile(source, config, plan["mcore"]["effective_args"], plan["rules"])
        if generated != plan["weight_mapping"]["mcore_to_hf"]["generated_tensors"]:
            raise MappingError("Generated HF targets differ from the prepared plan; regenerate it")
    groups = _groups(profile)
    reader = None if to_mcore else StreamingCheckpointReader(args.mcore_path, profile, skip_mtp=plan["mtp_dropped"])
    if reader is not None:
        dtypes = reader.tensor_dtypes()
        dtypes.update({item["hf"]: DENSE_DTYPES["fp32"] for item in generated})
        # Always bind native specs when the checkpoint stores quantized payloads,
        # even without --export-native-quantization: the flag-off path must hit
        # the explicit policy error, not a raw unpickling failure.
        native = {name: hf_storage.specs[name] for name in reader.native_hf_names
                  if name not in reader.preserved_specs}
        source = with_storage_dtypes(source, dtypes, native=native,
                                     preserved=reader.preserved_specs)
        if reader.preserved_specs:
            config = dict(config)
            for field in ("quantization_config", "scale_fmt", "scale_dtype", "expert_dtype"):
                config.pop(field, None)
            config.update(reader.preserved_config)
    archive = (QuantizedArchive(reader.directory, source, source_factory=PrecisionSource)
               if getattr(args, "restore_original_quantization", False) else None)
    storage_mode = (getattr(args, "conversion_mode", "dense") if to_mcore else
                    "preserve" if reader.preserved_specs else None)
    policy = resolve_precision(source, config, args.dtype, args.device, to_mcore=to_mcore,
                               archive=archive,
                               native_export=getattr(args, "export_native_quantization", False),
                               storage_mode=storage_mode)
    for message in policy.warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
    if to_mcore:
        source.dtype, source.device = DENSE_DTYPES[policy.dtype], policy.device
        for pieces in groups.values():
            _storage_dtype(pieces, source, policy)  # Fail before values/output, including dry-run.
    return config, source, profile, policy, reader, archive


def _convert(args, plan, context, reference):
    to_mcore = args.command == "hf-to-mcore"
    config, source, profile, policy, reader, archive = context
    destination = args.mcore_path if to_mcore else args.hf_path
    inputs = [args.hf_model] + ([reference] if reference is not None else [])
    if archive is not None:
        inputs.append(archive.source.reader.model_dir)
    if args.mapping_plan:
        inputs.append(args.mapping_plan)
    with _staged_output(destination, args.overwrite, inputs) as output:
        if to_mcore:
            # HF contributes model weights/config only. No training checkpoint
            # or unrelated runtime args are read to construct this output.
            details = save_streaming_checkpoint(
                source, profile, policy, config, output,
                model_args=plan["config_mapping"]["hf_to_mcore"]["target_args"],
                max_shard_size=args.max_shard_size,
            )
            if policy.archive_quantization:
                details["quantization_archive"] = save_quantized_archive(
                    source, config, output / "iter_0000000", args.max_shard_size)
        else:
            generated = plan["weight_mapping"]["mcore_to_hf"]["generated_tensors"]
            index = write_hf_checkpoint(
                encode_hf_tensors(hf_export_tensors(reader.tensors(), generated), source, policy, config,
                                  archive=archive),
                args.hf_model, output, args.max_shard_size,
                output_config=output_config(config, policy, archive=archive),
            )
            details = {"index": index.name, "generated_hf_tensors": generated}
            if archive is not None:
                details["quantization_archive"] = archive.report()
        report = {
            "schema_version": 4, "direction": args.command, "world_size": 1,
            "checkpoint_content": "model_only",
            "warnings": plan["warnings"] + [{"code": "source_precision_policy", "message": message}
                                              for message in policy.warnings],
            "mtp_dropped": plan["mtp_dropped"],
            "ignored_hf_keys": plan["hf"]["ignored_keys"],
            "ignored_mcore_keys": (plan["mcore"] or {}).get("ignored_keys", []),
            "precision": policy.report(), "config_mapping_revision": plan["config_mapping"]["revision"],
            "source_of_dimensions": plan["config_mapping"]["source_of_dimensions"][args.command.replace("-", "_")],
            "hf_tensor_count": len(source.keys), "mcore_tensor_count": len(profile.keys),
            "validation": "existing config/key plan, source shape/dtype/scale and global region coverage; no numerical test",
            **details,
        }
        if to_mcore:
            # The stage directory is renamed to the final destination after this
            # block; report the consumer-facing path, not the transient
            # .incomplete-* stage path (the report travels with the rename).
            details["iteration_dir"] = str(Path(destination).expanduser().resolve() / "iter_0000000")
            report["iteration_dir"] = details["iteration_dir"]
        (output / "conversion_mapping.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if to_mcore:
        print(f"DCP storage: {details['storage_file_count']} .distcp files; "
              f"target shard size {details['max_shard_size_bytes']} bytes")
    print(f"DSv4 {args.command} completed: {Path(destination).resolve()}")


def _add_common(parser, *, mcore_path_required=True):
    parser.add_argument("--hf-model", required=True, help="Existing HF directory with config.json and safetensors index")
    parser.add_argument("--mcore-path", required=mcore_path_required,
                        help="MCore output (hf-to-mcore), otherwise the actual MCore source checkpoint")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mapping-plan", help="V4 model-only plan prepared for this conversion direction and source")
    group.add_argument("--mapping-override", help="Fixed DSv4 weight-rule overrides in YAML/JSON")
    parser.add_argument("--hf-headers", action="store_true",
                        help="Also record existing HF headers; reference precision never controls export")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto",
                        help="auto: preserve source tensor dtypes; explicit values override dense "
                             "non-FP32 tensors; preserve mode requires auto")
    parser.add_argument("--device", default="auto", help="Device for decoding arithmetic, not a storage capability filter")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="mcore-to-hf only: source model's unpadded vocab_size. Required when the "
                             "checkpoint was produced with --use-checkpoint-args (args.vocab_size=None "
                             "is not recoverable from MCore state)")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata only; do not load tensor values")
    parser.add_argument("--overwrite", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    forward = sub.add_parser("hf-to-mcore", allow_abbrev=False)
    _add_common(forward)
    forward.add_argument("--conversion-mode", choices=("preserve", "dense"), default="dense",
                         help="preserve: keep original quantized payloads/scales in DCP with the "
                              "FlagScale load adapter; dense (default): decode to ordinary DCP tensors")
    forward.add_argument("--max-shard-size", default="4GB",
                         help="DCP file payload target (default: 4GB); an oversized chunk gets its own file")
    reverse = sub.add_parser("mcore-to-hf", allow_abbrev=False)
    _add_common(reverse)
    reverse.add_argument("--hf-path", required=True)
    reverse.add_argument("--max-shard-size", default="1GB")
    reverse.add_argument("--restore-original-quantization", action="store_true",
                         help="Explicitly restore source archive payloads/scales after checking current model bytes")
    reverse.add_argument("--export-native-quantization", action="store_true",
                         help="Keep quantized DCP payloads/scales as HF bytes (FP8 E4M3 + E8M0 or FP32 "
                              "block scales) instead of decoding; MXFP4-origin experts export as E4M3")
    preflight = sub.add_parser("preflight", allow_abbrev=False)
    _add_common(preflight, mcore_path_required=False)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        require_world_one()
        if getattr(args, "restore_original_quantization", False) and args.dtype != "auto":
            raise PrecisionError("--restore-original-quantization requires --dtype auto")
        if getattr(args, "export_native_quantization", False) and args.dtype != "auto":
            raise PrecisionError("--export-native-quantization requires --dtype auto")
        if (getattr(args, "export_native_quantization", False)
                and getattr(args, "restore_original_quantization", False)):
            raise PrecisionError(
                "--export-native-quantization and --restore-original-quantization are mutually exclusive")
        if args.command != "preflight":
            args.max_shard_size = parse_size(args.max_shard_size)
            if args.max_shard_size <= 0:
                raise ConversionConfigurationError("--max-shard-size must be positive")
        to_mcore = args.command == "hf-to-mcore"
        reference = None if to_mcore else args.mcore_path
        plan = _plan(args, reference)
        if args.command == "preflight":
            print(json.dumps({"mapping": plan["validation"], "config_mapping": plan["config_mapping"]}, indent=2))
            return 0
        context = _prepare_conversion(args, plan, to_mcore)
        print(json.dumps({"mapping": plan["validation"], "precision": context[3].report()}, indent=2))
        if not args.dry_run:
            _convert(args, plan, context, reference)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
