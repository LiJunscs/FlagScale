# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Freeze config-field translations and two sets of weight regions before I/O."""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tools.checkpoint.hf_mcore_dist.artifacts import is_mcore_mtp_key, read_hf_artifacts, read_mcore_artifacts
from .config_mapping import strip_inference_config, translate_configs, without_mtp
from .hf_config import ConversionConfigurationError, derive_compress_ratios
from . import MappingError
from .precision import PrecisionError, PrecisionSource, TensorSpec, is_hf_mtp_key
from .registry import Profile, compile_profile, load_rules, rule_contexts
from tools.checkpoint.hf_mcore_dist.safetensors_metadata import SafetensorMetadataReader
from tools.checkpoint.hf_mcore_dist.streaming import _groups
from .weight_schema import check_mcore_schema, index_specs, instantiate_schema, key_pattern, naming_specs
from tools.checkpoint.hf_mcore_dist.world_one import require_world_one


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _directions(profile):
    forward, reverse = defaultdict(list), defaultdict(list)
    for index, piece in enumerate(profile.pieces):
        forward[piece.hf].append(index)
        reverse[piece.key].append(index)
    return {"hf_to_mcore": dict(sorted(forward.items())), "mcore_to_hf": dict(sorted(reverse.items()))}


def schemas_from_plan(plan):
    refs = {name: TensorSpec(**{**spec, "shape": tuple(spec["shape"])})
            for name, spec in plan["hf"]["logical_storage_templates"].items()}
    hf_config = plan["hf"]["effective_config"]
    target_config = plan["config_mapping"]["mcore_to_hf"]["target_config"]
    return (
        instantiate_schema(hf_config, refs, plan["rules"], exact=True),
        instantiate_schema(target_config, naming_specs(refs), plan["rules"]),
    )


def _mcore_bias_enabled(args):
    return (bool(args.get("moe_router_enable_expert_bias", False))
            or args.get("moe_router_load_balancing_type") == "quantile_balancing")


def mcore_read_profile(source, config, args, rules):
    """Separate stored MCore regions from neutral HF-only router buffers.

    Callers must validate the saved args against actual source keys first.
    An enabled-but-missing bias is an error, never a generated tensor.
    """
    profile = compile_profile(source, config, derive_compress_ratios(config), rules=rules)
    if _mcore_bias_enabled(args):
        return profile, []
    generated = []
    for piece in profile.pieces:
        if not (piece.hf.endswith(".ffn.gate.bias") and piece.key.endswith(".mlp.router.expert_bias")):
            continue
        if (config["scoring_func"] not in {"sigmoid", "sqrtsoftplus"}
                or args.get("moe_router_load_balancing_type") == "sinkhorn"):
            raise MappingError("Zero router-bias export is only defined for sigmoid/sqrtsoftplus top-k routing")
        spec = source.specs[piece.hf]
        if spec.kind != "dense" or len(spec.shape) != 1 or piece.hf_index is not None:
            raise MappingError(f"Invalid disabled router-bias output template: {piece.hf}")
        generated.append({"hf": piece.hf, "name": spec.name, "shape": list(spec.shape),
                          "dtype": "F32", "value": 0, "operation": "zero_disabled_router_bias"})
    names = {item["hf"] for item in generated}
    if len(names) != len(generated):
        raise MappingError("Duplicate generated router-bias targets")
    stored = Profile([piece for piece in profile.pieces if piece.hf not in names],
                     {name: shape for name, shape in profile.shapes.items() if name not in names},
                     profile.name, profile.revision)
    return stored, sorted(generated, key=lambda item: item["hf"])


def _check_router_bias_layout(args, config, keys, *, label="Saved MCore"):
    # TopKRouter never registers expert_bias on hash layers, even when the
    # global flag is true. MTP always uses the non-hash routing path.
    eligible = {stored + "mlp.router.expert_bias"
                for scope, _, _, stored, _, is_hash in rule_contexts(config, derive_compress_ratios(config))
                if scope == "layer" and not is_hash}
    expected = eligible if _mcore_bias_enabled(args) else set()
    actual = {key for key in keys if key.endswith(".mlp.router.expert_bias")}
    if actual != expected:
        raise MappingError(f"{label} expert-bias flag/hash layout disagrees with the keys: "
                           f"missing={sorted(expected - actual)[:20]}, unexpected={sorted(actual - expected)[:20]}")


def _compile_candidate(hf_config, mc_args, weight_map, mc_schema, rules, reader, *, skip_mtp):
    translations = translate_configs(hf_config, mc_args)
    weights = {key: value for key, value in weight_map.items()
               if not (skip_mtp and is_hf_mtp_key(key))}
    tensors = {key: value for key, value in mc_schema.items()
               if not (skip_mtp and is_mcore_mtp_key(key))}
    refs = (PrecisionSource(reader, hf_config, skip_mtp=skip_mtp).specs
            if reader is not None else index_specs(weights))
    hf_source = instantiate_schema(hf_config, refs, rules, exact=True)
    if reader is not None:
        for name, spec in refs.items():
            if is_hf_mtp_key(name) and spec.shape != hf_source.specs[name].shape:
                raise MappingError(f"MTP shape does not match its HF config: {name}: "
                                   f"{spec.shape} != {hf_source.specs[name].shape}")
    bias = hf_config.get("moe_router_enable_expert_bias",
                         any(name.endswith("ffn.gate.bias") for name in hf_source.keys))
    translations["hf_to_mcore"]["target_args"]["moe_router_enable_expert_bias"] = bias
    translations["hf_to_mcore"]["fields"].append({
        "hf": "weight_keys:ffn.gate.bias", "mcore": ["moe_router_enable_expert_bias"],
        "operation": ("copy_source_model_setting" if "moe_router_enable_expert_bias" in hf_config
                      else "presence_in_existing_index"),
        "template_value": None,
        "source_value": bias,
    })
    forward, removed = mcore_read_profile(hf_source, hf_config, {"moe_router_enable_expert_bias": bias}, rules)
    ignored_bias = [item["hf"] for item in removed]
    translations["hf_to_mcore"]["ignored_zero_biases"] = ignored_bias
    _groups(forward)
    _check_router_bias_layout(translations["hf_to_mcore"]["target_args"], hf_config, forward.keys,
                             label="HF-derived MCore")
    if mc_args is None:
        return translations, refs, forward, None
    target_config = translations["mcore_to_hf"]["target_config"]
    _check_router_bias_layout(mc_args, target_config, tensors)
    mc_source = instantiate_schema(target_config, naming_specs(refs), rules)
    reverse, generated = mcore_read_profile(mc_source, target_config, mc_args, rules)
    translations["mcore_to_hf"]["generated_tensors"] = generated
    translations["mcore_to_hf"]["fields"].append({
        "hf": "weight_keys:ffn.gate.bias",
        "mcore": ["moe_router_enable_expert_bias", "moe_router_load_balancing_type", "moe_n_hash_layers"],
        "operation": "source_router_bias_layout", "template_value": bias,
        "source_value": _mcore_bias_enabled(mc_args),
    })
    _groups(reverse)
    check_mcore_schema(reverse, tensors)
    missing = {key_pattern(key) for key in forward.keys} - {key_pattern(key) for key in tensors}
    # The fixed rule supports bias even when the MCore reference disables it
    # or has only hash layers. Both source layouts were strictly checked above;
    # this exemption affects reference examples, never required source weights.
    missing.difference_update({"decoder.layers.{n}.mlp.router.expert_bias",
                               "mtp.layers.{n}.transformer_layer.mlp.router.expert_bias"})
    if missing:
        raise MappingError(f"Existing MCore checkpoint lacks module key templates: {sorted(missing)[:20]}")
    return translations, refs, forward, reverse


def emit_plan_warnings(plan):
    for entry in plan["warnings"]:
        warnings.warn(entry["message"], UserWarning, stacklevel=2)


def prepare_mapping_plan(hf_path, checkpoint_path=None, override=None, *, hf_headers=False, rules=None,
                         emit_warnings=True, vocab_size_override=None):
    require_world_one()
    if rules is not None and override:
        raise MappingError("Cannot override frozen mapping rules")
    rules = load_rules(override) if rules is None else load_rules(frozen=rules)
    hf = read_hf_artifacts(hf_path)
    # HF import has no MCore source checkpoint. A real MCore checkpoint is read
    # only when planning export of that checkpoint's own model weights.
    mcore = read_mcore_artifacts(checkpoint_path) if checkpoint_path is not None else None
    mc_args = mcore["args"] if mcore is not None else None
    mc_tensors = mcore["model_tensors"] if mcore is not None else {}
    clean_config, ignored_config = strip_inference_config(hf["config"], emit=False)
    notes = []
    if ignored_config:
        notes.append({"code": "ignored_inference_config", "fields": ignored_config,
                      "message": "Ignoring unsupported inference-only HF config fields: " + ", ".join(ignored_config)})
    if vocab_size_override is not None and mc_args is not None:
        mc_args = dict(mc_args)
        mc_args["vocab_size"] = int(vocab_size_override)
    reader = SafetensorMetadataReader(hf_path) if hf_headers else None
    headers = None
    if reader is not None:
        shapes, dtypes = reader.tensor_shapes(), reader.tensor_dtypes()
        headers = {name: {"shape": list(shapes[name]), "dtype": dtypes[name]} for name in sorted(reader.keys)}
    hf_mtp = sorted(key for key in hf["weight_map"] if is_hf_mtp_key(key))
    mc_mtp = sorted(key for key in mc_tensors if is_mcore_mtp_key(key))
    has_mtp = bool(hf_mtp or mc_mtp or clean_config.get("num_nextn_predict_layers")
                   or (mc_args or {}).get("mtp_num_layers"))
    effective_hf, effective_mc = clean_config, mc_args
    dropped = False
    if has_mtp:
        # Prove that every non-MTP tensor and weight-affecting config is valid
        # before considering a removable auxiliary-branch failure. Never catch
        # decoder/root errors and relabel them as an MTP warning.
        main_hf = without_mtp(clean_config)
        main_mc = without_mtp(mc_args, mcore=True) if mc_args is not None else None
        main_candidate = _compile_candidate(main_hf, main_mc, hf["weight_map"],
                                             mc_tensors, rules, reader, skip_mtp=True)
        try:
            candidate = _compile_candidate(clean_config, mc_args, hf["weight_map"],
                                             mc_tensors, rules, reader, skip_mtp=False)
        except (MappingError, PrecisionError, ConversionConfigurationError) as error:
            dropped, effective_hf, effective_mc, candidate = True, main_hf, main_mc, main_candidate
            notes.append({"code": "dropped_incompatible_mtp", "reason": str(error),
                          "hf_tensor_count": len(hf_mtp), "mcore_tensor_count": len(mc_mtp),
                          "message": f"Dropping the entire incompatible MTP branch ({len(hf_mtp)} HF entries, "
                                     f"{len(mc_mtp)} MCore tensors); output MTP layers=0. Reason: {error}"})
    else:
        candidate = _compile_candidate(clean_config, mc_args, hf["weight_map"],
                                         mc_tensors, rules, reader, skip_mtp=False)
    translations, refs, forward, reverse = candidate
    if mcore is None:
        ignored_bias = translations["hf_to_mcore"]["ignored_zero_biases"]
        if ignored_bias:
            notes.append({"code": "drop_generated_hf_bias", "keys": ignored_bias, "message":
                          f"Removing {len(ignored_bias)} HF-only generated router biases from MCore; "
                          "each must still be zero at conversion time; expert bias remains disabled"})
    translations["ignored_inference_config"] = ignored_config
    translations["mtp_dropped"] = dropped
    generated = translations.get("mcore_to_hf", {}).get("generated_tensors", [])
    if generated:
        notes.append({"code": "zero_disabled_router_bias", "direction": "mcore_to_hf",
                      "keys": [item["hf"] for item in generated],
                      "message": f"Saved MCore expert bias is disabled; MCore-to-HF will emit {len(generated)} "
                                 "zero FP32 router-bias buffers required by the HF template. "
                                 "These buffers are generated outputs, not missing checkpoint weights."})
    for row in translations["hf_to_mcore"]["fields"]:
        if row["operation"] == "fixed_profile_default" and row["hf"].startswith("rope_scaling."):
            notes.append({"code": "fixed_hf_yarn_default", "field": row["hf"],
                          "message": f"HF {row['hf']} is absent; using fixed DSv4 model setting {row['source_value']!r}"})
    for row in translations.get("mcore_to_hf", {}).get("fields", []):
        if row["operation"] == "fixed_mcore_default":
            field, value = row["mcore"][0], row["source_value"]
            notes.append({"code": "fixed_mcore_yarn_default", "field": field, "value": value,
                          "message": f"Saved MCore args omit {field}; using the fixed DSv4 "
                                     f"MLATransformerConfig default {field}={value!r} for {row['hf']}"})
    weight_mapping = {"hf_to_mcore": {**forward.report(), **_directions(forward),
                      "ignored_zero_biases": translations["hf_to_mcore"]["ignored_zero_biases"]}}
    if reverse is not None:
        weight_mapping["mcore_to_hf"] = {**reverse.report(), **_directions(reverse), "generated_tensors": generated}
    plan = _json({
        "schema_version": 4, "kind": "dsv4_mapping_plan", "profile": "deepseek_v4",
        "mode": "mcore_to_hf" if mcore is not None else "hf_to_mcore",
        "checkpoint_content": "model_only",
        "rules": rules, "config_mapping": translations, "warnings": notes,
        "mtp_dropped": dropped,
        "hf": {**hf, "effective_config": effective_hf, "headers": headers,
               "ignored_keys": hf_mtp if dropped else [],
               "logical_storage_templates": {name: asdict(spec) for name, spec in sorted(refs.items())}},
        "mcore": ({**mcore, "effective_args": effective_mc,
                   "ignored_keys": mc_mtp if dropped else []} if mcore is not None else None),
        "weight_mapping": weight_mapping,
        "validation": {
            "configs": ("actual source MCore args and existing HF config" if mcore is not None
                        else "existing HF config and fixed DSv4 model rules; no MCore reference"),
            "keys": "all retained keys are mapped; isolated incompatible MTP and proven generated zero biases may be omitted",
            "dimensions": "each direction uses source config; HF reference dimensions may differ",
            "hf_storage": ("headers and scale layout" if hf_headers else
                           "index only; export precision comes from actual MCore storage" if mcore is not None else
                           "index only; actual HF storage checked before model import"),
            "precision": "actual source storage; reference precision never triggers quantization",
            "tensor_values": "not loaded; source shape/region checks run before conversion",
        },
    })
    if emit_warnings:
        emit_plan_warnings(plan)
    return plan

def load_mapping_plan(path, hf_path, checkpoint_path=None, *, vocab_size_override=None):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise MappingError(f"Duplicate mapping plan field: {key}")
            result[key] = value
        return result

    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    if not isinstance(plan, dict) or plan.get("schema_version") != 4 or plan.get("kind") != "dsv4_mapping_plan":
        raise MappingError("Regenerate a v4 model-only mapping plan; HF import no longer accepts a MCore reference")
    mode = "mcore_to_hf" if checkpoint_path is not None else "hf_to_mcore"
    if plan.get("mode") != mode:
        raise MappingError(f"Mapping plan is not for {mode}; regenerate it for the actual source")
    current = prepare_mapping_plan(hf_path, checkpoint_path, rules=plan["rules"],
                                   hf_headers=plan["hf"]["headers"] is not None, emit_warnings=False,
                                   vocab_size_override=vocab_size_override)
    # Compare structural content directly, never hash files. Re-reading small
    # metadata also rejects stale args, names, dimensions and edited plan regions.
    if current != plan:
        raise MappingError("Existing configs/keys/metadata or mapping rules differ from the prepared plan; regenerate it")
    emit_plan_warnings(plan)
    return plan
