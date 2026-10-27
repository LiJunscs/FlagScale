# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Versioned DSv4 model-only configuration translations.

CLI args and TransformerConfig names are not interchangeable (num_experts,
swiglu, fp8_param_gather, etc.). HF import derives model fields from its own
config and pinned DSv4 rules, never another training run's args. MCore export
reads the actual source model's args. No models or library defaults are probed.
"""
from __future__ import annotations

from copy import deepcopy
import warnings

from .hf_config import derive_compress_ratios, validate_hf_config
from . import MappingError
from .hf_config import num_hash_layers

from pathlib import Path
import json
from tools.checkpoint.hf_mcore_dist.config import (
    MCORE_OUTPUT_OVERRIDES, get_alias as _get, record as _record,
    resolve, translate_fields,
)

PARAMETERS = json.loads(Path(__file__).with_name("parameters.json").read_text())
REVISION = PARAMETERS["REVISION"]
FIELD_RULES = PARAMETERS["fields"]
FIELDS = [(row["hf"], row["mcore"]) for row in FIELD_RULES]
REQUIRED = {row["hf"] for row in FIELD_RULES if row.get("required")}
YARN = PARAMETERS["YARN"]
MCORE_YARN_DEFAULTS = PARAMETERS["MCORE_YARN_DEFAULTS"]
HF_YARN_DEFAULTS = PARAMETERS["HF_YARN_DEFAULTS"]
MCORE_MODEL_FIELDS = {aliases[0] for _, aliases in FIELDS} | set(YARN.values()) | {
    "vocab_size", "padded_vocab_size", "actual_vocab_size", "make_vocab_size_divisible_by",
    "moe_shared_expert_intermediate_size", "csa_compress_ratios", "swiglu", "gated_linear_unit",
    "normalization", "untie_embeddings_and_output_weights", "multi_latent_attention",
    "qk_layernorm", "moe_layer_freq", "experimental_attention_variant", "enable_hyper_connections",
    "rotary_percent", "rotary_interleaved", "add_bias_linear", "add_qkv_bias", "position_embedding_type",
    "qk_head_dim", "kv_lora_rank", "kv_channels", "rope_type", "moe_router_enable_expert_bias",
}

# These fields control DSpark inference. They do not identify state-dict
# tensors; actual keys are checked independently by the mapping compiler.
INFERENCE_ONLY_FIELDS = set(PARAMETERS["INFERENCE_ONLY_FIELDS"])


def strip_inference_config(config, *, emit=True):
    result = deepcopy(config)
    ignored = {key: result.pop(key) for key in sorted(INFERENCE_ONLY_FIELDS & result.keys())}
    if ignored and emit:
        warnings.warn("Ignoring unsupported inference-only HF config fields: " + ", ".join(ignored),
                      UserWarning, stacklevel=2)
    return result, ignored


def without_mtp(config, *, mcore=False):
    """Remove the independent prediction branch, preserving all main layers."""
    result = deepcopy(config)
    layers = int(result["num_layers" if mcore else "num_hidden_layers"])
    result["mtp_num_layers" if mcore else "num_nextn_predict_layers"] = 0
    ratios = "csa_compress_ratios" if mcore else "compress_ratios"
    if isinstance(result.get(ratios), (list, tuple)):
        result[ratios] = list(result[ratios][:layers])
    if mcore:
        if "mtp_loss_scaling_factor" in result:
            result["mtp_loss_scaling_factor"] = 0.0
    else:
        if isinstance(result.get("layer_types"), list):
            result["layer_types"] = result["layer_types"][:layers]
        for key in list(result):
            if key.startswith("mtp_"):
                result.pop(key)
    return result


def _check_mcore(args):
    if args.get("experimental_attention_variant") != "dsv4_hybrid":
        raise MappingError("Saved args must declare experimental_attention_variant=dsv4_hybrid")
    for key in ("use_engram", "csa_dense_mode", "mla_down_proj_fusion",
                "attention_output_gate", "moe_router_pre_softmax", "rotary_interleaved"):
        if args.get(key):
            raise MappingError(f"Unsupported DSv4 model semantics: {key}={args[key]!r}")
    if args.get("normalization") != "RMSNorm" or args.get("swiglu") is not True:
        raise MappingError("Saved args must explicitly describe RMSNorm + SwiGLU")
    if args.get("enable_hyper_connections") is not True:
        raise MappingError("DSv4 requires enable_hyper_connections=True")
    if args.get("untie_embeddings_and_output_weights") is not True:
        raise MappingError("DSv4 requires untied MCore embeddings")
    if args.get("add_bias_linear", False) or args.get("add_qkv_bias", False):
        raise MappingError("Linear/attention biases are outside this DSv4 profile")
    freq = args.get("moe_layer_freq", 1)
    if freq != 1 and freq != [1] * int(args["num_layers"]):
        raise MappingError("Only all-MoE DSv4 layers are supported")
    for key in ("rotary_base_per_layer", "no_rope_freq"):
        if args.get(key) is not None:
            raise MappingError(f"No HF translation is defined for {key}")
    if args.get("rotary_percent", 1.0) != 1.0:
        raise MappingError("The DSv4 profile requires rotary_percent=1")
    epsilon = _get(args, ("norm_epsilon", "layernorm_epsilon"))
    for key in ("attention_latent_norm_epsilon", "dsa_indexer_k_norm_epsilon"):
        if args.get(key) not in (None, epsilon):
            raise MappingError(f"No separate HF normalization epsilon mapping is defined for {key}")


def _check_hf(config):
    validate_hf_config(config)
    factor = config.get("make_vocab_size_divisible_by", 1)
    if type(factor) is not int or factor <= 0:
        raise MappingError("make_vocab_size_divisible_by must be a positive integer")
    if "moe_router_enable_expert_bias" in config and type(config["moe_router_enable_expert_bias"]) is not bool:
        raise MappingError("moe_router_enable_expert_bias must be a boolean")
    if config.get("hidden_act", "silu") != "silu" or config.get("attention_bias", False):
        raise MappingError("Only bias-free DSv4 SiLU attention/MLP is supported")
    if int(config.get("first_k_dense_replace", 0) or 0):
        raise MappingError("Dense initial layers are outside the DSv4 profile")
    if int(config.get("num_key_value_heads", 1)) != 1:
        raise MappingError("DSv4 requires one shared KV head")
    if int(config.get("n_shared_experts", 1) or 0) < 1:
        raise MappingError("The fixed profile requires shared experts")
    if config.get("norm_topk_prob", True) is not True:
        raise MappingError("This DSv4 router requires normalized selected expert probabilities")
    hash_layers = num_hash_layers(config)
    if not 0 <= hash_layers <= int(config["num_hidden_layers"]):
        raise MappingError("Invalid number of hash MoE layers")
    layout = config.get("mlp_layer_types")
    expected = ["hash_moe" if i < hash_layers else "moe" for i in range(int(config["num_hidden_layers"]))]
    if layout is not None and layout != expected:
        raise MappingError("mlp_layer_types must describe leading hash MoE layers followed by MoE")
    for key in REQUIRED - {"scoring_func", "attention_dropout"} | {"vocab_size"}:
        if config.get(key) is None or not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise MappingError(f"HF config needs a positive explicit {key}")
    if int(config["head_dim"]) <= int(config["qk_rope_head_dim"]):
        raise MappingError("head_dim must exceed qk_rope_head_dim")
    if int(config["num_attention_heads"]) % int(config["o_groups"]):
        raise MappingError("num_attention_heads must be divisible by o_groups")
    if int(config["num_experts_per_tok"]) > int(config["n_routed_experts"]):
        raise MappingError("Expert top-k exceeds the number of experts")


def mcore_to_hf_config(args, template, *, vocab_size_override=None):
    template, ignored = strip_inference_config(template)
    _check_mcore(args)
    _check_hf(template)
    result, rows = deepcopy(template), []
    for key, value in ignored.items():
        _record(rows, key, (), "ignore_inference_config", value, None)
    result, field_rows = translate_fields(FIELD_RULES, args, False, template=result)
    rows.extend(field_rows)
    # Embedding, output and tid2eid share this unpadded row count. The streaming
    # reader crops stored padding; neither padded size nor an unrelated HF
    # reference can establish the source model's real vocabulary. Checkpoints
    # produced by --use-checkpoint-args carry vocab_size=None (the unpadded
    # value is not recoverable from mcore state), so it must be supplied
    # explicitly via --vocab-size for such sources.
    vocab = _get(args, ("vocab_size",))
    if vocab is None and vocab_size_override is not None:
        vocab = int(vocab_size_override)
        _record(rows, "vocab_size", (), "cli_vocab_size_override", template.get("vocab_size"), vocab)
    if vocab is None:
        raise MappingError("MCore-to-HF requires the source model's unpadded vocab_size "
                           "(checkpoint args.vocab_size is None; pass --vocab-size); "
                           "padded_vocab_size cannot determine how many rows to trim")
    result["vocab_size"] = int(vocab)
    _record(rows, "vocab_size", ("vocab_size",), "copy_unpadded_vocab_size",
            template.get("vocab_size"), int(vocab))
    # Carry model settings directly; tensor shapes still follow the HF vocabulary.
    result["make_vocab_size_divisible_by"] = args.get("make_vocab_size_divisible_by", 1)
    result["moe_router_enable_expert_bias"] = (
        bool(args.get("moe_router_enable_expert_bias", False))
        or args.get("moe_router_load_balancing_type") == "quantile_balancing")
    for field in ("make_vocab_size_divisible_by", "moe_router_enable_expert_bias"):
        _record(rows, field, (field,), "copy_source_model_setting", template.get(field), result[field])
    shared = int(_get(args, ("moe_shared_expert_intermediate_size",), required=True))
    ffn = int(result["moe_intermediate_size"])
    if shared < ffn or shared % ffn:
        raise MappingError("Shared expert intermediate size must be a positive multiple of MoE FFN size")
    result["n_shared_experts"] = shared // ffn
    _record(rows, "n_shared_experts", ("moe_shared_expert_intermediate_size", "moe_ffn_hidden_size"),
            "divide", template.get("n_shared_experts"), result["n_shared_experts"])
    result["compress_ratios"] = list(_get(args, ("csa_compress_ratios",), required=True))
    _record(rows, "compress_ratios", ("csa_compress_ratios",), "layout",
            template.get("compress_ratios"), result["compress_ratios"])
    result.update(hidden_act="silu", attention_bias=False, tie_word_embeddings=False,
                  num_key_value_heads=1, first_k_dense_replace=0, norm_topk_prob=True)
    _record(rows, "hidden_act", ("swiglu",), "silu", template.get("hidden_act"), "silu")
    _record(rows, "tie_word_embeddings", ("untie_embeddings_and_output_weights",),
            "not", template.get("tie_word_embeddings"), False)
    rope_type = args.get("rope_type") or "yarn"  # Saved None selects MLATransformerConfig's fixed default.
    if rope_type not in ("rope", "yarn"):
        raise MappingError(f"Unsupported MCore rope_type: {rope_type}")
    if rope_type == "yarn":
        result["rope_scaling"] = {"rope_type": "yarn"}
        for hf, mc in YARN.items():
            operation = "copy"
            if mc not in args and mc in MCORE_YARN_DEFAULTS:
                value = MCORE_YARN_DEFAULTS[mc]
                operation = "fixed_mcore_default"
            else:
                value = _get(args, (mc,), required=True)
            result["rope_scaling"][hf] = value
            _record(rows, "rope_scaling." + hf, (mc,), operation,
                    (template.get("rope_scaling") or {}).get(hf), value)
    else:
        result["rope_scaling"] = None
    _record(rows, "rope_scaling.rope_type", ("rope_type",), "rope_or_yarn",
            (template.get("rope_scaling") or {}).get("rope_type"), rope_type)
    _refresh_layout(result)
    _check_hf(result)
    return result, rows


def _refresh_layout(config):
    ratios = derive_compress_ratios(config)
    types = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    config["compress_ratios"] = ratios
    config["layer_types"] = [types[x] for x in ratios[:int(config["num_hidden_layers"])]]
    config["compress_rates"] = {types[4]: 4, types[128]: 128}
    config["mlp_layer_types"] = [
        "hash_moe" if i < int(config.get("num_hash_layers", 0)) else "moe"
        for i in range(int(config["num_hidden_layers"]))
    ]


def hf_to_mcore_args(config):
    """Build only model/format fields; there is deliberately no args template."""
    config, ignored = strip_inference_config(config)
    _check_hf(config)
    config = deepcopy(config)
    config["num_hash_layers"] = num_hash_layers(config)
    result, rows = {}, []
    for key, value in ignored.items():
        _record(rows, key, (), "ignore_inference_config", value, None)
    result, field_rows = translate_fields(FIELD_RULES, config, True)
    rows.extend(field_rows)
    config.setdefault("n_shared_experts", 1)
    config.setdefault("tie_word_embeddings", False)
    result.update(resolve(PARAMETERS["mcore_values"], {"hf": config, "mcore": result}))
    result.update(make_vocab_size_divisible_by=config.get("make_vocab_size_divisible_by", 1),
                  csa_compress_ratios=derive_compress_ratios(config))
    _record(rows, "make_vocab_size_divisible_by", ("make_vocab_size_divisible_by",),
            "copy" if "make_vocab_size_divisible_by" in config else "fixed_profile_default",
            None, result["make_vocab_size_divisible_by"])
    if "moe_router_enable_expert_bias" in config:
        result["moe_router_enable_expert_bias"] = config["moe_router_enable_expert_bias"]
    for hf, mc, op in (
        ("vocab_size", ("vocab_size", "padded_vocab_size", "actual_vocab_size"), "unpadded_world_one"),
        ("n_shared_experts", ("moe_shared_expert_intermediate_size",), "multiply_moe_intermediate_size"),
        ("compress_ratios", ("csa_compress_ratios",), "layout"),
        ("hidden_act", ("swiglu",), "silu"),
        ("tie_word_embeddings", ("untie_embeddings_and_output_weights",), "not"),
        ("head_dim-qk_rope_head_dim", ("qk_head_dim", "kv_lora_rank"), "subtract"),
    ):
        _record(rows, hf, mc, op, None,
                {key: result.get(key) for key in mc})
    scaling = config.get("rope_scaling") or {}
    rope_type = scaling.get("rope_type", scaling.get("type", "default"))
    if rope_type not in ("default", "rope", "yarn"):
        raise MappingError(f"Unsupported HF rope scaling: {rope_type}")
    result["rope_type"] = "yarn" if rope_type == "yarn" else "rope"
    if rope_type == "yarn":
        for hf, mc in YARN.items():
            if hf not in scaling:
                # DSv4 applies pure rotation. Pin the local MLA defaults for
                # optional concentration fields, independent of other models.
                if hf in HF_YARN_DEFAULTS:
                    result[mc] = HF_YARN_DEFAULTS[hf]
                    _record(rows, "rope_scaling." + hf, (mc,), "fixed_profile_default",
                            None, result[mc])
                    continue
                raise MappingError(f"HF rope_scaling must contain {hf}; export a complete config")
            if scaling[hf] is None:
                raise MappingError(f"HF rope_scaling.{hf} cannot be null")
            result[mc] = scaling[hf]
            _record(rows, "rope_scaling." + hf, (mc,), "copy", None, scaling[hf])
    _record(rows, "rope_scaling.rope_type", ("rope_type",), "rope_or_yarn",
            None, result["rope_type"])
    result.update(MCORE_OUTPUT_OVERRIDES)
    _record(rows, "conversion_context", tuple(MCORE_OUTPUT_OVERRIDES), "world_one_model_only_checkpoint",
            None, MCORE_OUTPUT_OVERRIDES)
    _check_mcore(result)
    return result, rows


def translate_configs(hf, mcore=None, *, vocab_size_override=None):
    exported_mcore, forward = hf_to_mcore_args(hf)
    result = {
        "revision": REVISION,
        "hf_to_mcore": {"target_args": exported_mcore, "fields": forward},
        "source_of_dimensions": {"hf_to_mcore": "HF config"},
        "precision": "HF storage quantization is handled separately; MCore fp8/fp4 training recipes are not HF byte formats",
    }
    if mcore is not None:
        exported_hf, reverse = mcore_to_hf_config(mcore, hf, vocab_size_override=vocab_size_override)
        result.update(
            mcore_to_hf={"target_config": exported_hf, "fields": reverse},
            hf_template_only_fields=sorted(set(hf) - {row["hf"].split(".")[0] for row in reverse}),
            mcore_unmapped_source_fields=sorted(set(mcore) - {key for row in reverse for key in row["mcore"]}),
        )
        result["source_of_dimensions"]["mcore_to_hf"] = "saved MCore args"
    return result
