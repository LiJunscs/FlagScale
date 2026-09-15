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

REVISION = "dsv4-config-v7"
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
# (HF field, saved argument aliases). First alias is the training CLI spelling.
FIELDS = (
    ("num_hidden_layers", ("num_layers",)),
    ("hidden_size", ("hidden_size",)),
    ("intermediate_size", ("ffn_hidden_size",)),
    ("num_attention_heads", ("num_attention_heads",)),
    ("n_routed_experts", ("num_experts", "num_moe_experts")),
    ("moe_intermediate_size", ("moe_ffn_hidden_size",)),
    ("q_lora_rank", ("q_lora_rank",)),
    ("qk_rope_head_dim", ("qk_pos_emb_head_dim",)),
    ("head_dim", ("v_head_dim",)),
    ("o_groups", ("o_groups",)), ("o_lora_rank", ("o_lora_rank",)),
    ("sliding_window", ("csa_window_size",)),
    ("index_n_heads", ("dsa_indexer_n_heads",)),
    ("index_head_dim", ("dsa_indexer_head_dim",)),
    ("index_topk", ("dsa_indexer_topk",)),
    ("hc_mult", ("num_residual_streams",)),
    ("hc_sinkhorn_iters", ("mhc_sinkhorn_iterations",)),
    ("num_experts_per_tok", ("moe_router_topk",)),
    ("num_nextn_predict_layers", ("mtp_num_layers",)),
    ("num_hash_layers", ("moe_n_hash_layers",)),
    ("rms_norm_eps", ("norm_epsilon", "layernorm_epsilon")),
    ("attention_dropout", ("attention_dropout",)),
    ("scoring_func", ("moe_router_score_function",)),
    ("routed_scaling_factor", ("moe_router_topk_scaling_factor",)),
    ("n_group", ("moe_router_num_groups",)),
    ("topk_group", ("moe_router_group_topk",)),
    ("swiglu_limit", ("activation_func_clamp_value",)),
    ("max_position_embeddings", ("max_position_embeddings",)),
    ("rope_theta", ("rotary_base",)),
    ("compress_rope_theta", ("csa_compress_rotary_base",)),
)
# Missing architecture dimensions cannot be supplied by a different-sized template.
REQUIRED = {
    "num_hidden_layers", "hidden_size", "num_attention_heads", "n_routed_experts",
    "moe_intermediate_size", "q_lora_rank", "qk_rope_head_dim", "head_dim",
    "o_groups", "o_lora_rank", "sliding_window", "index_n_heads", "index_head_dim",
    "index_topk", "hc_mult", "hc_sinkhorn_iters", "num_experts_per_tok",
    "rms_norm_eps", "max_position_embeddings", "rope_theta", "compress_rope_theta",
    "scoring_func", "attention_dropout",
}
YARN = {
    "factor": "rotary_scaling_factor",
    "original_max_position_embeddings": "original_max_position_embeddings",
    "beta_fast": "beta_fast", "beta_slow": "beta_slow",
    "mscale": "mscale", "mscale_all_dim": "mscale_all_dim",
}
# Pinned MLATransformerConfig defaults in Megatron core/transformer/transformer_config.py.
# These fields have no FlagScale training CLI arguments, so common.pt args
# normally omit them. Do not borrow their values from a different HF model
# or instantiate a library config. Explicit saved values always take precedence.
MCORE_YARN_DEFAULTS = {"beta_fast": 32, "beta_slow": 1}
# Model-only defaults for absent optional HF fields. These are fixed profile
# semantics, not defaults copied from a training checkpoint.
HF_OPTIONAL_DEFAULTS = {
    "intermediate_size": None, "num_nextn_predict_layers": 0, "num_hash_layers": 0,
    "routed_scaling_factor": 1.0, "n_group": None, "topk_group": None, "swiglu_limit": None,
}
HF_YARN_DEFAULTS = {"mscale": 1.0, "mscale_all_dim": 0.0}
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
INFERENCE_ONLY_FIELDS = {
    "dspark_block_size", "dspark_noise_token_id", "dspark_target_layer_ids", "dspark_markov_rank",
}


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


def _get(args, aliases, *, required=False):
    found = [(key, args[key]) for key in aliases if args.get(key) is not None]
    if found:
        if any(value != found[0][1] for _, value in found[1:]):
            raise MappingError(f"Conflicting saved MCore aliases: {found}")
        return found[0][1]
    if required:
        raise MappingError(f"Saved MCore args are missing {aliases}; supply a complete checkpoint config")
    return None


def _set(args, aliases, value):
    args[aliases[0]] = deepcopy(value)
    for key in aliases[1:]:
        if key in args:
            args[key] = deepcopy(value)


def _record(rows, hf, mc, op, before, after):
    rows.append({"hf": hf, "mcore": list(mc), "operation": op,
                 "template_value": deepcopy(before), "source_value": deepcopy(after)})


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
    for hf, aliases in FIELDS:
        value = _get(args, aliases, required=hf in REQUIRED)
        if hf in {"num_nextn_predict_layers", "num_hash_layers"}:
            value = int(value or 0)
        elif hf in {"n_group", "topk_group", "routed_scaling_factor"} and value is None:
            value = 1  # MCore None means no grouping/scaling.
        elif value is None:
            # MCore None is meaningful for clamp/group/scaling options; never
            # carry a differently configured HF value into the exported model.
            if hf in result:
                result[hf] = None
            _record(rows, hf, aliases, "nullable", template.get(hf), None)
            continue
        result[hf] = value
        _record(rows, hf, aliases, "copy", template.get(hf), value)
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
    for hf, aliases in FIELDS:
        operation = "copy"
        if hf not in config:
            if hf in REQUIRED:
                raise MappingError(f"HF config is missing {hf}; no model/library default probing is allowed")
            value = HF_OPTIONAL_DEFAULTS[hf]
            operation = "fixed_profile_default"
        else:
            value = config[hf]
        if hf in {"num_nextn_predict_layers", "num_hash_layers"}:
            value = int(value or 0)
        _set(result, aliases, value)
        _record(rows, hf, aliases, operation, None, value)
    result.update(
        vocab_size=int(config["vocab_size"]), padded_vocab_size=int(config["vocab_size"]),
        actual_vocab_size=int(config["vocab_size"]),
        make_vocab_size_divisible_by=config.get("make_vocab_size_divisible_by", 1),
        moe_shared_expert_intermediate_size=int(config.get("n_shared_experts", 1))
            * int(config["moe_intermediate_size"]),
        csa_compress_ratios=derive_compress_ratios(config),
        swiglu=True, gated_linear_unit=True, normalization="RMSNorm",
        untie_embeddings_and_output_weights=not config.get("tie_word_embeddings", False),
        multi_latent_attention=True, qk_layernorm=True, moe_layer_freq=1,
        experimental_attention_variant="dsv4_hybrid", enable_hyper_connections=True,
        rotary_percent=1.0, rotary_interleaved=False,
        add_bias_linear=False, add_qkv_bias=False,
        position_embedding_type="rope",
        qk_head_dim=int(config["head_dim"]) - int(config["qk_rope_head_dim"]),
        kv_lora_rank=int(config["head_dim"]) - int(config["qk_rope_head_dim"]),
        kv_channels=int(config["head_dim"]),
    )
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
