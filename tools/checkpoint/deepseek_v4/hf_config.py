# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Validation of an existing DSv4 HF configuration; no library/model probing."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from .precision import validate_quantization_config

_LAYER_TYPE_TO_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}
_SUPPORTED_RATIOS = frozenset(_LAYER_TYPE_TO_RATIO.values())
_REQUIRED_HYBRID_FIELDS = (
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_groups",
    "o_lora_rank",
    "sliding_window",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "hc_mult",
    "hc_sinkhorn_iters",
    "n_routed_experts",
)


class ConversionConfigurationError(ValueError):
    pass


def read_hf_config(hf_model: str | Path) -> dict[str, Any]:
    model_path = Path(hf_model).expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ConversionConfigurationError(
            f"Missing Hugging Face config: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConversionConfigurationError(f"Expected a JSON object in {config_path}")
    return config


def derive_compress_ratios(config: Mapping[str, Any]) -> list[int]:
    num_layers = int(config["num_hidden_layers"])
    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0) or 0)
    expected = num_layers + num_mtp_layers
    raw_ratios = config.get("compress_ratios")
    if raw_ratios is not None:
        ratios = [int(ratio) for ratio in raw_ratios]
    else:
        layer_types = config.get("layer_types")
        compress_rates = config.get("compress_rates")
        if layer_types is None or compress_rates is None:
            raise ConversionConfigurationError(
                "DeepSeek-V4 config must define compress_ratios or layer_types plus compress_rates."
            )
        ratios = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                ratios.append(0)
            elif layer_type in compress_rates:
                ratios.append(int(compress_rates[layer_type]))
            elif layer_type in _LAYER_TYPE_TO_RATIO:
                ratios.append(_LAYER_TYPE_TO_RATIO[layer_type])
            else:
                raise ConversionConfigurationError(
                    f"Unsupported attention layer type: {layer_type!r}"
                )
    if len(ratios) == num_layers and num_mtp_layers:
        ratios.extend([0] * num_mtp_layers)
    if len(ratios) != expected:
        raise ConversionConfigurationError(
            f"Compression layout has {len(ratios)} entries; expected {expected}."
        )
    unsupported = sorted(set(ratios) - _SUPPORTED_RATIOS)
    if unsupported:
        raise ConversionConfigurationError(
            f"Unsupported compression ratios {unsupported}; only 0, 4 and 128 are accepted."
        )
    if any(ratios[num_layers:]):
        raise ConversionConfigurationError("This DSv4 profile requires sliding-only MTP layers")
    return ratios


def validate_hf_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_type = str(config.get("model_type", ""))
    architectures = [str(item) for item in config.get("architectures", [])]
    architecture_match = any(
        "DeepseekV4" in name or "DeepSeekV4" in name for name in architectures
    )
    if model_type != "deepseek_v4" and not architecture_match:
        raise ConversionConfigurationError(
            "The source config is not recognized as DeepSeek-V4."
        )
    missing = [
        name
        for name in ("num_hidden_layers", *_REQUIRED_HYBRID_FIELDS)
        if config.get(name) is None
    ]
    if missing:
        raise ConversionConfigurationError(
            "Missing required DSv4 fields: " + ", ".join(missing)
        )
    if bool(config.get("tie_word_embeddings", False)):
        raise ConversionConfigurationError(
            "This DSv4 business converter requires untied embed.weight and head.weight."
        )
    validate_quantization_config(config)
    ratios = derive_compress_ratios(config)
    return {
        "model_type": model_type or None,
        "architectures": architectures,
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_mtp_layers": int(config.get("num_nextn_predict_layers", 0) or 0),
        "num_hash_layers": int(config.get("num_hash_layers", 0) or 0),
        "num_experts": int(config["n_routed_experts"]),
        "attention_layers": {
            "sliding": ratios.count(0),
            "csa": ratios.count(4),
            "hca": ratios.count(128),
        },
        "compress_ratios": ratios,
    }


def num_hash_layers(config: Mapping[str, Any]) -> int:
    if config.get("num_hash_layers") is not None:
        return int(config["num_hash_layers"])
    count = 0
    for layer_type in config.get("mlp_layer_types", []):
        if layer_type != "hash_moe":
            break
        count += 1
    return count



