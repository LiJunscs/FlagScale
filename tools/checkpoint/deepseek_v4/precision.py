# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""DSv4 naming and expert-format declarations for the common quantization engine."""
import re
from tools.checkpoint.hf_mcore_dist import quantization as _impl
from tools.checkpoint.hf_mcore_dist.quantization import (
    encode_hf_tensors, output_config, validate_quantization_config,
)
from tools.checkpoint.hf_mcore_dist.precision import (
    DENSE_DTYPES, HEADER_DTYPES, FP8_DTYPES, E8M0_DTYPES, SCALE_SUFFIXES,
    PrecisionError, TensorSpec, PrecisionPolicy,
)

def canonical_hf_name(name: str) -> str:
    """Normalize the published HF and native inference names to one schema."""
    name = name.removeprefix("model.")
    replacements = {
        "embed_tokens": "embed", "lm_head": "head", "self_attn": "attn",
        "mlp": "ffn", "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm", "q_a_proj": "wq_a",
        "q_a_layernorm": "q_norm", "q_b_proj": "wq_b",
        "gate_proj": "w1", "up_proj": "w3", "down_proj": "w2",
        "e_score_correction_bias": "bias",
    }
    name = ".".join(replacements.get(part, part) for part in name.split("."))
    return name.replace(".indexer.weights_proj.", ".indexer.scorer.weights_proj.")

def is_hf_mtp_key(name: str) -> bool:
    return re.match(r"^mtp\.\d+\.", canonical_hf_name(name)) is not None

def _output_quantization_layout(specs):
    """Describe the observed restored layout; never invent a quantized module."""
    modules, expert_groups = set(), {}
    for name, spec in specs.items():
        module = "lm_head" if name == "head.weight" else spec.name.removesuffix(".weight")
        expert = re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", name)
        if expert:
            module = spec.name.rsplit(".experts.", 1)[0] + ".experts"
            expert_groups.setdefault(module, set()).add(spec.kind)
        elif name.endswith(".weight") and spec.kind == "dense":
            modules.add(module)
    expert_kinds = set()
    for module, kinds in expert_groups.items():
        if len(kinds) > 1:
            raise PrecisionError(f"Mixed storage within one routed-expert module is unsupported: {module}")
        if kinds == {"dense"}:
            modules.add(module)
        else:
            expert_kinds.update(kinds)
    if len(expert_kinds) > 1:
        raise PrecisionError("A single expert_dtype cannot describe multiple routed-expert formats")
    expert_dtype = {"mxfp4": "fp4", "fp8": "fp8"}.get(next(iter(expert_kinds), "dense"))
    return tuple(sorted(modules)), expert_dtype


class PrecisionSource(_impl.PrecisionSource):
    def __init__(self, reader, config, *, skip_mtp=False, **kwargs):
        super().__init__(reader, config, normalize=canonical_hf_name,
                         skip=is_hf_mtp_key if skip_mtp else None, strict_scales=True,
                         mxfp4=lambda name: config.get("expert_dtype") == "fp4" and bool(re.fullmatch(
                             r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight",
                             canonical_hf_name(name))), **kwargs)


def resolve_precision(*args, **kwargs):
    return _impl.resolve_precision(*args, layout=_output_quantization_layout, **kwargs)
