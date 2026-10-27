# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""DSv4 vocabulary padding and generated router-bias policy."""
from tools.checkpoint.hf_mcore_dist import streaming as _impl
import re
import torch
from . import MappingError
from .weight_schema import mcore_shape_matches
from .precision import canonical_hf_name
from .config_mapping import MCORE_MODEL_FIELDS, MCORE_OUTPUT_OVERRIDES


def save_streaming_checkpoint(source, profile, policy, config, output_root, *, model_args,
                              max_shard_size="4GB"):
    unknown = model_args.keys() - MCORE_MODEL_FIELDS - MCORE_OUTPUT_OVERRIDES.keys()
    if unknown:
        raise MappingError(f"Model-only checkpoint args contain non-model fields: {sorted(unknown)}")
    ignored = [name for name in source.specs if name.endswith(".ffn.gate.bias")
               and model_args.get("moe_router_enable_expert_bias") is False]
    for name in ignored:
        if bool((source.get_tensor(name) != 0).any()):
            raise MappingError(f"Cannot omit nonzero HF router bias with MCore expert bias disabled: {name}")
    details = _impl.save_streaming_checkpoint(
        source, profile, policy, config, output_root, model_args=model_args,
        max_shard_size=max_shard_size, content_metadata={
            "deepseek_v4_conversion": policy.report(), "deepseek_v4_hf_config": dict(config),
            "profile_revision": "dsv4-global-v1"})
    details["removed_generated_hf_biases"] = ignored
    return details


class StreamingCheckpointReader(_impl.StreamingCheckpointReader):
    def __init__(self, checkpoint_path, profile, *, skip_mtp=False):
        super().__init__(checkpoint_path, profile, skip_mtp=skip_mtp,
                         shape_matches=mcore_shape_matches)


def hf_export_tensors(tensors, generated):
    """Stream actual weights, then materialize only planned neutral router biases."""
    names = set()
    for item in generated:
        name, shape = item["hf"], item["shape"]
        if (item["operation"] != "zero_disabled_router_bias" or item["value"] != 0
                or item["dtype"] != "F32" or not re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.gate\.bias", name)
                or canonical_hf_name(item["name"]) != name
                or len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0 or name in names):
            raise MappingError(f"Invalid generated HF router-bias target: {name}")
        names.add(name)
    for name, tensor in tensors:
        if name in names:
            raise MappingError(f"Generated HF target overlaps a stored source weight: {name}")
        yield name, tensor
    for item in generated:
        tensor = torch.zeros(tuple(item["shape"]), dtype=torch.float32, device="cpu")
        yield item["hf"], tensor
        del tensor
