# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile one DSv4 rule table into reversible global checkpoint regions.

The physical contract is non_homogeneous_layers=True and
singleton_local_shards=False. SwiGLU halves occupy different row offsets;
experts prepend a global expert axis. No runtime GPTModel is instantiated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import MappingError
from .hf_config import num_hash_layers
from .precision import canonical_hf_name


def load_rules(override=None, *, frozen=None):
    base = (json.loads(Path(__file__).with_name("modules.json").read_text(encoding="utf-8"))["rules"]
            if frozen is None else frozen)
    if not isinstance(base, list) or not base:
        raise MappingError("Mapping rules must be a nonempty list")
    rules = {}
    for rule in base:
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str) or not rule["id"]:
            raise MappingError("Mapping rule requires a nonempty id")
        if rule["id"] in rules:
            raise MappingError(f"Duplicate mapping rule: {rule['id']}")
        rules[rule["id"]] = rule
    if override:
        path = Path(override)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "profile", "rules"}:
            raise MappingError("Override requires schema_version, profile and rules only")
        if payload["schema_version"] != 1 or payload["profile"] != "deepseek_v4":
            raise MappingError("Only the deepseek_v4 v1 override contract is supported")
        seen = set()
        for rule in payload["rules"]:
            if rule["id"] in seen:
                raise MappingError(f"Duplicate override rule: {rule['id']}")
            seen.add(rule["id"])
            rules[rule["id"]] = rule  # Replace the complete rule, never merge fields.
    arities = {"identity": (1, 1), "swiglu": (2, 1), "expert": (1, 1),
               "expert_swiglu": (2, 1), "hc_scale": (1, 3)}
    for rule in rules.values():
        required = {"id", "scope", "hf", "mcore", "op"}
        if not required <= rule.keys() or rule.keys() - required - {"te", "optional", "when"}:
            raise MappingError(f"Invalid fields in rule: {rule}")
        if rule["scope"] not in ("root", "layer", "mtp") or rule["op"] not in arities:
            raise MappingError(f"Unsupported rule: {rule}")
        if rule.get("when") not in (None, "compressor", "indexer", "hash", "non_hash"):
            raise MappingError(f"Unsupported rule condition: {rule}")
        if rule["id"] == "router_bias" and (rule["scope"] != "layer" or rule.get("when") != "non_hash"):
            raise MappingError("DSv4 router_bias requires scope=layer and when=non_hash; regenerate old rules/plans")
        if (len(rule["hf"]), len(rule["mcore"])) != arities[rule["op"]]:
            raise MappingError(f"Operator arity mismatch: {rule['id']}")
        if not isinstance(rule["hf"], list) or not isinstance(rule["mcore"], list):
            raise MappingError(f"hf/mcore must be lists: {rule['id']}")
        if rule["op"].startswith("expert") and any(
            ".experts.linear_fc" not in name or not name.endswith(".weight{expert}")
            for name in rule["mcore"]
        ):
            raise MappingError("Expert operators require the fixed experts.linear_fcN.weight{expert} contract")
        for name in rule["hf"] + rule["mcore"]:
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise MappingError(f"Invalid tensor name: {name!r}")
            if "{" in name.replace("{expert}", "") or "}" in name.replace("{expert}", ""):
                raise MappingError("Only the {expert} placeholder is supported")
    return list(rules.values())


@dataclass(frozen=True)
class Piece:
    hf: str
    runtime_key: str
    key: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    global_shape: tuple[int, ...]
    hf_index: int | None = None
    extra_key: str | None = None


@dataclass
class Profile:
    pieces: list[Piece]
    shapes: dict[str, tuple[int, ...]]

    @property
    def keys(self):
        return {piece.key for piece in self.pieces}

    def report(self):
        return {"profile": "deepseek_v4", "revision": "dsv4-global-v1",
                "physical_contract": {"non_homogeneous_layers": True,
                                      "singleton_local_shards": False},
                "pieces": [asdict(piece) for piece in self.pieces]}


def _expected_shapes(rule_id, config, ratio):
    h, d = int(config["hidden_size"]), int(config["head_dim"])
    heads, q = int(config["num_attention_heads"]), int(config["q_lora_rank"])
    groups, out = int(config["o_groups"]), int(config["o_lora_rank"])
    hc, experts = int(config["hc_mult"]), int(config["n_routed_experts"])
    ffn = int(config["moe_intermediate_size"])
    shared = ffn * int(config.get("n_shared_experts", 1))
    table = {
        "embedding": [(int(config["vocab_size"]), h)], "output": [(int(config["vocab_size"]), h)],
        "final_norm": [(h,)], "attn_norm": [(h,)], "ffn_norm": [(h,)],
        "q_down": [(q, h)], "q_up": [(heads * d, q)], "q_norm": [(q,)],
        "kv": [(d, h)], "kv_norm": [(d,)], "o_group": [(groups * out, heads * d // groups)],
        "o_proj": [(h, groups * out)], "sink": [(heads,)],
        "router": [(experts, h)], "router_bias": [(experts,)],
        "hash_router": [(int(config["vocab_size"]), int(config["num_experts_per_tok"]))],
        "expert_fc1": [(ffn, h), (ffn, h)], "expert_fc2": [(h, ffn)],
        "shared_fc1": [(shared, h), (shared, h)], "shared_fc2": [(h, shared)],
        "head_fn": [(hc, hc * h)], "head_base": [(hc,)], "head_scale": [(1,)],
        "mtp_enorm": [(h,)], "mtp_hnorm": [(h,)], "mtp_norm": [(h,)],
        "mtp_e_proj": [(h, h)], "mtp_h_proj": [(h, h)],
    }
    for prefix in ("hc_attn", "hc_ffn"):
        table.update({prefix + "_fn": [(hc * hc + 2 * hc, hc * h)],
                      prefix + "_base": [(hc * hc + 2 * hc,)], prefix + "_scale": [(3,)]})
    for suffix in ("fn", "base", "scale"):
        table["mtp_head_" + suffix] = table["head_" + suffix]
    for prefix, dim in (("compress", d), ("index_compress", int(config["index_head_dim"]))):
        width = (2 if ratio == 4 else 1) * dim
        table.update({prefix + "_kv": [(width, h)], prefix + "_gate": [(width, h)],
                      prefix + "_ape": [(ratio, width)], prefix + "_norm": [(dim,)]})
    table["index_q"] = [(int(config["index_n_heads"]) * int(config["index_head_dim"]), q)]
    table["index_score"] = [(int(config["index_n_heads"]), h)]
    return table.get(rule_id)


def rule_contexts(config, ratios):
    contexts = [("root", "", "", "", 0, False)]
    for i in range(int(config["num_hidden_layers"])):
        prefix = f"decoder.layers.{i}."
        contexts.append(("layer", f"layers.{i}.", prefix, prefix,
                         ratios[i], i < num_hash_layers(config)))
    for i in range(int(config.get("num_nextn_predict_layers", 0) or 0)):
        prefix = f"mtp.layers.{i}."
        contexts.extend([
            ("layer", f"mtp.{i}.", prefix + "mtp_model_layer.",
             prefix + "transformer_layer.", 0, False),
            ("mtp", f"mtp.{i}.", prefix, prefix, 0, False),
        ])
    return contexts


def active_rules(config, ratios, rules):
    for context in rule_contexts(config, ratios):
        scope, _, _, _, ratio, hash_router = context
        for rule in rules:
            if rule["scope"] != scope:
                continue
            when = rule.get("when")
            if (when == "compressor" and ratio == 0 or when == "indexer" and ratio != 4
                    or when == "hash" and not hash_router or when == "non_hash" and hash_router):
                continue
            yield context, rule


def compile_profile(source, config, ratios, override=None, *, rules=None):
    """Bind logical shapes to fixed rules; reject missing/unknown weights."""
    shapes = {name: tuple(spec.shape) for name, spec in source.specs.items()}
    if rules is not None and override is not None:
        raise MappingError("Frozen mapping rules cannot be combined with an override")
    rules = load_rules(override) if rules is None else load_rules(frozen=rules)
    pieces, consumed = [], set()
    experts = int(config["n_routed_experts"])
    for context, rule in active_rules(config, ratios, rules):
        scope, hf_prefix, runtime_prefix, stored_prefix, ratio, hash_router = context
        op = rule["op"]
        for expert in range(experts if op.startswith("expert") else 1):
            names = [canonical_hf_name(hf_prefix + key.format(expert=expert)) for key in rule["hf"]]
            if rule.get("optional") and not any(name in shapes for name in names):
                continue
            if any(name not in shapes for name in names):
                raise MappingError(f"Missing DSv4 tensors for {rule['id']}: {names}")
            if consumed.intersection(names):
                raise MappingError(f"Overlapping HF mapping rules: {names}")
            consumed.update(names)
            actual = [shapes[name] for name in names]
            expected = _expected_shapes(rule["id"], config, ratio)
            if expected is not None and actual != expected:
                raise MappingError(f"DSv4 shape mismatch for {names}: {actual} != {expected}")
            runtime = [runtime_prefix + key.format(expert=expert) for key in rule["mcore"]]
            stored = [stored_prefix + key.format(expert=expert) for key in rule["mcore"]]
            if op.startswith("expert"):
                # TEGroupedMLP and SequentialMLP share this physical key.
                stored = [key.replace(".experts.linear_", ".experts.experts.linear_")
                          .removesuffix(str(expert)) for key in stored]
            if op == "hc_scale":
                if actual != [(3,)]:
                    raise MappingError(f"Expected three mHC scale entries: {names}")
                for i, key in enumerate(stored):
                    pieces.append(Piece(names[0], runtime[i], key, (1,), (0,), (1,), i))
                continue
            if op in ("swiglu", "expert_swiglu") and (actual[0] != actual[1] or len(actual[0]) != 2):
                raise MappingError(f"SwiGLU gate/up shapes differ: {names}")
            global_shape = list(actual[0])
            if op in ("swiglu", "expert_swiglu"):
                global_shape[0] *= 2
            if op.startswith("expert"):
                global_shape.insert(0, experts)
            extra_key = None
            if rule.get("te"):
                extra_key = stored[0].removesuffix("weight") + "_extra_state"
                extra_key += f"/shard_{expert}_{experts}" if op.startswith("expert") else "/shard_0_1"
            for half, name in enumerate(names):
                shape = actual[half]
                offset = (half * shape[0],) + (0,) * (len(shape) - 1)
                if op.startswith("expert"):
                    shape, offset = (1,) + shape, (expert,) + offset
                pieces.append(Piece(name, runtime[0], stored[0], shape, offset,
                                    tuple(global_shape), extra_key=extra_key))
    if consumed != source.keys:
        raise MappingError(f"Unmapped HF DSv4 tensors: {sorted(source.keys - consumed)[:20]}")
    return Profile(pieces, shapes)



