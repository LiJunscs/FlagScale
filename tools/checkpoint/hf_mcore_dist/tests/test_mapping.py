# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader, load

from tools.checkpoint.hf_mcore_dist import MappingError
from tools.checkpoint.hf_mcore_dist.config import resolve, translate_config
from tools.checkpoint.hf_mcore_dist.convert import main
from tools.checkpoint.hf_mcore_dist.engine import compile_mapping
from tools.checkpoint.hf_mcore_dist.hf_io import RawEncoded, SafetensorReader, ShardedSafetensorWriter
from tools.checkpoint.hf_mcore_dist.mapping import load_mapping
from tools.checkpoint.hf_mcore_dist.operators import transform
from tools.checkpoint.hf_mcore_dist.precision import TensorSpec

ROOT = Path(__file__).resolve().parents[4]


def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def make_hf(path, tensors, config=None):
    path.mkdir()
    write_json(path / "config.json", config or {})
    writer = ShardedSafetensorWriter(path, "2KB")
    for name, tensor in tensors.items():
        writer.add(name, tensor)
    writer.finish()


def test_overrides_extensions_disabling_and_external_table(tmp_path, monkeypatch):
    base = {"schema_version": 1, "rules": [
        {"id": "a", "hf": ["a"], "mcore": ["x"]},
        {"id": "b", "hf": ["b"], "mcore": ["y"]}]}
    write_json(tmp_path / "base.json", base)
    patch = {"schema_version": 1, "rules": [
        {"id": "a", "hf": ["a"], "mcore": ["renamed"]},
        {"id": "b", "disabled": True}, {"id": "c", "hf": ["c"], "mcore": ["z"]}]}
    write_json(tmp_path / "patch.json", patch)
    merged = load_mapping(tmp_path / "base.json", [str(tmp_path / "patch.json")])
    assert [r["id"] for r in merged["rules"]] == ["a", "c"]
    assert merged["rules"][0]["mcore"] == ["renamed"]
    write_json(tmp_path / "child.json", {**patch, "extends": "base.json"})
    assert load_mapping(tmp_path / "child.json") == merged
    module = types.ModuleType("local_conversion_table")
    module.MAPPING = base
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert load_mapping("python:local_conversion_table:MAPPING")["rules"] == base["rules"]
    assert main(["export-mapping", "--mapping", "python:local_conversion_table:MAPPING",
                 "--output", str(tmp_path / "export.json")]) == 0
    assert json.loads((tmp_path / "export.json").read_text())["rules"] == base["rules"]
    write_json(tmp_path / "cycle.json", {"schema_version": 1, "extends": "cycle.json"})
    with pytest.raises(MappingError, match="Cyclic"):
        load_mapping(tmp_path / "cycle.json")
    write_json(tmp_path / "duplicate.json", {**base, "rules": [base["rules"][0]] * 2})
    with pytest.raises(MappingError, match="Duplicate"):
        load_mapping(tmp_path / "duplicate.json")


def test_config_aliases_defaults_nested_and_arithmetic():
    mapping = {"config_map": {
        "experts": {"mcore": ["num_experts", "num_moe_experts"], "required": True},
        "rope.factor": "rotary_scaling_factor",
        "optional": {"mcore": "optional", "hf_default": 4}},
        "mcore_args": {"ffn": {"op": "mul", "args": ["$hf.experts", 16]}}}
    result = translate_config(mapping, {"experts": 8, "rope": {"factor": 2}}, True)
    assert result == {"num_experts": 8, "rotary_scaling_factor": 2, "optional": 4, "ffn": 128}
    assert translate_config(mapping, {"num_moe_experts": 8, "rotary_scaling_factor": 2}, False) == {
        "experts": 8, "rope": {"factor": 2}}
    with pytest.raises(MappingError, match="Conflicting"):
        translate_config(mapping, {"num_experts": 4, "num_moe_experts": 8}, False)
    with pytest.raises(MappingError, match="Unknown config"):
        resolve({"op": "eval", "args": ["print(1)"]}, {})


@pytest.mark.parametrize("op,shapes,params", [
    ("identity", [(3, 4)], {}),
    ("concat", [(2, 3), (2, 5)], {"axis": 1, "sizes": [3, 5]}),
    ("split", [(2, 8)], {"axis": 1, "sizes": [3, 5]}),
    ("swiglu", [(3, 4), (3, 4)], {}),
    ("permute", [(2, 3, 4)], {"dims": [2, 0, 1]}),
    ("reshape", [(2, 3, 4)], {"hf_shape": [2, 3, 4], "mcore_shape": [6, 4]}),
    ("stack", [(3, 4)] * 3, {"axis": 1}),
    ("expert_swiglu", [(3, 4)] * 6, {}),
    ("qkv", [(8, 5), (4, 5), (4, 5)], {"heads": 4, "groups": 2}),
    ("qkv", [(8,), (4,), (4,)], {"heads": 4, "groups": 2, "order": ["v", "q", "k"]}),
])
def test_reversible_operators(op, shapes, params):
    inputs = [torch.arange(torch.Size(s).numel()).float().reshape(s) + i * 100 for i, s in enumerate(shapes)]
    outputs = transform(inputs, op, params)
    restored = transform(outputs, op, params, False)
    assert len(inputs) == len(restored)
    for expected, actual in zip(inputs, restored):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_shape_compatible_nonstandard_order_is_executed():
    q, k, v = [torch.full((4, 3), x) for x in (1., 2., 3.)]
    result = transform([q, k, v], "qkv", {"heads": 2, "groups": 2, "order": ["v", "q", "k"]})[0]
    assert result[:, 0].tolist() == [3, 3, 1, 1, 2, 2] * 2
    gate, up = torch.ones(3, 4), torch.full((3, 4), 2.)
    assert torch.equal(transform([up, gate], "swiglu", {})[0], torch.cat([up, gate]))


def test_missing_sizes_and_unmapped_inputs_are_rejected():
    with pytest.raises(MappingError, match="sizes"):
        transform([torch.empty(6, 4)], "concat", {}, False)
    mapping = {"rules": [{"id": "a", "hf": ["a"], "mcore": ["x"]}]}
    specs = {name: TensorSpec(name, (3, 4), "F32") for name in ["a", "b"]}
    with pytest.raises(MappingError, match="Unmapped"):
        compile_mapping(mapping, specs, {"hf": {}, "mcore": {}}, forward=True)
    mapping["ignore"] = {"hf": ["b"]}
    assert compile_mapping(mapping, specs, {}, forward=True)["ignored"] == ["b"]


def test_collected_experts_expand_from_config_in_both_directions():
    mapping = {"rules": [{"id": "experts", "hf": ["layers.{layer}.experts.{expert}.gate", "layers.{layer}.experts.{expert}.up"],
                            "mcore": ["layers.{layer}.fc1"], "collect": {"expert": "$hf.experts"}, "op": "expert_swiglu"}]}
    specs = {f"layers.{layer}.experts.{expert}.{half}": TensorSpec("", (3, 4), "F32")
             for layer in (0, 2) for expert in range(2) for half in ("gate", "up")}
    forward = compile_mapping(mapping, specs, {"hf": {"experts": 2}}, forward=True)
    assert forward["outputs"]["layers.2.fc1"].shape == (2, 6, 4)
    reverse = compile_mapping(mapping, forward["outputs"], {"hf": {"experts": 2}}, forward=False)
    assert reverse["outputs"].keys() == specs.keys()


def test_dense_qwen3_cli_roundtrip_and_standard_dcp(tmp_path):
    config = dict(model_type="qwen3", num_hidden_layers=2, hidden_size=8, intermediate_size=12,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=2, vocab_size=16)
    tensors = {"model.embed_tokens.weight": torch.arange(128).reshape(16, 8).to(torch.bfloat16),
               "lm_head.weight": torch.arange(128).reshape(16, 8).to(torch.bfloat16),
               "model.norm.weight": torch.ones(8)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        shapes = {"input_layernorm.weight": (8,), "post_attention_layernorm.weight": (8,),
                  "self_attn.q_proj.weight": (8, 8), "self_attn.k_proj.weight": (4, 8),
                  "self_attn.v_proj.weight": (4, 8), "self_attn.o_proj.weight": (8, 8),
                  "self_attn.q_norm.weight": (2,), "self_attn.k_norm.weight": (2,),
                  "mlp.gate_proj.weight": (12, 8), "mlp.up_proj.weight": (12, 8), "mlp.down_proj.weight": (8, 12)}
        for i, (name, shape) in enumerate(shapes.items()):
            tensors[prefix + name] = (torch.arange(torch.Size(shape).numel()).reshape(shape) + i + layer).to(torch.bfloat16)
    hf, mc, exported = tmp_path / "hf", tmp_path / "mc", tmp_path / "export"
    make_hf(hf, tensors, config)
    cmd = ["hf-to-mcore", "--source", str(hf), "--output", str(mc), "--mapping",
           str(ROOT / "tools/checkpoint/qwen3/mapping.yaml"), "--max-shard-size", "1KB"]
    assert main([*cmd, "--dry-run"]) == 0
    assert not mc.exists()
    assert main(cmd) == 0
    metadata = FileSystemReader(mc / "iter_0000000").read_metadata()
    key = "decoder.layers.1.mlp.linear_fc1.weight"
    target = {key: torch.empty(metadata.state_dict_metadata[key].size, dtype=torch.bfloat16)}
    load(target, checkpoint_id=mc / "iter_0000000")
    assert torch.equal(target[key], torch.cat([tensors["model.layers.1.mlp.gate_proj.weight"], tensors["model.layers.1.mlp.up_proj.weight"]]))
    # No HF model/index/reference is needed for reverse conversion.
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(exported)]) == 0
    reader = SafetensorReader(exported)
    assert reader.keys == tensors.keys()
    for key, value in tensors.items():
        assert torch.equal(reader.get_tensor(key), value)
    assert json.loads((exported / "config.json").read_text())["hidden_size"] == 8
    assert not json.loads((mc / "conversion_report.json").read_text())["model_semantics_validated"]


@pytest.mark.parametrize("kind", ["fp8", "mxfp4"])
def test_arbitrary_quantized_names_use_common_codec(tmp_path, kind):
    hf, mc, out = tmp_path / "hf", tmp_path / "mc", tmp_path / "out"
    hf.mkdir()
    config = {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0"}}
    write_json(hf / "config.json", config)
    writer = ShardedSafetensorWriter(hf)
    name = "custom.projection.weight"
    shape = (2, 64) if kind == "fp8" else (2, 32)
    code, value = ("F8_E4M3", 56) if kind == "fp8" else ("U8", 34)
    writer.add(name, RawEncoded(torch.full(shape, value, dtype=torch.uint8), code))
    writer.add("custom.projection.weight_scale_inv", RawEncoded(
        torch.full((1, 1) if kind == "fp8" else (2, 2), 127, dtype=torch.uint8), "U8"))
    writer.finish()
    mapping = {"schema_version": 1, "rules": [{"id": "w", "hf": [name], "mcore": ["custom.weight"]}],
               "quantization": {"mxfp4": ["custom.*"]} if kind == "mxfp4" else {}}
    table = write_json(tmp_path / "mapping.json", mapping)
    assert main(["hf-to-mcore", "--source", str(hf), "--output", str(mc), "--mapping", table]) == 0
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(out)]) == 0
    restored = SafetensorReader(out).get_tensor(name)
    assert restored.shape == (2, 64)
    assert torch.equal(restored, torch.ones(2, 64, dtype=torch.bfloat16))
    assert "quantization_config" not in json.loads((out / "config.json").read_text())


def test_failed_conversion_does_not_replace_output(tmp_path):
    hf, output = tmp_path / "hf", tmp_path / "output"
    make_hf(hf, {"weight": torch.full((2, 3), float("nan"))})
    output.mkdir()
    (output / "sentinel").write_text("keep")
    mapping = write_json(tmp_path / "map.json", {"schema_version": 1, "rules": [
        {"id": "w", "hf": ["weight"], "mcore": ["weight"]}]})
    assert main(["hf-to-mcore", "--source", str(hf), "--output", str(output), "--mapping", mapping, "--overwrite"]) == 2
    assert (output / "sentinel").read_text() == "keep"
    assert not list(tmp_path.glob(".output.incomplete-*"))


def test_common_archive_restores_exact_bytes_and_rejects_changed_weights(tmp_path):
    from dataclasses import replace
    from tools.checkpoint.hf_mcore_dist.quantization import PrecisionSource
    from tools.checkpoint.hf_mcore_dist.quantized_archive import QuantizedArchive, save_quantized_archive

    hf = tmp_path / "hf"
    hf.mkdir()
    writer = ShardedSafetensorWriter(hf)
    writer.add("projection.weight", RawEncoded(torch.full((2, 64), 56, dtype=torch.uint8), "F8_E4M3"))
    writer.add("projection.weight_scale_inv", torch.ones(1, 1))
    writer.finish()
    config = {"quantization_config": {"quant_method": "fp8"}}
    source = PrecisionSource(SafetensorReader(hf), config)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_quantized_archive(source, config, checkpoint, "1KB")
    target = types.SimpleNamespace(specs={key: replace(spec, dtype="BF16", kind="dense", scale=None, scale_dtype=None)
                                         for key, spec in source.specs.items()})
    archive = QuantizedArchive(checkpoint, target)
    values = source.get_tensor("projection.weight")
    weight, scale = archive.restore("projection.weight", values)
    assert torch.equal(weight.reader.get_bytes(weight.name), torch.full((2, 64), 56, dtype=torch.uint8))
    assert torch.equal(scale.reader.get_tensor(scale.name), torch.ones(1, 1))
    archive.validate_consumed()
    with pytest.raises(ValueError, match="differs"):
        QuantizedArchive(checkpoint, target).restore("projection.weight", values + 1)


def test_generic_conversion_does_not_import_training_or_transformers(tmp_path):
    import os
    import subprocess

    script = '''
import sys
class NoTraining:
    def find_spec(self, name, *args):
        if name.startswith(('transformers', 'megatron', 'flagscale.train')):
            raise AssertionError('Unexpected runtime import: ' + name)
sys.meta_path.insert(0, NoTraining())
from pathlib import Path
import torch, json
from tools.checkpoint.hf_mcore_dist.convert import main
from tools.checkpoint.hf_mcore_dist.hf_io import ShardedSafetensorWriter
root=Path(sys.argv[1]); hf=root/'hf'; hf.mkdir()
(hf/'config.json').write_text('{}')
w=ShardedSafetensorWriter(hf); w.add('w',torch.ones(2,3)); w.finish()
(root/'map.json').write_text(json.dumps({'schema_version':1,'rules':[{'id':'w','hf':['w'],'mcore':['w']}]}))
assert main(['hf-to-mcore','--source',str(hf),'--output',str(root/'mc'),'--mapping',str(root/'map.json')]) == 0
assert main(['mcore-to-hf','--source',str(root/'mc'),'--output',str(root/'out')]) == 0
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dense_scale_suffix_is_not_assumed_to_be_quantization(tmp_path):
    from tools.checkpoint.hf_mcore_dist.quantization import PrecisionSource
    hf = tmp_path / "hf"
    make_hf(hf, {"norm.scale": torch.ones(4), "projection.weight_scale_inv": torch.ones(2)})
    source = PrecisionSource(SafetensorReader(hf), {})
    assert source.keys == {"norm.scale", "projection.weight_scale_inv"}
    assert all(spec.kind == "dense" for spec in source.specs.values())


def test_common_parameter_dictionary_is_sparse_and_model_overrides_are_explicit():
    common = load_mapping(ROOT / "tools/checkpoint/hf_mcore_dist/parameters.yaml")
    assert translate_config(common, {}, True) == {}
    assert translate_config(common, {}, False) == {}
    assert translate_config(common, {"hidden_size": 128}, True) == {"hidden_size": 128}
    assert translate_config(common, {"num_moe_experts": 8}, False) == {"n_routed_experts": 8}
    assert translate_config(common, {"head_dim": 64}, True) == {"kv_channels": 64}
    ds = load_mapping(ROOT / "tools/checkpoint/deepseek_v4/mapping.yaml")
    assert translate_config({"config_map": {"head_dim": ds["config_map"]["head_dim"]}},
                            {"head_dim": 64}, True) == {"v_head_dim": 64}
    # Defaults and alternate meanings are model deltas, not activated by the common registry.
    assert "sliding_window" not in common["config_map"]
    assert ds["config_map"]["sliding_window"] == "csa_window_size"
    assert ds["config_map"]["num_hash_layers"]["hf_default"] == 0
    assert common["config_map"]["num_hash_layers"] == "moe_n_hash_layers"


def test_absent_optional_module_does_not_require_its_configuration():
    mapping = {"rules": [
        {"id": "core", "hf": ["w"], "mcore": ["w"]},
        {"id": "experts", "optional": True, "hf": ["experts.{expert}.w"],
         "mcore": ["experts"], "op": "stack", "collect": {"expert": "$hf.num_experts"}}]}
    specs = {"w": TensorSpec("w", (2, 3), "F32")}
    assert compile_mapping(mapping, specs, {"hf": {}}, forward=True)["outputs"].keys() == {"w"}
    specs["experts.0.w"] = TensorSpec("experts.0.w", (2, 3), "F32")
    with pytest.raises(MappingError, match="num_experts"):
        compile_mapping(mapping, specs, {"hf": {}}, forward=True)


def test_models_inherit_shared_weight_rules_without_duplicate_tables():
    import yaml
    common = load_mapping(ROOT / "tools/checkpoint/hf_mcore_dist/mapping.yaml")
    path = ROOT / "tools/checkpoint/deepseek_v4/mapping.yaml"
    delta = yaml.safe_load(path.read_text())
    resolved = load_mapping(path)
    base = {rule["id"]: rule for rule in common["rules"]}
    merged = {rule["id"]: rule for rule in resolved["rules"]}
    local = {rule["id"]: rule for rule in delta["rules"]}
    for name in ("q_down", "q_up", "q_norm", "router", "expert_fc1", "expert_fc2", "shared_fc1", "shared_fc2"):
        assert name not in local
        assert merged[name] == base[name]
    assert merged["attn_norm"]["mcore"] != base["attn_norm"]["mcore"]
    assert resolved["naming"]["strip_prefix"] == common["naming"]["strip_prefix"]
    assert not (path.parent / "parameters.yaml").exists()
