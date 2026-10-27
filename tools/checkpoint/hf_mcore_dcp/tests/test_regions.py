# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import json
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint import FileSystemReader, load

from tools.checkpoint.hf_mcore_dcp.convert import main
from tools.checkpoint.hf_mcore_dcp.hf_io import RawEncoded, SafetensorReader, ShardedSafetensorWriter
from tools.checkpoint.hf_mcore_dcp.operators import transform

ROOT = Path(__file__).resolve().parents[4]


def save_hf(path, tensors, config=None):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config or {}))
    writer = ShardedSafetensorWriter(path, "32KB")
    for name, value in tensors.items():
        writer.add(name, value)
    writer.finish()


def convert_pair(tmp_path, tensors, mapping, mode="dense", config=None):
    hf, mc, out = (tmp_path / key for key in ("hf", "mc", "out"))
    save_hf(hf, tensors, config)
    table = tmp_path / "mapping.json"
    table.write_text(json.dumps({"schema_version": 1, **mapping}))
    assert main(["hf-to-mcore", "--source", str(hf), "--output", str(mc), "--mapping", str(table),
                 "--conversion-mode", mode, "--max-shard-size", "2KB"]) == 0
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(out)]) == 0
    return hf, mc, out


@pytest.mark.parametrize("op,shapes,params,outputs", [
    ("concat", [(2, 3), (2, 5)], {"axis": 1, "sizes": [3, 5]}, 1),
    ("split", [(2, 8)], {"axis": 1, "sizes": [3, 5]}, 2),
    ("stack", [(2, 3)] * 3, {"axis": -1}, 1),
    ("permute", [(2, 3, 4)], {"dims": [2, 0, 1]}, 1),
    ("reshape", [(2, 3, 4)], {"hf_shape": [2, 3, 4], "mcore_shape": [6, 4]}, 1),
    ("qkv", [(8, 3), (4, 3), (4, 3)], {"heads": 4, "groups": 2, "order": ["v", "q", "k"]}, 1),
    ("expert_swiglu", [(2, 3)] * 4, {}, 1),
])
def test_region_roundtrip_matches_declared_operator(tmp_path, op, shapes, params, outputs):
    tensors = {f"w{i}": torch.arange(torch.Size(shape).numel()).reshape(shape).float() + i * 100
               for i, shape in enumerate(shapes)}
    targets = [f"target{i}" for i in range(outputs)]
    mapping = {"rules": [{"id": "rule", "hf": list(tensors), "mcore": targets, "op": op, "params": params}]}
    _, mc, out = convert_pair(tmp_path, tensors, mapping)
    reader = SafetensorReader(out)
    for key, value in tensors.items():
        torch.testing.assert_close(reader.get_tensor(key), value, atol=0, rtol=0)
    expected = transform(list(tensors.values()), op, params)
    target = {key: torch.empty_like(value) for key, value in zip(targets, expected)}
    load(target, checkpoint_id=mc / "iter_0000000")
    for key, value in zip(targets, expected):
        torch.testing.assert_close(target[key], value, atol=0, rtol=0)


def quantized_tensors():
    return {"gate.weight": RawEncoded(torch.full((2, 64), 56, dtype=torch.uint8), "F8_E4M3"),
            "gate.weight_scale_inv": torch.tensor([[0.3]]),
            "up.weight": RawEncoded(torch.full((2, 32), 0x21, dtype=torch.uint8), "U8"),
            "up.scale": RawEncoded(torch.full((2, 2), 127, dtype=torch.uint8), "F8_E8M0")}


def test_preserve_raw_roundtrip_dense_export_and_user_swap(tmp_path):
    mapping = {"quantization": {"mxfp4": ["up.weight"]},
               "rules": [{"id": "fc1", "hf": ["gate.weight", "up.weight"], "mcore": ["fc1"], "op": "swiglu"}]}
    config = {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0"}}
    hf, mc, out = convert_pair(tmp_path, quantized_tensors(), mapping, "preserve", config)
    before, after = SafetensorReader(hf), SafetensorReader(out)
    assert before.keys == after.keys
    for key in before.keys:
        assert before.tensor_storage(key)[:2] == after.tensor_storage(key)[:2]
        assert b''.join(before.iter_tensor_bytes(key)) == b''.join(after.iter_tensor_bytes(key))
    dense = tmp_path / "dense"
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(dense),
                 "--conversion-mode", "dense", "--dtype", "fp32"]) == 0
    reader = SafetensorReader(dense)
    torch.testing.assert_close(reader.get_tensor("gate.weight"), torch.full((2, 64), .3), rtol=0, atol=0)
    assert reader.get_tensor("up.weight")[0, :4].tolist() == [.5, 1., .5, 1.]
    swapped = tmp_path / "swap.json"
    swapped.write_text(json.dumps({"schema_version": 1, "rules": [
        {"id": "fc1", "hf": ["up.weight", "gate.weight"], "mcore": ["fc1"], "op": "swiglu"}]}))
    target = tmp_path / "swapped"
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(target),
                 "--mapping-override", str(swapped)]) == 0
    reader = SafetensorReader(target)
    assert reader.tensor_dtypes()["gate.weight"] == "U8"
    assert reader.tensor_dtypes()["up.weight"] == "F8_E4M3"
    assert reader.get_bytes("gate.weight")[0, 0] == 0x21
    assert reader.get_bytes("up.weight")[0, 0] == 56


def test_preserve_mixed_dense_fusion(tmp_path):
    tensors = {"a": torch.full((3, 4), .1234567), "b": torch.ones(3, 4, dtype=torch.float16)}
    _, _, out = convert_pair(tmp_path, tensors, {"rules": [
        {"id": "fc1", "hf": ["a", "b"], "mcore": ["fc1"], "op": "swiglu"}]}, "preserve")
    reader = SafetensorReader(out)
    for key, value in tensors.items():
        assert reader.get_tensor(key).dtype == value.dtype
        torch.testing.assert_close(reader.get_tensor(key), value, atol=0, rtol=0)


def test_preserve_rejects_quantized_permutation_before_writing(tmp_path):
    hf = tmp_path / "hf"
    save_hf(hf, quantized_tensors(), {"quantization_config": {"quant_method": "fp8", "scale_fmt": "ue8m0"}})
    table = tmp_path / "map.json"
    table.write_text(json.dumps({"schema_version": 1, "quantization": {"mxfp4": ["up.weight"]}, "rules": [
        {"id": "a", "hf": ["gate.weight"], "mcore": ["a"], "op": "permute", "params": {"dims": [1, 0]}},
        {"id": "b", "hf": ["up.weight"], "mcore": ["b"]}]}))
    assert main(["hf-to-mcore", "--source", str(hf), "--output", str(tmp_path / "mc"),
                 "--mapping", str(table), "--conversion-mode", "preserve", "--dry-run"]) == 2
    assert not (tmp_path / "mc").exists()


def test_export_slice_is_explicit_and_uses_source_storage(tmp_path):
    _, mc, _ = convert_pair(tmp_path, {"embedding": torch.arange(24).reshape(8, 3).float()}, {"rules": [
        {"id": "embedding", "hf": ["embedding"], "mcore": ["embedding"]}]})
    override = tmp_path / "crop.json"
    override.write_text(json.dumps({"schema_version": 1, "rules": [
        {"id": "embedding", "hf": ["embedding"], "mcore": ["embedding"], "export_slice": {"axis": 0, "size": 5}}]}))
    target = tmp_path / "cropped"
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(target),
                 "--mapping-override", str(override)]) == 0
    torch.testing.assert_close(SafetensorReader(target).get_tensor("embedding"), torch.arange(15).reshape(5, 3).float())


def test_generated_tensors_and_zero_omission_are_explicit(tmp_path):
    _, mc, _ = convert_pair(tmp_path, {"w": torch.ones(2, 3), "bias": torch.zeros(2)}, {
        "zero_checks": {"hf": ["bias"]}, "rules": [{"id": "w", "hf": ["w"], "mcore": ["w"]}]})
    override = tmp_path / "generate.json"
    override.write_text(json.dumps({"schema_version": 1, "rules": [
        {"id": "bias", "hf": ["bias"], "mcore": [], "generate": {"side": "hf", "shape": [2], "dtype": "F32", "value": 0}}]}))
    target = tmp_path / "generated"
    assert main(["mcore-to-hf", "--source", str(mc), "--output", str(target), "--mapping-override", str(override)]) == 0
    assert torch.equal(SafetensorReader(target).get_tensor("bias"), torch.zeros(2))
    meta = FileSystemReader(mc / "iter_0000000").read_metadata()
    assert "bias" not in meta.state_dict_metadata


def test_qkv_source_matrices_are_read_once_per_target(tmp_path, monkeypatch):
    from collections import Counter
    from tools.checkpoint.hf_mcore_dcp.quantization import PrecisionSource
    reads = Counter()
    original = PrecisionSource.get_tensor
    def counted(self, name):
        reads[name] += 1
        return original(self, name)
    monkeypatch.setattr(PrecisionSource, "get_tensor", counted)
    convert_pair(tmp_path, {"q": torch.ones(8, 3), "k": torch.ones(4, 3), "v": torch.ones(4, 3)}, {
        "rules": [{"id": "qkv", "hf": ["q", "k", "v"], "mcore": ["qkv"],
                   "op": "qkv", "params": {"heads": 4, "groups": 2}}]})
    assert reads == {"q": 1, "k": 1, "v": 1}


def test_nonzero_omission_fails_and_preserves_existing_output(tmp_path):
    hf, output = tmp_path / "hf", tmp_path / "mc"
    save_hf(hf, {"w": torch.ones(2,3), "bias": torch.ones(2)})
    output.mkdir()
    (output / "sentinel").write_text("keep")
    table = tmp_path / "map.json"
    table.write_text(json.dumps({"schema_version": 1, "zero_checks": {"hf": ["bias"]},
                                 "rules": [{"id": "w", "hf": ["w"], "mcore": ["w"]}]}))
    args = ["hf-to-mcore", "--source", str(hf), "--output", str(output), "--mapping", str(table), "--overwrite"]
    assert main([*args, "--dry-run"]) == 0
    assert main(args) == 2
    assert (output / "sentinel").read_text() == "keep"
