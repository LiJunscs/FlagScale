# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""DSv4 exercises exactly the same entrypoint/regions as other mapping tables."""
import json
from pathlib import Path

import pytest
import torch

from tools.checkpoint.hf_mcore_dcp.convert import main
from tools.checkpoint.hf_mcore_dcp.hf_io import RawEncoded, SafetensorReader, ShardedSafetensorWriter
from tools.checkpoint.hf_mcore_dcp.mapping import load_mapping

ROOT = Path(__file__).resolve().parents[4]
TABLE = ROOT / "tools/checkpoint/deepseek_v4/mapping.yaml"


def tiny_source(path, modern_names=False):
    config = dict(num_hidden_layers=1, hidden_size=64, vocab_size=64, num_attention_heads=2,
                  head_dim=32, qk_rope_head_dim=16, q_lora_rank=64, o_groups=1, o_lora_rank=32,
                  n_routed_experts=2, num_experts_per_tok=1, moe_intermediate_size=64,
                  n_shared_experts=1, hc_mult=2, compress_ratios=[0], num_hash_layers=0,
                  num_nextn_predict_layers=1, moe_router_enable_expert_bias=False,
                  expert_dtype="fp4", quantization_config={"quant_method": "fp8", "scale_fmt": "ue8m0"},
                  rope_scaling={"type": "yarn", "factor": 1., "original_max_position_embeddings": 128,
                                "beta_fast": 32., "beta_slow": 1.})
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    table = load_mapping(TABLE)
    # Fixture dimensions are independent of the runtime converter: it has no shape table.
    shapes = dict(embedding=(64,64), output=(64,64), final_norm=(64,), head_fn=(2,128),
                  head_base=(2,), head_scale=(1,), attn_norm=(64,), ffn_norm=(64,),
                  q_down=(64,64), q_norm=(64,), q_up=(64,64), kv=(32,64), kv_norm=(32,),
                  o_group=(32,64), o_proj=(64,32), sink=(2,), router=(2,64), router_bias=(2,),
                  expert_fc1=(64,64), expert_fc2=(64,64), shared_fc1=(64,64), shared_fc2=(64,64),
                  hc_attn_fn=(8,128), hc_attn_base=(8,), hc_attn_scale=(3,),
                  hc_ffn_fn=(8,128), hc_ffn_base=(8,), hc_ffn_scale=(3,),
                  mtp_enorm=(64,), mtp_hnorm=(64,), mtp_norm=(64,), mtp_e_proj=(64,64),
                  mtp_h_proj=(64,64), mtp_head_fn=(2,128), mtp_head_base=(2,), mtp_head_scale=(1,))
    writer = ShardedSafetensorWriter(path, "32KB")
    aliases = {v:k for k,v in table["naming"]["segments"].items()}
    for row in table["rules"]:
        if row["id"] not in shapes:
            continue
        scopes = table["scopes"].get(row.get("scope"), [{"hf": ""}])
        for scope in scopes:
            for expert in range(2 if row.get("collect") else 1):
                for template in row["hf"]:
                    name = (scope["hf"] + template).format(layer=0, expert=expert)
                    if modern_names:
                        name = "model." + ".".join(aliases.get(p,p) for p in name.split("."))
                    shape = shapes[row["id"]]
                    if row.get("collect"):
                        writer.add(name, RawEncoded(torch.full((64,32),0x21,dtype=torch.uint8),"U8"))
                        writer.add(name.removesuffix(".weight")+".scale", RawEncoded(torch.full((64,2),127,dtype=torch.uint8),"F8_E8M0"))
                    elif row["id"] == "q_down":
                        writer.add(name,RawEncoded(torch.full(shape,56,dtype=torch.uint8),"F8_E4M3"))
                        writer.add(name.removesuffix(".weight")+".weight_scale_inv",torch.tensor([[.3]]))
                    else:
                        tensor = (torch.arange(torch.Size(shape).numel()).reshape(shape) % 7).float()
                        if row["id"] == "router_bias":
                            tensor.zero_()
                        writer.add(name,tensor if len(shape)==1 else tensor.bfloat16())
    writer.finish()


@pytest.mark.parametrize("mode",["dense","preserve"])
@pytest.mark.parametrize("modern_names",[False,True])
def test_complete_dsv4_uses_public_pipeline(tmp_path, mode, modern_names):
    hf,mc,out=(tmp_path/k for k in ("hf","mc","out"))
    tiny_source(hf,modern_names)
    assert main(["hf-to-mcore","--mapping",str(TABLE),"--source",str(hf),"--output",str(mc),
                 "--conversion-mode",mode])==0
    assert main(["mcore-to-hf","--source",str(mc),"--output",str(out)])==0
    before,after=SafetensorReader(hf),SafetensorReader(out)
    if mode=="preserve":
        assert before.keys==after.keys
        for name in before.keys:
            assert before.tensor_storage(name)[:2]==after.tensor_storage(name)[:2]
            assert b''.join(before.iter_tensor_bytes(name))==b''.join(after.iter_tensor_bytes(name))
    else:
        assert not any(n.endswith((".scale",".weight_scale_inv")) for n in after.keys)
    config=json.loads((out/"config.json").read_text())
    assert config["num_nextn_predict_layers"]==1
    assert config["compress_ratios"]==[0,0]
    assert config["moe_router_enable_expert_bias"] is False
    assert config["rope_scaling"]["mscale"]==1.0
    assert not json.loads((mc/"conversion_report.json").read_text())["model_semantics_validated"]


def test_dsv4_does_not_reject_executable_nonstandard_layout(tmp_path):
    hf,mc=tmp_path/"hf",tmp_path/"mc"
    tiny_source(hf)
    override=tmp_path/"override.json"
    table=load_mapping(TABLE)
    rule=next(row for row in table["rules"] if row["id"]=="shared_fc1")
    rule["hf"].reverse()
    override.write_text(json.dumps({"schema_version":1,"rules":[rule]}))
    config=json.loads((hf/"config.json").read_text())
    # Former profile semantics rejected these flags without regard to executable weights.
    config.update(model_type="custom", hidden_act="custom_activation", tie_word_embeddings=True)
    (hf/"config.json").write_text(json.dumps(config))
    assert main(["hf-to-mcore","--mapping",str(TABLE),"--mapping-override",str(override),
                 "--source",str(hf),"--output",str(mc)])==0


def test_unpadded_vocab_override_uses_actual_source(tmp_path):
    hf,mc,out=(tmp_path/k for k in ("hf","mc","out"))
    tiny_source(hf)
    assert main(["hf-to-mcore","--mapping",str(TABLE),"--source",str(hf),"--output",str(mc)])==0
    path=mc/"iter_0000000"/"common.pt"
    common=torch.load(path,weights_only=False)
    common["args"].vocab_size=None
    torch.save(common,path)
    args=["mcore-to-hf","--source",str(mc),"--output",str(out)]
    assert main(args)==2
    assert not out.exists()
    assert main([*args,"--vocab-size","32"])==0
    assert SafetensorReader(out).get_tensor("embed.weight").shape==(32,64)
