# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Precision-first conversion must be readable by stock Torch DCP."""
import json
import subprocess
import sys

import pytest
import torch
from torch.distributed.checkpoint import load

from tools.checkpoint.hf_mcore_dcp.convert import main
from tools.checkpoint.hf_mcore_dcp.storage.hf import RawEncoded, SafetensorReader, ShardedSafetensorWriter


def source_fixture(root):
    hf = root / 'hf'
    hf.mkdir()
    (hf / 'config.json').write_text(json.dumps({
        'quantization_config': {'quant_method': 'fp8', 'fmt': 'e4m3',
                                'weight_block_size': [128, 128], 'scale_fmt': 'ue8m0'},
        'expert_dtype': 'fp4',
    }))
    writer = ShardedSafetensorWriter(hf, '64KB')
    writer.add('attn.weight', RawEncoded(torch.full((128,128), 56, dtype=torch.uint8), 'F8_E4M3'))
    writer.add('attn.scale', torch.tensor([[1.0001]], dtype=torch.float32))
    for name, byte in [('gate.weight', 0x21), ('up.weight', 0x43)]:
        writer.add(name, RawEncoded(torch.full((128,64), byte, dtype=torch.uint8), 'I8'))
        writer.add(name.removesuffix('.weight') + '.scale',
                   RawEncoded(torch.full((128,4), 126, dtype=torch.uint8), 'F8_E8M0'))
    writer.add('dense.weight', torch.full((8,8), 1.5, dtype=torch.bfloat16))
    writer.add('norm.weight', torch.tensor([1.0001], dtype=torch.float32))
    writer.add('ids', torch.tensor([0, 2**40], dtype=torch.int64))
    writer.finish()
    table = root / 'mapping.json'
    rules = [{'id':n, 'hf':[n], 'mcore':['model.'+n], 'op':'identity'}
             for n in ['attn.weight','dense.weight','norm.weight','ids']]
    rules.append({'id':'fc1','hf':['gate.weight','up.weight'], 'mcore':['model.fc1.weight'],
                  'op':'swiglu','extra_state':['model.fc1._extra_state/shard_0_1']})
    table.write_text(json.dumps({'schema_version':2, 'rules':rules,
                                'quantization':{'mxfp4':['gate.weight','up.weight']}}))
    return hf,table


def expected():
    # Independent E2M1 values: 0x21 -> (0.5,1), 0x43 -> (1.5,2), scale=0.5.
    gate = torch.tensor([0.25,0.5]).repeat(128,64)
    up = torch.tensor([0.75,1.0]).repeat(128,64)
    return {'model.attn.weight':torch.full((128,128),1.0001),
            'model.fc1.weight':torch.cat([gate,up]),
            'model.dense.weight':torch.full((8,8),1.5),
            'model.norm.weight':torch.tensor([1.0001]),
            'model.ids':torch.tensor([0,2**40],dtype=torch.int64)}


@pytest.mark.parametrize('dtype',[None,'auto','bf16'])
def test_quantized_import_stock_dcp_and_hf_roundtrip(tmp_path,dtype):
    hf,table=source_fixture(tmp_path)
    mc,out=tmp_path/'mc',tmp_path/'out'
    args=['hf-to-mcore','--mapping',str(table),'--source',str(hf),'--output',str(mc)]
    if dtype: args += ['--dtype',dtype]
    assert main(args)==0
    reference=expected()
    if dtype=='auto': reference['model.dense.weight']=reference['model.dense.weight'].bfloat16()
    if dtype=='bf16':
        for name in ['model.attn.weight','model.fc1.weight','model.dense.weight']:
            reference[name]=reference[name].bfloat16()
    target={k:torch.empty_like(v) for k,v in reference.items()}
    load(target,checkpoint_id=mc/'iter_0000000',no_dist=True)
    for name,value in reference.items(): assert torch.equal(target[name],value),name
    common=torch.load(mc/'iter_0000000/common.pt',weights_only=False)
    assert 'preserved_quantization' not in common['content_metadata']
    assert common['args'].fp8_param_gather is False
    assert common['args'].params_dtype== (torch.bfloat16 if dtype=='bf16' else torch.float32)
    assert main(['mcore-to-hf','--source',str(mc),'--hf-reference',str(hf),'--output',str(out)])==0
    reader=SafetensorReader(out)
    assert reader.keys=={'attn.weight','gate.weight','up.weight','dense.weight','norm.weight','ids'}
    assert torch.equal(reader.get_tensor('attn.weight'), reference['model.attn.weight'])
    assert torch.equal(reader.get_tensor('gate.weight'), reference['model.fc1.weight'][:128])
    assert torch.equal(reader.get_tensor('up.weight'), reference['model.fc1.weight'][128:])
    assert not json.loads((out/'config.json').read_text()).get('quantization_config')
    if dtype!='bf16':
        assert not torch.equal(target['model.attn.weight'],target['model.attn.weight'].bfloat16().float())


def test_no_training_or_te_imports(tmp_path):
    hf,table=source_fixture(tmp_path)
    code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('flagscale.train','megatron','transformer_engine')):
            raise AssertionError('Unexpected training dependency: '+fullname)
sys.meta_path.insert(0,Block())
from tools.checkpoint.hf_mcore_dcp.convert import main
raise SystemExit(main(sys.argv[1:]))
'''
    result=subprocess.run([sys.executable,'-c',code,'hf-to-mcore','--mapping',str(table),
                           '--source',str(hf),'--output',str(tmp_path/'mc')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_removed_preserve_mode_fails_before_writing(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['hf-to-mcore','--conversion-mode','preserve','--output',str(tmp_path/'out')])
    assert error.value.code==2
    assert not (tmp_path/'out').exists()
