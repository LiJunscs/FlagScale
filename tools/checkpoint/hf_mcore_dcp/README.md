# HF ↔ MCore DCP Conversion

Convert model weights between HF safetensors and MCore DCP. Run from the repository root in a single process (world size 1, rank 0).

Use `tools/checkpoint/deepseek_v4/mapping.yaml` for DSv4 or `tools/checkpoint/qwen3/mapping.yaml` for Qwen3. Shared mappings are inherited automatically.

## HF → MCore

The source needs `config.json` and safetensors weights, with an index for sharded weights.

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /models/dsv4-hf \
  --output /outputs/dsv4-mcore \
  --conversion-mode dense --dtype bf16
```

| Mode | Output |
| --- | --- |
| `dense` (default) | Ordinary tensors; quantized inputs are decoded. |
| `preserve` | Original FP8/MXFP4 payloads and scales; requires `--dtype auto` and a supported whole-matrix layout. |

`preserve` requires the FlagScale `quantized_checkpoint` and `checkpoint_serialization` adapters in the conversion/loading environment. Use `dense` for layouts that require slicing or permuting quantized matrices.

## MCore → HF

The source can be a checkpoint root or a completed iteration directory.

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert mcore-to-hf \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /checkpoints/dsv4/iter_0000100 \
  --hf-reference /models/dsv4-hf \
  --output /outputs/dsv4-export
```

`--hf-reference` is optional: it supplies config, naming metadata and tokenizer assets, without reading reference weight values. No tokenizer is copied when omitted.

If the source retains the converter's `hf_export.json`, `--mapping` can be omitted. Training saves may not retain this file. For DSv4, supply `--vocab-size N` if saved args lack the unpadded vocabulary size.

Export follows actual source storage by default. To decode a preserve checkpoint into FP32, add `--conversion-mode dense --dtype fp32`.

## Options

| Option | Usage |
| --- | --- |
| `--dry-run` | Print the metadata-based plan without writing weights. Value checks run during conversion. |
| `--dtype` | `auto` (default), `bf16`, `fp16` or `fp32`. Auto preserves dense dtypes and decodes quantized imports to BF16; explicit casts retain existing FP32 and integer tensors. |
| `--device` | Decode on `cpu` (default) or `cuda[:N]`. |
| `--max-shard-size 4GB` | Target output shard size; individual regions may exceed it. |
| `--overwrite` | Replace an existing output only after conversion succeeds. Input/output directories must not overlap. |
| `--mapping-override custom.yaml` | Apply YAML/JSON overrides; repeat to apply multiple files in order. |

Overrides replace complete rules by `id`; new IDs add rules, and `{id: name, disabled: true}` removes a rule. Unmapped source tensors require explicit `ignore` entries. With the shared mappings, use normalized HF names in rules and ignore patterns.

To export a resolved mapping from a file or an importable module:

```bash
python -m tools.checkpoint.hf_mcore_dcp.convert export-mapping \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --output /tmp/dsv4-mapping.json
```

`--mapping python:package.module:MAPPING` imports a mapping dictionary from the environment. Conversion executes declared layouts; it does not verify model equivalence.
