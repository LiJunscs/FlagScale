# HF ↔ MCore DCP 转换

`hf_mcore_dist` 存放公共转换基础设施。模型目录只增加必要的映射配置；确实无法由常见操作表达的规则保留为模型专用代码。转换器执行用户指定的布局，检查存储和操作是否可执行，不认证模型语义或前向结果等价。

## 代码组织

| 文件 | 职责 |
| --- | --- |
| `mapping.py`、`mapping.yaml` | 小型映射表、继承、覆盖、名称模板展开及常见 dense decoder 预设 |
| `operators.py`、`engine.py` | 双向布局操作、metadata 预检查及逐操作组执行 |
| `config.py` | 公共参数转换：字段别名、嵌套字段、默认值、简单派生表达式 |
| `precision.py`、`quantization.py`、`codecs.py` | 精度描述、FP8/MXFP4 识别与软件解码、导出编码及精度策略 |
| `fp8_native.py`、`quantized_archive.py` | 原生 TE 存储及原编码归档/验证恢复 |
| `hf_io.py`、`safetensors_*.py`、`artifacts.py` | HF / MCore 元数据和存储读写 |
| `streaming.py`、`plan.py` | 复用 DSv4 的 DCP 区域读写、分片规划、量化存储和覆盖检查 |
| `transaction.py`、`world_one.py` | 输出提交保护和单进程执行边界 |
| `convert.py` | 统一入口 |

`../qwen3/mapping.yaml` 仅声明公共 decoder 映射之外的 Q/K norm 和配置差异。`../deepseek_v4/modules.json`、`parameters.json` 声明 DSv4 权重和参数规则；其压缩层、MTP、hash router、词表裁剪等专用策略仍在模型目录，调用公共参数转换、量化及存储实现。已移除原 DSv4 存储模块的兼容转发文件。

## 使用

从仓库根目录执行，转换进程为 world size 1、rank 0：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert hf-to-mcore \
  --mapping tools/checkpoint/qwen3/mapping.yaml \
  --source /models/qwen3 --output /outputs/qwen3-mcore

python -m tools.checkpoint.hf_mcore_dist.convert mcore-to-hf \
  --source /outputs/qwen3-mcore --output /outputs/qwen3-hf
```

通用表入口目前输出普通浮点/整数 DCP；HF FP8 E4M3（128×128 scale）和显式声明的 MXFP4（32 值一组）使用公共 codec 解码。BF16/FP16/FP32、I32/I64 的实际存储来自源元数据。`--dtype auto` 保留普通浮点 tensor 的 dtype，量化输入默认解码为 BF16。显式 dtype 不改变 FP32 和整数 tensor；混合 dtype 融合仍要求最终 dtype 一致。

原编码 `preserve`、原生 TE 对象及归档恢复的读写已归入公共基础设施；当前由 DSv4 的区域映射接入。通用表入口不将任意 permute/QKV 重排隐式应用于量化 payload，而是先解码；它也不会把量化 DCP 当作普通浮点存储导出。DSv4 使用同一个入口：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert deepseek-v4 hf-to-mcore \
  --hf-model /models/dsv4 --mcore-path /outputs/dsv4 \
  --conversion-mode preserve --device cpu
```

量化 payload 的 DCP 序列化/训练加载继续使用现有 `flagscale.train.utils.quantized_checkpoint`、`checkpoint_serialization` 适配器；本次仅修改 `tools/checkpoint`，未修改或收录工作树中这些未追踪文件。通用表的 dense 导入/导出及软件解码不依赖这些适配器，也不导入 Transformers 或构造训练模型。原生 TE 存储仍需对应 TE 环境。

`--dry-run` 检查元数据并打印计划，不创建输出；`--max-shard-size` 控制文件目标大小；`--overwrite` 在转换成功后替换已有输出。每次只物化一个映射操作组，HF writer 另有一个输出分片缓冲。一个大型专家堆叠组仍可能很大；此处没有整模型缓存，也没有固定内存上限的承诺。

导入时在 iteration 目录保存 `hf_export.json`，仅含最终映射模板和 HF config。直接反向转换不需要参考 HF index 或模型类。训练重存后若未携带该文件，需要重新指定 `--mapping`；无法从 MCore args 恢复的字段由配置覆盖补充。`--hf-reference` 可选，用来提供 HF config 和 tokenizer/附属文件；实际源 MCore 配置的映射字段优先。没有参考目录时，不生成 tokenizer。此入口不会修改训练保存流程来自动传递 sidecar。

## 映射配置

每条规则有稳定的 `id`。覆盖为**整条替换**，避免隐式字段合并；新增 id 即增加规则；`disabled: true` 删除已有规则。`extends` 支持相对文件路径，`--mapping-override` 可重复指定，按顺序应用。

```yaml
schema_version: 1
extends: /path/to/base/mapping.yaml
rules:
  - id: fc1
    hf: ['model.layers.{layer}.mlp.up_proj.weight', 'model.layers.{layer}.mlp.gate_proj.weight']
    mcore: ['decoder.layers.{layer}.mlp.linear_fc1.weight']
    op: swiglu
    extra_state: ['decoder.layers.{layer}.mlp.linear_fc1._extra_state/shard_0_1']
  - id: output
    disabled: true
  - id: additional_buffer
    hf: [custom.buffer]
    mcore: [custom.buffer]
ignore:
  hf: [lm_head.weight]
  mcore: [output_layer.weight]
```

这里刻意把顺序写成 `[up, gate]`。即使实际模型期望相反顺序，只要尺寸合法就按配置执行。未映射源 tensor 默认报错；删除规则不等于自动忽略其权重，忽略必须明确声明。`optional: true` 允许整条规则没有源匹配，但不容忍匹配组内缺少部分输入。

`{layer}` 等数字占位符从真实源 key 枚举，不维护层数对应的完整 key 表。堆叠专家等多对一映射可指定 `collect: {expert: $hf.num_experts}`；专家按编号展开，同一专家内部保留 `hf` 列表顺序。需要 TE extra-state 时在规则中声明其全局存储 key；核心不会从模块名称猜测。

支持的操作：

| op | HF → MCore | 反向所需信息 |
| --- | --- | --- |
| `identity`（默认） | 复制/改名 | 无 |
| `concat` / `split` | 按 `axis` 拼接/切分 | 切分方向必须有 `sizes` |
| `swiglu` | 两个同形 tensor 按给定顺序沿轴 0 拼接 | 等分轴 0 |
| `permute` | 按 `dims` 调整轴顺序 | 置换自动求逆 |
| `reshape` | 使用 `mcore_shape` | 必须声明 `hf_shape` |
| `qkv` | 输入 `[q,k,v]`，按 `heads/groups/order` 分组融合 | 同一规则求逆；支持 weight/bias |
| `stack` | 按 `axis` 增加堆叠轴 | 目标名称数量与轴大小一致 |
| `expert_swiglu` | 专家编号优先，每个专家两个同形输入，得到 `[E,2F,H]` | 拆专家轴后等分 FC1 |

`params` 中可以引用 `$hf.field` 或 `$mcore.field`。shape/dtype 和输出区域从源元数据及操作推导，写入报告，不要求维护静态 shape 表。

参数转换采用 `config_map: {HF字段: MCore字段或别名列表}`，也可使用 `{mcore: [...], required: true, hf_default: ...}`。点号表示嵌套字段。`mcore_args` / `hf_config` 是对应方向的显式覆盖；支持 `$hf.*` / `$mcore.*` 引用，以及 `{op: mul, args: [...]}` 等 `add/sub/mul/div/floordiv/not/int` 运算。不存在任意表达式 `eval`。复杂模型约定留在模型适配器。

MXFP4 必须显式标识，不能把任意 U8/I8 buffer 当量化权重：

```yaml
quantization:
  mxfp4: ['model.layers.*.mlp.experts.*.weight']
```

## 从环境仓库提供映射

可以直接引用其他仓库的同格式 YAML/JSON，也可以显式导入已安装模块提供的字典。以下命令将表固化为 JSON，后续不再依赖该仓库导入：

```bash
PYTHONPATH=/path/to/repository:$PYTHONPATH \
python -m tools.checkpoint.hf_mcore_dist.convert export-mapping \
  --mapping python:package.conversion_table:MAPPING \
  --output /tmp/model-mapping.json
```

显式导入会执行指定模块的 Python 顶层代码。此接口读取已声明的转换表，不尝试解析任意 Python 转换函数或自动猜测模型语义。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tools/checkpoint/hf_mcore_dist/tests
```

覆盖配置继承/覆盖、双向布局、非标准布局执行、Qwen3 tiny 双向转换、标准 DCP 读回、任意参数名的 FP8/MXFP4 软件解码及失败时输出保护。验证的是声明操作和存储行为；真实模型的 forward 对齐和训练验证独立进行。
