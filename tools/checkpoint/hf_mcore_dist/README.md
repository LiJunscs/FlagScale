# HF ↔ MCore DCP 转换

`hf_mcore_dist` 存放公共转换基础设施。模型目录只增加必要的映射配置；确实无法由常见操作表达的规则保留为模型专用代码。转换器执行用户指定的布局，检查存储和操作是否可执行，不认证模型语义或前向结果等价。

## 代码组织

| 文件 | 职责 |
| --- | --- |
| `mapping.py`、`mapping.yaml` | 公共 dense/MLA/MoE 权重表、继承、覆盖与名称模板展开 |
| `operators.py`、`engine.py`、`regions.py` | 双向布局操作、metadata 预检查及 HF fragment ↔ DCP 区域编译 |
| `config.py`、`parameters.yaml` | 公共参数转换执行及可持续扩展的字段对应字典 |
| `precision.py`、`quantization.py`、`codecs.py` | 精度描述、FP8/MXFP4 识别与软件解码、导出编码及精度策略 |
| `fp8_native.py`、`quantized_archive.py` | 原生 TE 存储及原编码归档/验证恢复 |
| `hf_io.py`、`safetensors_*.py`、`artifacts.py` | HF / MCore 元数据和存储读写 |
| `streaming.py`、`plan.py` | 复用 DSv4 的 DCP 区域读写、分片规划、量化存储和覆盖检查 |
| `transaction.py`、`world_one.py` | 输出提交保护和单进程执行边界 |
| `convert.py` | 统一入口 |

配置继承关系为：

```text
hf_mcore_dist/parameters.yaml    公共参数名/别名字典，不设置模型默认值
             ↓
hf_mcore_dist/mapping.yaml       公共命名、dense/MLA/MoE 权重规则及基础布局约定
             ↓
qwen3/mapping.yaml               Qwen3 的 norm 存在性及配置差异
或 deepseek_v4/mapping.yaml      DSv4 的新增模块、命名/布局覆盖、参数语义与默认值差异
```

DSv4 不再维护独立的 `parameters.yaml`。常见参数、MLA 投影、路由器、专家和共享专家映射集中在公共表中。模型文件只保存差异；相同规则通过继承复用，不复制完整表。公共表中的可选模块不存在时，连对应的 `collect`/`params` 配置也不会读取，因此 dense 模型不需要提供 MoE 参数。

`deepseek_v4/adapter.py` 只负责压缩布局/YaRN 规范化和关闭 router bias 时的约定，不构造另一条转换流程。

新增模型参数的维护规则：无歧义的 HF↔MCore 字段对应加入公共 `parameters.yaml`，源字段不存在时跳过；模型特有默认值、派生值和不同语义留在模型覆盖中。例如公共 `head_dim → kv_channels`，DSv4 覆盖为 `head_dim → v_head_dim`；DSv4 的 `sliding_window → csa_window_size` 也留在模型文件。公共字典的增长不应隐式启用新模块或注入其他模型的默认值。

既有未追踪的 DSv4 文档和加载示例未修改，其旧导入路径不再适用于本版；本文件描述当前入口。

## 使用

从仓库根目录执行，转换进程为 world size 1、rank 0：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert hf-to-mcore \
  --mapping tools/checkpoint/qwen3/mapping.yaml \
  --source /models/qwen3 --output /outputs/qwen3-mcore

python -m tools.checkpoint.hf_mcore_dist.convert mcore-to-hf \
  --source /outputs/qwen3-mcore --output /outputs/qwen3-hf
```

所有映射表均支持 `--conversion-mode dense`（导入默认）和 `preserve`。FP8 E4M3（128×128 scale）与显式声明的 MXFP4（32 值一组）由公共 codec 处理；BF16/FP16/FP32、I32/I64 以实际源存储为准。

- `dense`：解码量化输入，输出普通 DCP tensor。`--dtype auto` 保留普通浮点 dtype、默认将量化输入解码为 BF16；显式 dtype 保留 FP32 和整数 tensor。
- `preserve`：量化矩阵作为完整 payload/scale 区域保存，支持拼接、专家堆叠及混合 dtype 融合，要求 `--dtype auto`。不会为任意切片、permute 或 reshape 隐式重新计算 scale；无法保留原编码的布局在写入前报错，需使用 dense。
- 反向默认跟随实际源存储：普通 DCP 导出浮点，preserve DCP 导出原编码；可显式用 `--conversion-mode dense --dtype fp32` 解码导出。原始编码按实际 DCP 区域绑定，用户交换同形参数时也按声明执行。当前入口不导入历史原生 TE pickle checkpoint；TE/归档工具仍是公共模块中的底层 API。

DSv4 使用相同命令及参数：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /models/dsv4 --output /outputs/dsv4 \
  --conversion-mode preserve --device cpu

python -m tools.checkpoint.hf_mcore_dist.convert mcore-to-hf \
  --source /outputs/dsv4 --output /outputs/dsv4-hf
```

旧的 `deepseek-v4` 分派子命令、`convert_deepseek_v4.py` 和专用计划文件入口已经删除，不保留兼容转发。preserve 描述现在记录物理区域位置；缺少这些字段的早期转换产物需重新转换。

量化 payload 的 DCP 序列化/训练加载继续使用现有 `flagscale.train.utils.quantized_checkpoint`、`checkpoint_serialization` 适配器；本次仅修改 `tools/checkpoint`，未修改或收录工作树中这些未追踪文件。通用表的 dense 导入/导出及软件解码不依赖这些适配器，也不导入 Transformers 或构造训练模型。原生 TE 存储仍需对应 TE 环境。

`--dry-run` 检查元数据并打印计划，不创建输出；`--max-shard-size` 控制文件目标大小；`--overwrite` 在转换成功后替换已有输出。按源矩阵和目标区域写入，专家融合不物化整个专家组。QKV 同一个目标内复用源矩阵，最后一个片段写完即释放；反向逐 HF tensor 组装，HF writer 另有一个输出分片缓冲。这里没有整模型缓存，也不承诺固定内存上限。

导入时在 iteration 目录保存 `hf_export.json`，仅含映射模板、HF config 和去重后的序列化名称模板。直接反向转换不需要参考 HF index 或模型类。训练重存后若未携带该文件，需要重新指定 `--mapping`；无法从 MCore args 恢复的字段由配置覆盖补充。`--hf-reference` 可选，用来提供 HF config 和 tokenizer/附属文件；实际源 MCore 配置的映射字段优先。没有参考目录时，不生成 tokenizer。此入口不会修改训练保存流程来自动传递 sidecar。

## 映射配置

每条规则有稳定的 `id`。覆盖为**整条替换**，避免隐式字段合并；新增 id 即增加规则；`disabled: true` 删除已有规则。`extends` 支持相对文件路径，`--mapping-override` 可重复指定，按顺序应用。

```yaml
schema_version: 1
extends: /path/to/base/mapping.yaml
rules:
  - id: fc1
    scope: layer
    hf: [ffn.w3.weight, ffn.w1.weight]
    mcore: [mlp.linear_fc1.weight]
    op: swiglu
    extra_state: [mlp.linear_fc1._extra_state/shard_0_1]
  - id: output
    disabled: true
  - id: additional_buffer
    hf: [custom.buffer]
    mcore: [custom.buffer]
ignore:
  hf: [head.weight]
  mcore: [output_layer.weight]
```

继承公共表时，`hf` 和 `ignore.hf` 使用命名规范化后的名字；没有继承时默认使用原始名字。上例中的 `w3/w1` 对应 up/gate，刻意按 `[up, gate]` 排列。即使实际模型期望相反顺序，只要尺寸合法就按配置执行。未映射源 tensor 默认报错；删除规则不等于自动忽略其权重，忽略必须明确声明。`optional: true` 允许整条规则没有源匹配，但不容忍匹配组内缺少部分输入。

`{layer}` 等数字占位符从真实源 key 枚举，不维护层数对应的完整 key 表。`scopes` 可给同一组规则添加 decoder/MTP 等不同前缀；可选 scope 没有匹配时跳过。堆叠专家等多对一映射可指定 `collect: {expert: $hf.num_experts}`；专家按编号展开，同一专家内部保留 `hf` 列表顺序。需要 TE extra-state 时在规则中声明其全局存储 key；核心不会从模块名称猜测。extra-state 模板中的 `{expert_count}` 由 `collect.expert` 自动提供。

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

参数转换采用 `config_map: {HF字段: MCore字段或别名列表}`，也可使用 `{mcore: [...], required: true, hf_default: ...}`。点号表示嵌套字段。`mcore_args` / `hf_config` 是对应方向的显式覆盖；支持 `$hf.*` / `$mcore.*` 引用，以及 `{op: mul, args: [...]}` 等 `add/sub/mul/div/floordiv/exactdiv/not/int` 运算。不存在任意表达式 `eval`。复杂模型约定留在模型适配器。

MXFP4 必须显式标识，不能把任意 U8/I8 buffer 当量化权重：

```yaml
quantization:
  mxfp4: ['layers.*.ffn.experts.*.weight']
```

### 显式例外

- `export_slice: {axis: 0, size: $hf.vocab_size}`：identity 规则反向导出时裁去声明的尾部 padding，真实大小来自 DCP 元数据，裁剪大小必须由配置提供。DSv4 源缺少真实词表大小时可用 `--vocab-size`，不会借参考模型猜测。
- `zero_checks: {hf: [pattern]}`：明确省略匹配的源 tensor，实际执行时先证明其值为零。dry-run 仅记录待检查项。
- `generate: {side: hf, shape: [2], dtype: F32, value: 0}`：规则的输入列表为空，输出为声明的常量 tensor。
- `naming`：可声明前缀移除、路径段别名及字符串替换。实际输出名称来自保存的名称模板或可选 HF 参考索引；量化编码仍来自实际源存储。

不再自动丢弃不兼容 MTP。需要排除时明确提供 `ignore.hf: ['mtp.*']`、`ignore.mcore: ['mtp.layers.*']`，并将 `mcore_args.mtp_num_layers`、`hf_config.num_nextn_predict_layers` 设为 0。未知主干/辅助 tensor 默认报错，不会通过隐式丢弃绕过错误。

`adapter` 是可选的显式 Python 模块：仅提供源配置规范化、目标配置收尾和模型特有映射策略；不定义第二条执行链。大多数模型只需要配置文件。

## 从环境仓库提供映射

可以直接引用其他仓库的同格式 YAML/JSON，也可以显式导入已安装模块提供的字典。以下命令将表固化为 JSON；若未声明外部 adapter，后续不再依赖该仓库导入：

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

覆盖配置继承/覆盖、双向布局、非标准布局执行、Qwen3/DSv4 tiny 双向转换、标准 DCP 读回、区域拼装、MTP/bias、显式词表裁剪和 preserve 原字节恢复、任意参数名的 FP8/MXFP4 软件解码及失败时输出保护。验证的是声明操作和存储行为；真实模型的 forward 对齐和训练验证独立进行。
