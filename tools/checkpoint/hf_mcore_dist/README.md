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

本文件是这套 HF ↔ MCore 转换框架的统一使用文档。DSv4 的旧 README 已移除；旧评审记录和加载示例不能作为当前命令行接口说明。

## 使用

从仓库根目录执行，直接使用 `python -m`，不要通过多 rank launcher 启动。转换进程必须为 world size 1、rank 0。

只需指定模型的 `mapping.yaml`，公共权重表和参数表通过 `extends` 自动加载。例如 Qwen3：

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

### DSv4：HF → MCore

生成普通 BF16 DCP；已有 FP32 参数和整数 tensor 保持原类型：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /models/dsv4-hf \
  --output /outputs/dsv4-mcore \
  --conversion-mode dense --dtype bf16 \
  --device cpu --max-shard-size 4GB
```

保留原始 FP8/MXFP4 payload 和 scale：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert hf-to-mcore \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /models/dsv4-hf \
  --output /outputs/dsv4-preserved \
  --conversion-mode preserve --dtype auto \
  --device cpu --max-shard-size 4GB
```

源 HF 目录需有 `config.json` 和实际 safetensors 权重；多分片通过 `model.safetensors.index.json` 定位，也支持单文件 `model.safetensors`。不需要构造 HF 模型或提供另一个 MCore checkpoint。

### DSv4：MCore → HF

导出训练后的 checkpoint，显式指定映射及可选 HF 附属文件目录：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert mcore-to-hf \
  --mapping tools/checkpoint/deepseek_v4/mapping.yaml \
  --source /checkpoints/dsv4/iter_0000100 \
  --hf-reference /models/dsv4-hf \
  --output /outputs/dsv4-export \
  --dtype auto --max-shard-size 4GB
```

`--source` 可以是 checkpoint 根目录或完成保存的 iteration 目录。`--hf-reference` 提供配置、可选命名索引和 tokenizer 等附属文件，不读取参考模型权重值；目标尺寸和精度以实际源为准。源 args 缺少真实词表大小时，追加 `--vocab-size <未填充的词表大小>`，不能用参考模型大小或 `padded_vocab_size` 猜测。

直接导回本工具刚生成的 checkpoint，可使用其 `hf_export.json`，省略映射和参考目录：

```bash
python -m tools.checkpoint.hf_mcore_dist.convert mcore-to-hf \
  --source /outputs/dsv4-preserved \
  --output /outputs/dsv4-roundtrip
```

没有 `--hf-reference` 时不复制 tokenizer。若要把 preserve DCP 解码导出为 FP32，在反向命令中追加 `--conversion-mode dense --dtype fp32`。

### 检查与覆盖

在上述转换命令中追加以下选项即可：

| 选项 | 行为 |
| --- | --- |
| `--dry-run` | 检查元数据并打印计划，不创建权重输出；零值检查等数值验证留待实际执行 |
| `--mapping-override /path/override.yaml` | 按规则 id 覆盖、增加或禁用；可重复指定，按顺序应用 |
| `--overwrite` | 允许替换已有非空输出目录；转换成功后才提交替换 |
| `--max-shard-size 4GB` | 输出分片的目标大小，不是硬上限；单个大区域可能超过该值 |
| `--device cpu` | 解码运算设备，默认 CPU；也支持 `cuda[:N]`，不决定训练设备能力 |

输出目录不能与输入目录重叠；`--overwrite` 不会解除这一限制。

旧的 `deepseek-v4` 分派子命令、`convert_deepseek_v4.py` 和专用计划文件入口已经删除，不保留兼容转发。preserve 描述现在记录物理区域位置；缺少这些字段的早期转换产物需重新转换。

preserve DCP 的序列化/训练加载依赖环境中的 `flagscale.train.utils.quantized_checkpoint`、`checkpoint_serialization` 适配器；使用前需确保转换和加载环境均具备这些模块。通用表的 dense 导入/导出及软件解码不依赖这些适配器，也不导入 Transformers 或构造训练模型。原生 TE 存储仍需对应 TE 环境。

按源矩阵和目标区域写入，专家融合不物化整个专家组。QKV 同一个目标内复用源矩阵，最后一个片段写完即释放；反向逐 HF tensor 组装，HF writer 另有一个输出分片缓冲。这里没有整模型缓存，也不承诺固定内存上限。

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
