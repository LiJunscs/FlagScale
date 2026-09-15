# Engram 性能扫描

从已手动跑通训练的环境、在 FlagScale 根目录执行。默认使用当前
`examples/deepseek_v3/conf` 中的两套配置，4 节点、每节点 8 GPU。

```bash
# 只做全网格预检查、生成训练脚本及汇总，不连接 SSH、不启动训练。
DRY_RUN=1 bash sweep_deepseek_engram.sh
DRY_RUN=1 bash sweep_baai_engram.sh

# 实际训练：每组结束并清理完成后才启动下一组。
bash sweep_deepseek_engram.sh
bash sweep_baai_engram.sh
```

每个模型默认 14 组：baseline 扫描 PP={2,4} × TP={1,2,4,8}；optimized
固定 TP=4，扫描 PP={2,4} × Egp={2,4,8}。EP=8、expert TP=1 等参数来自 YAML。
两个模型的 PP=4 均采用 7/8/8/7 层分配。

先做两组训练检查可使用：

```bash
BASELINE_PP_VALUES=2 BASELINE_TP_VALUES=4 \
OPTIMIZED_PP_VALUES=2 OPTIMIZED_EGP_VALUES=4 \
bash sweep_deepseek_engram.sh
```

## 可调整的环境变量

| 变量 | 默认值 / 用途 |
| --- | --- |
| `TRAIN_ITERS` | 25 |
| `WARMUP_ITERS` | 5，统计排除前 5 步 |
| `EXPERIMENT_TIMEOUT` | 1800 秒，训练启动后的整组超时；远程预检查及清理另有超时 |
| `NNODES` / `NPROC_PER_NODE` | 4 / 8，预检查要求 hostfile 和可见设备满足请求 |
| `BASE_PORT` | 10720，每组递增 |
| `OPTIMIZED_TP` | 4，显式固定 optimized 的 TP |
| `BASELINE_PP_VALUES` / `BASELINE_TP_VALUES` | 空格分隔的 baseline 网格 |
| `OPTIMIZED_PP_VALUES` / `OPTIMIZED_EGP_VALUES` | 空格分隔的 optimized 网格 |
| `CONFIG_DIR` | `${FLAGSCALE_ROOT}/examples/deepseek_v3/conf` |
| `SWEEP_OUTPUT_DIR` | 自动创建带模型名和时间的唯一目录；拒绝覆盖已有目录 |
| `PYTHON` | python，可指定已验证训练环境的解释器路径 |

## 运行与失败语义

`run_engram_sweep.py` 使用 FlagScale 的配置处理和前台训练脚本生成逻辑；
由 sweep 自己管理各节点的 SSH 进程，不使用后台提交的成功状态判断训练成功。
本版本支持 SSH、共享文件系统、`per_node_task=false`，不同时启用 heartbeat
或 `enable_monitoring`。所有节点需要能访问实验目录和相同路径的 Python 解释器，
并具备 bash、setsid、flock，以及可读取的 Linux `/proc`（启动前自动检查）。

每组任务在独立进程组中运行。任一节点失败、超时或收到中断后，监督程序清理
本组所有节点，先 TERM、必要时 KILL。清理按各节点唯一、被子进程继承的环境标识
定位进程，因此也覆盖 torchrun 创建独立会话的 worker；优先使用 pidfd，旧 Python
则在发信号前复核标识和进程启动时间。取消标记与启动锁防止延迟到达的 SSH
请求重新启动已取消任务。
清理不使用按程序名匹配的 `pkill`。

成功需要所有节点返回 0，日志达到 `TRAIN_ITERS`，且存在预热后的有效计时。
普通失败在确认清理后继续；清理失败、用户中断或缺失结果状态会停止扫描。
任一实验或汇总失败，整个 sweep 返回非零。两个 wrapper 在同一共享 checkout
通过 `.engram-sweep.lock` 互斥；不会自动避让手动启动的训练或其他 checkout 的任务。

## 输出

- 每组 `.command.txt`、`.launcher.log`、`.exit_code`。
- 每组目录中的 `resolved_config.yaml`、`sweep_result.json`。
- `sweep_nodes/` 中各节点的日志、退出码、进程组和清理日志。
- 训练原始日志仍位于各实验的 `logs/`。
- 顶层 `results.csv`、`results.md`，可重新执行：

```bash
python collect_engram_sweep.py /path/to/sweep-output
```

汇总从单个最完整的 rank 日志取数，避免 torchrun、tee 镜像日志重复计数。
全局 tokens/s = global batch size × sequence length / 平均迭代时间。
失败行可能包含部分计时，只比较 `success` 行；`dryrun` 行不表示训练通过。
固定同一 PP、TP=4 时比较 allreduce 与 Egp=4 的 alltoall，可减少并行布局
变化对通信方式比较的影响。25 步适合初筛，最优候选应延长运行或重复验证。
