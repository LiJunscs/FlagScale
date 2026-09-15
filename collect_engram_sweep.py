#!/usr/bin/env python3
"""Summarize sweep results from training logs, excluding warmup iterations."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ITERATION = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)"
)
THROUGHPUT = re.compile(r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)")
OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory", re.I)
FIELDS = (
    "experiment",
    "status",
    "pp",
    "tp",
    "egp",
    "method",
    "world_size",
    "dp",
    "completed_iters",
    "train_iters",
    "warmup_iters",
    "samples",
    "mean_ms",
    "median_ms",
    "tokens_per_second",
    "tflops_per_gpu",
    "exit_code",
    "cleanup_ok",
    "metrics_log",
)


def summarize(exp_dir, result):
    """Use one rank's most complete log; never count mirrored/rank logs twice."""
    exp_dir = Path(exp_dir)
    for key in ("mean_ms", "median_ms", "tokens_per_second", "tflops_per_gpu"):
        result.pop(key, None)
    best, best_path, oom = {}, "", False
    paths = sorted(
        set(exp_dir.glob("logs/details/**/stdout"))
        | set(exp_dir.glob("logs/details/**/*.log"))
        | set(exp_dir.glob("logs/host_*.output"))
        | set(exp_dir.glob("sweep_nodes/*.log"))
    )
    for path in paths:
        samples = {}
        with path.open(errors="replace") as stream:
            for line in stream:
                oom = oom or bool(OOM.search(line))
                match = ITERATION.search(line)
                if match:
                    step, total, ms = match.groups()
                    if int(total) != result["train_iters"]:
                        continue
                    throughput = THROUGHPUT.search(line)
                    samples[int(step)] = (float(ms), float(throughput[1]) if throughput else None)
        if (max(samples, default=0), len(samples)) > (max(best, default=0), len(best)):
            best, best_path = samples, str(path)
    result["completed_iters"] = max(best, default=0)
    steady = [
        v
        for k, v in sorted(best.items())
        if result["warmup_iters"] < k <= result["train_iters"] and v[0] > 0
    ]
    result.update(samples=len(steady), metrics_log=best_path)
    if steady:
        mean = statistics.mean(v[0] for v in steady)
        flops = [v[1] for v in steady if v[1] is not None]
        result.update(
            mean_ms=round(mean, 3),
            median_ms=round(statistics.median(v[0] for v in steady), 3),
            tokens_per_second=round(
                result["global_batch_size"] * result["seq_length"] * 1000 / mean, 3
            ),
            tflops_per_gpu=round(statistics.mean(flops), 3) if flops else "",
        )
    if result["status"] == "success" and (
        result["completed_iters"] < result["train_iters"] or not steady
    ):
        result["status"] = "incomplete"
        result["exit_code"] = 1
    if oom and result["status"] in {"failed", "incomplete"}:
        result["status"] = "oom"
    return result


def collect(root):
    root = Path(root)
    rows = [
        summarize(path.parent, json.loads(path.read_text()))
        for path in sorted(root.glob("*/sweep_result.json"))
    ]
    if not rows:
        raise ValueError(f"No sweep_result.json files in {root}")
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    columns = ("experiment", "status", "completed_iters", "samples", "mean_ms", "tokens_per_second")
    lines = [
        "# Engram sweep",
        "",
        "Times exclude warmup; tokens/s is global batch × sequence length / mean iteration time.",
        "Only successful rows should be used for performance comparisons.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += [
        "| " + " | ".join(str(row.get(k, "")).replace("|", "\\|") for k in columns) + " |"
        for row in rows
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    return int(any(row["status"] not in {"success", "dryrun"} for row in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    raise SystemExit(collect(args.output_dir))
