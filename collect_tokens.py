#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


# ============================================================
# Benchmark protocol
# ============================================================

TRAIN_ITERS = 25
LAST_N_ITERS = 20

SUCCESS_THRESHOLD = 0.10


# ============================================================
# Generic number
# ============================================================

NUMBER = r"[0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"


# ============================================================
# Experiment name
#
# DeepSeek-Engram-baseline-pp2-tp4
# DeepSeek-Engram-optimized-pp4-egp8
#
# BAAI-Engram-baseline-pp2-tp4
# ...
# ============================================================

EXP_RE = re.compile(
    r"^(?P<model>.+)-"
    r"(?P<mode>baseline|optimized)-"
    r"pp(?P<pp>\d+)-"
    r"(?:(?:tp(?P<tp>\d+))|(?:egp(?P<egp>\d+)))$",
    re.IGNORECASE,
)


# ============================================================
# FlagScale host directories
#
# host_0_10.8.1.106
# ============================================================

HOST_RE = re.compile(
    r"^host_(\d+)_"
)

ATTEMPT_RE = re.compile(
    r"^attempts?_(\d+)$"
)


# ============================================================
# Parameters
#
# Total number of parameters in billions: 26.73
# ============================================================

PARAM_RE = re.compile(
    rf"Total\s+number\s+of\s+parameters\s+in\s+billions"
    rf"\s*:\s*({NUMBER})",
    re.IGNORECASE,
)


# ============================================================
# Megatron args
# ============================================================

TP_RE = re.compile(
    r"\btensor_model_parallel_size\b"
    r"\s*(?:=|:|\.+)?\s*(\d+)",
    re.IGNORECASE,
)

EGP_RE = re.compile(
    r"\bengram_embedding_parallel_size\b"
    r"\s*(?:=|:|\.+)?\s*(\d+)",
    re.IGNORECASE,
)


# ============================================================
# Iteration
#
# iteration       19/      25
# ============================================================

ITER_RE = re.compile(
    r"\biteration\s+(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)


# ============================================================
# Direct Tokens/s/GPU
# ============================================================

TOKENS_PATTERNS = [

    # tokens per GPU (tokens/s/GPU): 1234.5
    re.compile(
        rf"tokens?\s+per\s+GPU\s*"
        rf"\(\s*tokens?\s*/\s*s\s*/\s*GPU\s*\)"
        rf"\s*[:=]\s*({NUMBER})",
        re.IGNORECASE,
    ),

    # tokens/s/GPU: 1234.5
    re.compile(
        rf"tokens?\s*/\s*s\s*/\s*GPU"
        rf"\s*[:=]\s*({NUMBER})",
        re.IGNORECASE,
    ),

    # tokens per second per GPU: 1234.5
    re.compile(
        rf"tokens?\s+per\s+second\s+per\s+GPU"
        rf"\s*[:=]\s*({NUMBER})",
        re.IGNORECASE,
    ),
]


# ============================================================
# Official Megatron fallback
#
# throughput per GPU (TFLOP/s/GPU): 93.1
# ============================================================

TFLOPS_RE = re.compile(
    rf"throughput\s+per\s+GPU\s*"
    rf"\(\s*TFLOP/s/GPU\s*\)"
    rf"\s*:\s*({NUMBER})",
    re.IGNORECASE,
)


# ============================================================
# Helpers
# ============================================================

def parse_float(value):
    return float(
        value.replace(",", "")
    )


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.2f}"


def read_text(path):
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


# ============================================================
# Host discovery
# ============================================================

def get_host_index(path):

    match = HOST_RE.match(
        path.name
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def find_hosts(details_dir):

    hosts = []

    if not details_dir.is_dir():
        return hosts


    for path in details_dir.iterdir():

        if not path.is_dir():
            continue

        index = get_host_index(
            path
        )

        if index is None:
            continue

        hosts.append(
            (index, path)
        )


    hosts.sort(
        key=lambda x: x[0]
    )

    return hosts


# ============================================================
# Rank logs
# ============================================================

def find_rank_logs(
    host_dir,
    rank,
):

    result = []


    for log in host_dir.rglob(
        "stdout.log"
    ):

        if log.parent.name != str(rank):
            continue

        result.append(log)


    return result


def get_attempt_number(path):

    for parent in path.parents:

        match = ATTEMPT_RE.match(
            parent.name
        )

        if match:

            return int(
                match.group(1)
            )


    return -1


def get_timestamp(
    log,
    host_dir,
):

    try:

        relative = log.relative_to(
            host_dir
        )

    except ValueError:

        return ""


    if not relative.parts:
        return ""


    # host_x/
    #   20260913_224728.816980/
    #
    return relative.parts[0]


def log_sort_key(
    log,
    host_dir,
):

    timestamp = get_timestamp(
        log,
        host_dir,
    )

    attempt = get_attempt_number(
        log
    )


    try:
        mtime = log.stat().st_mtime
    except OSError:
        mtime = 0


    return (
        timestamp,
        attempt,
        mtime,
    )


def find_latest_rank_log(
    host_dir,
    rank,
):

    logs = find_rank_logs(
        host_dir,
        rank,
    )


    if not logs:
        return None


    logs.sort(
        key=lambda x: log_sort_key(
            x,
            host_dir,
        )
    )


    return logs[-1]


# ============================================================
# Determine last local rank
#
# 0 ... 7 -> 7
# ============================================================

def find_local_ranks(host_dir):

    ranks = set()


    for log in host_dir.rglob(
        "stdout.log"
    ):

        try:

            rank = int(
                log.parent.name
            )

        except ValueError:

            continue


        ranks.add(rank)


    return sorted(ranks)


# ============================================================
# Match same FlagScale launch on another host
# ============================================================

def find_matching_rank_log(
    source_log,
    source_host,
    target_host,
    target_rank,
):

    try:

        relative = source_log.relative_to(
            source_host
        )

    except ValueError:

        return None


    parts = list(
        relative.parts
    )


    # Exact same relative path except rank.
    if (
        len(parts) >= 2
        and parts[-1] == "stdout.log"
        and parts[-2].isdigit()
    ):

        parts[-2] = str(
            target_rank
        )

        candidate = (
            target_host
            / Path(*parts)
        )


        if candidate.exists():
            return candidate


    # --------------------------------------------------------
    # Fallback:
    #
    # same timestamp + same attempt
    # --------------------------------------------------------

    target_timestamp = get_timestamp(
        source_log,
        source_host,
    )

    target_attempt = get_attempt_number(
        source_log
    )


    candidates = []


    for log in find_rank_logs(
        target_host,
        target_rank,
    ):

        if (
            get_timestamp(
                log,
                target_host,
            )
            != target_timestamp
        ):
            continue


        if (
            get_attempt_number(log)
            != target_attempt
        ):
            continue


        candidates.append(log)


    if not candidates:
        return None


    candidates.sort(
        key=lambda x: log_sort_key(
            x,
            target_host,
        )
    )


    return candidates[-1]


# ============================================================
# Locate rank-0 and throughput logs
# ============================================================

def locate_logs(exp_dir):

    details_dir = (
        exp_dir
        / "logs"
        / "details"
    )


    hosts = find_hosts(
        details_dir
    )


    if not hosts:

        return None


    first_host = hosts[0][1]
    last_host = hosts[-1][1]


    local_ranks = find_local_ranks(
        last_host
    )


    if not local_ranks:
        return None


    last_local_rank = max(
        local_ranks
    )


    # --------------------------------------------------------
    # Throughput:
    #
    # last host / last rank
    # --------------------------------------------------------

    throughput_log = (
        find_latest_rank_log(
            last_host,
            last_local_rank,
        )
    )


    if throughput_log is None:
        return None


    # --------------------------------------------------------
    # Parameters:
    #
    # first host / rank 0 / SAME launch
    # --------------------------------------------------------

    parameter_log = (
        find_matching_rank_log(
            throughput_log,
            last_host,
            first_host,
            0,
        )
    )


    if parameter_log is None:

        parameter_log = (
            find_latest_rank_log(
                first_host,
                0,
            )
        )


    return {
        "first_host":
            first_host,

        "last_host":
            last_host,

        "last_local_rank":
            last_local_rank,

        "parameter_log":
            parameter_log,

        "throughput_log":
            throughput_log,
    }


# ============================================================
# Parameters
# ============================================================

def extract_parameters_billions(text):

    values = []


    for match in PARAM_RE.finditer(
        text
    ):

        value = parse_float(
            match.group(1)
        )

        if math.isfinite(value):
            values.append(value)


    if not values:
        return None


    return values[-1]


# ============================================================
# Megatron config values
# ============================================================

def extract_int(text, pattern):

    match = pattern.search(
        text
    )

    if match is None:
        return None


    return int(
        match.group(1)
    )


# ============================================================
# Extract direct token throughput
# ============================================================

def extract_token_value(line):

    for pattern in TOKENS_PATTERNS:

        match = pattern.search(
            line
        )

        if match is None:
            continue


        value = parse_float(
            match.group(1)
        )


        if (
            math.isfinite(value)
            and value > 0
        ):

            return value


    return None


# ============================================================
# Parse iteration metrics
# ============================================================

def parse_iteration_metrics(text):

    # iteration -> metric
    token_by_iter = {}
    tflops_by_iter = {}

    # Also support token output without iteration number.
    standalone_tokens = []


    for line in text.splitlines():

        iteration_match = (
            ITER_RE.search(line)
        )


        iteration = None


        if iteration_match:

            iteration = int(
                iteration_match.group(1)
            )


        # ----------------------------------------------------
        # Tokens/s/GPU
        # ----------------------------------------------------

        token_value = extract_token_value(
            line
        )


        if token_value is not None:

            if iteration is not None:

                token_by_iter[
                    iteration
                ] = token_value

            else:

                standalone_tokens.append(
                    token_value
                )


        # ----------------------------------------------------
        # TFLOP/s/GPU
        # ----------------------------------------------------

        tflops_match = (
            TFLOPS_RE.search(line)
        )


        if (
            tflops_match
            and iteration is not None
        ):

            value = parse_float(
                tflops_match.group(1)
            )


            if (
                math.isfinite(value)
                and value > 0
            ):

                tflops_by_iter[
                    iteration
                ] = value


    # ========================================================
    # Last 20 iterations
    # ========================================================

    token_values = []


    if token_by_iter:

        iterations = sorted(
            token_by_iter
        )

        iterations = iterations[
            -LAST_N_ITERS:
        ]

        token_values = [
            token_by_iter[i]
            for i in iterations
        ]

    elif standalone_tokens:

        token_values = (
            standalone_tokens[
                -LAST_N_ITERS:
            ]
        )


    tflops_iterations = sorted(
        tflops_by_iter
    )

    tflops_iterations = (
        tflops_iterations[
            -LAST_N_ITERS:
        ]
    )


    tflops_values = [
        tflops_by_iter[i]
        for i in tflops_iterations
    ]


    return (
        token_values,
        tflops_values,
    )


# ============================================================
# Statistics
# ============================================================

def stats(values):

    if not values:

        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }


    return {
        "n":
            len(values),

        "mean":
            statistics.mean(values),

        "median":
            statistics.median(values),

        "min":
            min(values),

        "max":
            max(values),

        "std":
            (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
    }


# ============================================================
# Parse experiment name
# ============================================================

def parse_exp_name(name):

    match = EXP_RE.match(
        name
    )


    if not match:
        return None


    result = match.groupdict()


    return {
        "model":
            result["model"],

        "mode":
            result["mode"].lower(),

        "pp":
            int(result["pp"]),

        "name_tp":
            (
                int(result["tp"])
                if result["tp"]
                else None
            ),

        "name_egp":
            (
                int(result["egp"])
                if result["egp"]
                else None
            ),
    }


# ============================================================
# Parse one experiment
# ============================================================

def parse_experiment(exp_dir):

    name_info = parse_exp_name(
        exp_dir.name
    )


    if name_info is None:
        return None


    located = locate_logs(
        exp_dir
    )


    if located is None:

        return {
            **name_info,

            "exp_name":
                exp_dir.name,

            "status":
                "NO_LOG",

            "parameters_billions":
                None,

            "actual_tp":
                name_info["name_tp"],

            "actual_egp":
                name_info["name_egp"],

            "tokens":
                stats([]),

            "tflops":
                stats([]),

            "parameter_log":
                "-",

            "throughput_log":
                "-",
        }


    # ========================================================
    # Rank 0
    # ========================================================

    parameter_log = (
        located["parameter_log"]
    )


    if parameter_log:

        rank0_text = read_text(
            parameter_log
        )

    else:

        rank0_text = ""


    parameters_billions = (
        extract_parameters_billions(
            rank0_text
        )
    )


    actual_tp = extract_int(
        rank0_text,
        TP_RE,
    )


    actual_egp = extract_int(
        rank0_text,
        EGP_RE,
    )


    # Fallback to experiment name.
    if actual_tp is None:

        actual_tp = (
            name_info["name_tp"]
        )


    if actual_egp is None:

        actual_egp = (
            name_info["name_egp"]
        )


    # ========================================================
    # Last host / last rank
    # ========================================================

    throughput_log = (
        located["throughput_log"]
    )


    throughput_text = read_text(
        throughput_log
    )


    token_values, tflops_values = (
        parse_iteration_metrics(
            throughput_text
        )
    )


    token_stats = stats(
        token_values
    )


    tflops_stats = stats(
        tflops_values
    )


    # At least one complete last-20 metric is sufficient.
    if (
        token_stats["n"] >= LAST_N_ITERS
        or tflops_stats["n"] >= LAST_N_ITERS
    ):

        status = "OK"

    else:

        status = "INSUFFICIENT_ITERS"


    return {
        **name_info,

        "exp_name":
            exp_dir.name,

        "status":
            status,

        "parameters_billions":
            parameters_billions,

        "actual_tp":
            actual_tp,

        "actual_egp":
            actual_egp,

        "tokens":
            token_stats,

        "tflops":
            tflops_stats,

        "parameter_log":
            (
                str(parameter_log)
                if parameter_log
                else "-"
            ),

        "throughput_log":
            str(throughput_log),
    }


# ============================================================
# Discover experiments
# ============================================================

def find_experiments(root):

    result = []


    for path in root.iterdir():

        if not path.is_dir():
            continue


        if parse_exp_name(
            path.name
        ) is None:

            continue


        result.append(path)


    result.sort(
        key=lambda x: x.name
    )


    return result


# ============================================================
# Select one common metric for the whole sweep
#
# Important:
#
# Never compare Tokens/s/GPU from one experiment against
# TFLOP/s/GPU from another experiment.
#
# If every valid experiment has 20 token samples:
#
#   Tokens/s/GPU
#
# otherwise, if every valid experiment has 20 TFLOP samples:
#
#   TFLOP/s/GPU
# ============================================================

def select_metric(results):

    usable = [
        r for r in results
        if r["status"] == "OK"
    ]


    if not usable:
        return None


    if all(
        r["tokens"]["n"] >= LAST_N_ITERS
        for r in usable
    ):

        return "tokens"


    if all(
        r["tflops"]["n"] >= LAST_N_ITERS
        for r in usable
    ):

        return "tflops"


    return None


def metric_name(kind):

    if kind == "tokens":
        return "Tokens/s/GPU"

    if kind == "tflops":
        return "TFLOP/s/GPU"

    return "-"


def get_metric_stats(
    result,
    kind,
):

    if kind == "tokens":
        return result["tokens"]

    if kind == "tflops":
        return result["tflops"]

    return stats([])


# ============================================================
# Grid sorting
# ============================================================

def result_sort_key(r):

    if r["mode"] == "baseline":

        return (
            0,
            r["pp"],
            r["name_tp"] or -1,
        )


    return (
        1,
        r["pp"],
        r["name_egp"] or -1,
    )


# ============================================================
# Full console print
# ============================================================

def print_grid(
    results,
    kind,
):

    label = metric_name(
        kind
    )


    baseline = [
        r for r in results
        if r["mode"] == "baseline"
    ]


    optimized = [
        r for r in results
        if r["mode"] == "optimized"
    ]


    print()
    print("=" * 110)
    print("Baseline: PP x TP")
    print("=" * 110)

    print(
        f"{'PP':>4} "
        f"{'TP':>4} "
        f"{'Params(B)':>12} "
        f"{'20-iters mean':>18} "
        f"{'Metric':>16} "
        f"{'Samples':>8} "
        f"{'Status':>20}"
    )

    print("-" * 110)


    for r in baseline:

        s = get_metric_stats(
            r,
            kind,
        )


        print(
            f"{r['pp']:>4} "
            f"{str(r['actual_tp']):>4} "
            f"{fmt(r['parameters_billions']):>12} "
            f"{fmt(s['mean']):>18} "
            f"{label:>16} "
            f"{s['n']:>8} "
            f"{r['status']:>20}"
        )


    print()
    print("=" * 110)
    print("Our optimized: PP x Egp")
    print("=" * 110)

    print(
        f"{'PP':>4} "
        f"{'Egp':>4} "
        f"{'TP':>4} "
        f"{'Params(B)':>12} "
        f"{'20-iters mean':>18} "
        f"{'Metric':>16} "
        f"{'Samples':>8} "
        f"{'Status':>20}"
    )

    print("-" * 110)


    for r in optimized:

        s = get_metric_stats(
            r,
            kind,
        )


        print(
            f"{r['pp']:>4} "
            f"{str(r['actual_egp']):>4} "
            f"{str(r['actual_tp']):>4} "
            f"{fmt(r['parameters_billions']):>12} "
            f"{fmt(s['mean']):>18} "
            f"{label:>16} "
            f"{s['n']:>8} "
            f"{r['status']:>20}"
        )


# ============================================================
# Find best
# ============================================================

def find_best(
    results,
    mode,
    kind,
):

    candidates = []


    for r in results:

        if r["mode"] != mode:
            continue


        if r["status"] != "OK":
            continue


        s = get_metric_stats(
            r,
            kind,
        )


        if s["n"] < LAST_N_ITERS:
            continue


        if s["mean"] is None:
            continue


        candidates.append(r)


    if not candidates:
        return None


    return max(
        candidates,
        key=lambda r: get_metric_stats(
            r,
            kind,
        )["mean"],
    )


# ============================================================
# Summary / comparison
# ============================================================

def compare_best(
    results,
    kind,
):

    baseline = find_best(
        results,
        "baseline",
        kind,
    )


    optimized = find_best(
        results,
        "optimized",
        kind,
    )


    if (
        baseline is None
        or optimized is None
    ):

        return None


    baseline_mean = (
        get_metric_stats(
            baseline,
            kind,
        )["mean"]
    )


    optimized_mean = (
        get_metric_stats(
            optimized,
            kind,
        )["mean"]
    )


    improvement = (
        optimized_mean
        - baseline_mean
    ) / baseline_mean


    success = (
        improvement
        >= SUCCESS_THRESHOLD
    )


    return {
        "baseline":
            baseline,

        "optimized":
            optimized,

        "baseline_mean":
            baseline_mean,

        "optimized_mean":
            optimized_mean,

        "improvement":
            improvement,

        "success":
            success,
    }


def print_comparison(
    comparison,
    kind,
):

    print()
    print("=" * 110)
    print("Best configuration comparison")
    print("=" * 110)


    if comparison is None:

        print(
            "Unable to compare baseline and optimized."
        )

        return


    b = comparison["baseline"]
    o = comparison["optimized"]


    print(
        f"Metric              : {metric_name(kind)}"
    )

    print()


    print(
        "Baseline best       : "
        f"PP={b['pp']}, "
        f"TP={b['actual_tp']}"
    )

    print(
        f"Baseline throughput : "
        f"{comparison['baseline_mean']:.2f}"
    )


    print()


    print(
        "Our optimized best  : "
        f"PP={o['pp']}, "
        f"Egp={o['actual_egp']}, "
        f"TP={o['actual_tp']}"
    )

    print(
        f"Optimized throughput: "
        f"{comparison['optimized_mean']:.2f}"
    )


    print()


    print(
        "Improvement         : "
        f"{comparison['improvement'] * 100:.2f}%"
    )

    print(
        "Success threshold   : >= 10.00%"
    )


    print()

    if comparison["success"]:

        print(
            "RESULT              : SUCCESS"
        )

    else:

        print(
            "RESULT              : FAIL"
        )


    print("=" * 110)


# ============================================================
# CSV
# ============================================================

CSV_FIELDS = [
    "model",
    "mode",

    "pp",
    "tp",
    "egp",

    "parameters_billions",

    "metric",
    "mean_metric",
    "median_metric",
    "min_metric",
    "max_metric",
    "std_metric",
    "metric_samples",

    "mean_tokens_s_gpu",
    "tokens_samples",

    "mean_tflops_s_gpu",
    "tflops_samples",

    "status",

    "exp_name",

    "parameter_log",
    "throughput_log",
]


def write_csv(
    results,
    kind,
    output,
):

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()


        for r in results:

            selected = get_metric_stats(
                r,
                kind,
            )


            writer.writerow(
                {
                    "model":
                        r["model"],

                    "mode":
                        (
                            "Baseline"
                            if r["mode"] == "baseline"
                            else "Our optimized"
                        ),

                    "pp":
                        r["pp"],

                    "tp":
                        r["actual_tp"],

                    "egp":
                        r["actual_egp"],

                    "parameters_billions":
                        r["parameters_billions"],

                    "metric":
                        metric_name(kind),

                    "mean_metric":
                        selected["mean"],

                    "median_metric":
                        selected["median"],

                    "min_metric":
                        selected["min"],

                    "max_metric":
                        selected["max"],

                    "std_metric":
                        selected["std"],

                    "metric_samples":
                        selected["n"],

                    "mean_tokens_s_gpu":
                        r["tokens"]["mean"],

                    "tokens_samples":
                        r["tokens"]["n"],

                    "mean_tflops_s_gpu":
                        r["tflops"]["mean"],

                    "tflops_samples":
                        r["tflops"]["n"],

                    "status":
                        r["status"],

                    "exp_name":
                        r["exp_name"],

                    "parameter_log":
                        r["parameter_log"],

                    "throughput_log":
                        r["throughput_log"],
                }
            )


# ============================================================
# Markdown
# ============================================================

def write_markdown(
    results,
    kind,
    comparison,
    output,
):

    label = metric_name(
        kind
    )


    lines = [
        "# Engram Performance Sweep",
        "",
        f"Primary metric: **{label}**",
        "",
        (
            f"Each result is the mean of the last "
            f"{LAST_N_ITERS} iterations of a "
            f"{TRAIN_ITERS}-iteration run."
        ),
        "",
        "## Baseline — PP × TP",
        "",
        (
            "| PP | TP | Params(B) | "
            f"{label} | Samples | Status |"
        ),
        "|---:|---:|---:|---:|---:|---|",
    ]


    for r in results:

        if r["mode"] != "baseline":
            continue


        s = get_metric_stats(
            r,
            kind,
        )


        lines.append(
            f"| {r['pp']} "
            f"| {r['actual_tp']} "
            f"| {fmt(r['parameters_billions'])} "
            f"| {fmt(s['mean'])} "
            f"| {s['n']} "
            f"| {r['status']} |"
        )


    lines.extend(
        [
            "",
            "## Our optimized — PP × Egp",
            "",
            (
                "| PP | Egp | TP | Params(B) | "
                f"{label} | Samples | Status |"
            ),
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )


    for r in results:

        if r["mode"] != "optimized":
            continue


        s = get_metric_stats(
            r,
            kind,
        )


        lines.append(
            f"| {r['pp']} "
            f"| {r['actual_egp']} "
            f"| {r['actual_tp']} "
            f"| {fmt(r['parameters_billions'])} "
            f"| {fmt(s['mean'])} "
            f"| {s['n']} "
            f"| {r['status']} |"
        )


    lines.extend(
        [
            "",
            "## Best comparison",
            "",
        ]
    )


    if comparison is None:

        lines.append(
            "Unable to make a valid comparison."
        )

    else:

        b = comparison["baseline"]
        o = comparison["optimized"]


        lines.extend(
            [
                (
                    f"- Baseline best: "
                    f"`PP={b['pp']}, TP={b['actual_tp']}`"
                ),
                (
                    f"- Baseline {label}: "
                    f"`{comparison['baseline_mean']:.2f}`"
                ),
                (
                    f"- Our optimized best: "
                    f"`PP={o['pp']}, Egp={o['actual_egp']}, "
                    f"TP={o['actual_tp']}`"
                ),
                (
                    f"- Our optimized {label}: "
                    f"`{comparison['optimized_mean']:.2f}`"
                ),
                (
                    f"- Improvement: "
                    f"`{comparison['improvement'] * 100:.2f}%`"
                ),
                "- Required improvement: `>= 10.00%`",
                (
                    "- Result: "
                    f"**{'SUCCESS' if comparison['success'] else 'FAIL'}**"
                ),
            ]
        )


    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Collect Engram PP/TP and PP/Egp "
            "performance sweep results."
        )
    )


    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Sweep output directory. "
            "This is the only required argument."
        ),
    )


    args = parser.parse_args()


    root = args.output_dir.resolve()


    if not root.is_dir():

        raise SystemExit(
            f"Directory not found: {root}"
        )


    experiments = find_experiments(
        root
    )


    if not experiments:

        raise SystemExit(
            f"No sweep experiments found under {root}"
        )


    print(
        f"Found {len(experiments)} experiments under:"
    )

    print(
        f"  {root}"
    )

    print()


    results = []


    for exp_dir in experiments:

        print(
            f"[PARSE] {exp_dir.name}"
        )


        result = parse_experiment(
            exp_dir
        )


        if result is None:
            continue


        results.append(result)


    results.sort(
        key=result_sort_key
    )


    # ========================================================
    # Choose ONE common metric for fair comparison.
    # ========================================================

    kind = select_metric(
        results
    )


    if kind is None:

        print()
        print(
            "ERROR: no common metric with 20 samples "
            "is available across successful runs."
        )

        print(
            "Need either Tokens/s/GPU or TFLOP/s/GPU "
            "for all successful experiments."
        )

        raise SystemExit(2)


    print()
    print(
        f"Selected common metric: "
        f"{metric_name(kind)}"
    )


    if kind == "tflops":

        print(
            "Tokens/s/GPU is not complete across the sweep; "
            "falling back to TFLOP/s/GPU for ALL configurations."
        )


    comparison = compare_best(
        results,
        kind,
    )


    print_grid(
        results,
        kind,
    )


    print_comparison(
        comparison,
        kind,
    )


    csv_path = (
        root
        / "results.csv"
    )

    md_path = (
        root
        / "results.md"
    )


    write_csv(
        results,
        kind,
        csv_path,
    )


    write_markdown(
        results,
        kind,
        comparison,
        md_path,
    )


    print()
    print(
        f"CSV      : {csv_path}"
    )

    print(
        f"Markdown : {md_path}"
    )


if __name__ == "__main__":
    main()
