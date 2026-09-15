#!/usr/bin/env python3
"""Run one Engram experiment and supervise every node (shared filesystem, SSH)."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from collect_engram_sweep import summarize
from cleanup_engram_sweep import TOKEN_ENV


def compose_config(args):
    with initialize_config_dir(version_base=None, config_dir=str(args.config_path.resolve())):
        return compose(config_name=args.config_name, overrides=args.overrides)


def validate(config, warmup):
    from flagscale.runner.utils import parse_hostfile, update_nodes_envs

    runner, system, model = config.experiment.runner, config.train.system, config.train.model
    if config.experiment.task.type != "train" or config.experiment.task.backend != "megatron":
        raise ValueError("Sweep requires Megatron training")
    if (
        runner.get("type", "ssh") != "ssh"
        or runner.get("no_shared_fs", False)
        or runner.get("per_node_task", False)
    ):
        raise ValueError("Sweep requires SSH, shared filesystem and per_node_task=false")
    if runner.get("heartbeat", {}).get("enabled", False) or runner.get("enable_monitoring", False):
        raise ValueError("Disable heartbeat/enable_monitoring for the sweep supervisor")
    nnodes, nproc = int(runner.nnodes), int(runner.nproc_per_node)
    if nnodes < 1 or nproc < 1 or not 0 < int(runner.master_port) < 65536:
        raise ValueError("Invalid node count, GPU count or master port")
    hostfile = runner.get("hostfile")
    if hostfile and not Path(hostfile).is_file():
        raise ValueError(f"Missing hostfile: {hostfile}; refusing single-node fallback")
    resources = parse_hostfile(hostfile) if hostfile else None
    if not resources:
        if nnodes != 1:
            raise ValueError("Multi-node sweep requires a hostfile")
        resources = {"localhost": {"slots": nproc, "type": None}}
    if len(resources) < nnodes:
        raise ValueError("Hostfile has fewer nodes than requested")
    nodes = list(resources.items())[:nnodes]
    for host, resource in nodes:
        envs = update_nodes_envs(config.experiment.get("envs", {}), host, resource)
        visible = envs.get("CUDA_VISIBLE_DEVICES")
        if resource["slots"] < nproc or (
            visible is not None and len(str(visible).split(",")) < nproc
        ):
            raise ValueError(f"Insufficient GPU slots on {host}")
    world = nnodes * nproc
    tp, pp, cp = (
        int(system.tensor_model_parallel_size),
        int(system.pipeline_model_parallel_size),
        int(system.get("context_parallel_size", 1)),
    )
    ep, etp = int(system.get("expert_model_parallel_size", 1)), int(
        system.get("expert_tensor_parallel_size", tp)
    )
    method = model.engram_embedding_parallel_method
    if method not in {"allreduce", "alltoall"}:
        raise ValueError("Sweep supports allreduce and alltoall")
    if method == "alltoall" and not system.get("engram_embedding_parallel_size"):
        raise ValueError("alltoall requires engram_embedding_parallel_size")
    egp = int(system.get("engram_embedding_parallel_size") or 1) if method == "alltoall" else 1
    for divisor in (tp * pp * cp, ep * etp * pp, egp * pp):
        if divisor <= 0 or world % divisor:
            raise ValueError(
                f"World size {world} is not divisible by parallel group size {divisor}"
            )
    dp = world // (tp * pp * cp)
    if model.global_batch_size % (model.micro_batch_size * dp):
        raise ValueError("Global batch size must be divisible by micro batch size × DP")
    first, last = system.get("decoder_first_pipeline_num_layers"), system.get(
        "decoder_last_pipeline_num_layers"
    )
    remaining = int(model.num_layers) - int(first or 0) - int(last or 0)
    middle = pp - int(first is not None) - int(last is not None)
    if remaining < 0 or (middle and remaining % middle) or (not middle and remaining):
        raise ValueError("Invalid pipeline layer partition")
    if not 0 <= warmup < int(model.train_iters):
        raise ValueError("Require 0 <= WARMUP_ITERS < TRAIN_ITERS")
    if int(model.get("num_experts", ep)) % ep:
        raise ValueError("Number of experts must be divisible by EP")
    OmegaConf.to_container(config, resolve=True)
    return nodes


def node_command(host, port, command):
    if host == "localhost":
        return ["bash", "-c", command]
    return [
        "ssh",
        "-n",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
        host,
        command,
    ]


def write_node_scripts(directory, rank, training_script):
    """A launch lock and cancellation marker also protect against delayed SSH startup."""
    prefix = directory / f"node_{rank}"

    def q(suffix):
        return shlex.quote(str(prefix) + suffix)

    wrapper, cleanup = Path(str(prefix) + ".sh"), Path(str(prefix) + ".cleanup.sh")
    cleanup_command = shlex.join(
        [sys.executable, str(Path(__file__).with_name("cleanup_engram_sweep.py")), str(wrapper)]
    )
    wrapper.write_text(
        "#!/bin/bash\nset -u\n"
        f"exec 9>{q('.lock')}\nflock -x 9 || exit 125\n"
        f"[[ ! -e {q('.cancel')} ]] || exit 125\n"
        f"echo $$ > {q('.pgid')}\nflock -u 9\nexec 9>&-\n"
        f"bash {shlex.quote(str(training_script))}\nrc=$?\n"
        f"echo $rc > {q('.exit_code')}\nexit $rc\n"
    )
    cleanup.write_text(
        "#!/bin/bash\nset -eu\n"
        f"exec 9>{q('.lock')}\nflock -w 10 -x 9\ntouch {q('.cancel')}\n"
        f"{cleanup_command}\n"
    )
    return wrapper, cleanup


def supervise(nodes, port, timeout):
    """All nodes run concurrently; any failed node ends the experiment, then all are cleaned."""
    processes, logs = [], []
    result = {"status": "failed", "exit_code": 1, "cleanup_ok": True}
    previous = {}

    def interrupted(signum, _frame):
        raise InterruptedError(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, interrupted)
    try:
        for host, wrapper, _ in nodes:
            # Read-only remote checks before starting any training.
            python_check = shlex.join(
                [
                    sys.executable,
                    "-c",
                    "import os; os.kill(os.getpid(),0); open('/proc/self/environ','rb').read(); open('/proc/self/status').read()",
                ]
            )
            command = (
                f"command -v bash setsid flock >/dev/null && test -r {shlex.quote(str(wrapper))}"
                f" && {python_check}"
            )
            subprocess.run(node_command(host, port, command), check=True, timeout=30)
        start = time.monotonic()
        for host, wrapper, _ in nodes:
            stream = wrapper.with_suffix(".log").open("w")
            logs.append(stream)
            command = shlex.join(
                ["exec", "env", f"{TOKEN_ENV}={wrapper}", "setsid", "--wait", "bash", str(wrapper)]
            )
            processes.append(
                subprocess.Popen(
                    node_command(host, port, command),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                break
            if all(code == 0 for code in codes):
                result.update(status="success", exit_code=0)
                break
            if time.monotonic() - start >= timeout:
                result.update(status="timeout", exit_code=124)
                break
            time.sleep(0.2)
    except InterruptedError as error:
        result.update(status="interrupted", exit_code=128 + int(error.args[0]))
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        # Do not let a second interrupt skip remote cleanup.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        for host, _, cleanup in nodes:
            try:
                with cleanup.with_suffix(".log").open("w") as cleanup_log:
                    subprocess.run(
                        node_command(host, port, "bash " + shlex.quote(str(cleanup))),
                        check=True,
                        timeout=35,
                        stdout=cleanup_log,
                        stderr=subprocess.STDOUT,
                    )
            except (OSError, subprocess.SubprocessError) as error:
                result["cleanup_ok"] = False
                result["error"] = f"Cannot confirm cleanup on {host}: {error}"
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result["node_exit_codes"] = [process.returncode for process in processes]
        for stream in logs:
            stream.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not result["cleanup_ok"]:
            result.update(status="cleanup_failed", exit_code=125)
    return result


def run(args):
    config = compose_config(args)
    resources = validate(config, args.warmup_iters)
    if args.check_only:
        print("Configuration preflight passed")
        return 0
    exp_dir = Path(config.experiment.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=False)
    directory = exp_dir / "sweep_nodes"
    directory.mkdir()
    system, model = config.train.system, config.train.model
    world = int(config.experiment.runner.nnodes) * int(config.experiment.runner.nproc_per_node)
    result = dict(
        experiment=config.experiment.exp_name,
        status="preparation_failed",
        exit_code=1,
        cleanup_ok=True,
        pp=system.pipeline_model_parallel_size,
        tp=system.tensor_model_parallel_size,
        egp=system.get("engram_embedding_parallel_size", ""),
        method=model.engram_embedding_parallel_method,
        world_size=world,
        dp=world
        // (
            system.pipeline_model_parallel_size
            * system.tensor_model_parallel_size
            * system.get("context_parallel_size", 1)
        ),
        train_iters=model.train_iters,
        warmup_iters=args.warmup_iters,
        global_batch_size=model.global_batch_size,
        seq_length=model.seq_length,
    )
    try:
        from flagscale.run import get_runner
        from flagscale.runner.utils import update_nodes_envs

        runner = get_runner(config, "train")
        launcher = getattr(runner, "launcher", runner)
        nodes = []
        for rank, (host, resource) in enumerate(resources):
            launcher._run_each(
                host,
                config.experiment.runner.get("master_addr", resources[0][0]),
                config.experiment.runner.master_port,
                len(resources),
                rank,
                config.experiment.runner.nproc_per_node,
                device_type=resource["type"],
                background=False,
                dryrun=True,
                cur_envs=update_nodes_envs(config.experiment.get("envs", {}), host, resource),
            )
            script = Path(config.train.system.logging.scripts_dir) / f"host_{rank}_{host}_run.sh"
            wrapper, cleanup = write_node_scripts(directory, rank, script)
            nodes.append((host, wrapper, cleanup))
        OmegaConf.save(config, exp_dir / "resolved_config.yaml", resolve=True)
        if args.dry_run:
            result.update(status="dryrun", exit_code=0)
        else:
            result.update(
                supervise(nodes, config.experiment.runner.get("ssh_port", 22), args.timeout)
            )
            summarize(exp_dir, result)
    except Exception as error:
        result["error"] = str(error)
        result["exit_code"] = 1
        if result["status"] == "success":
            result["status"] = "failed"
        raise
    finally:
        (exp_dir / "sweep_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    arguments = parser.parse_args()
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    raise SystemExit(run(arguments))
