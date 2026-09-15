#!/usr/bin/env python3
"""Terminate only processes carrying one sweep node's unique inherited token."""

import argparse
import os
import signal
import time
from pathlib import Path

TOKEN_ENV = "FLAGSCALE_ENGRAM_SWEEP_TOKEN"


def processes(token):
    expected = f"{TOKEN_ENV}={token}".encode()
    namespace = os.readlink("/proc/self/ns/pid")
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            # /proc can be mounted from an outer namespace. Signal only our namespace.
            if os.readlink(path / "ns/pid") != namespace:
                continue
            if expected not in (path / "environ").read_bytes().split(b"\0"):
                continue
            status = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines())
            if status["State"].strip().startswith("Z"):
                continue
            pid = int(status["NSpid"].split()[-1])
            if pid > 1 and pid != os.getpid():
                # Field 22 in /proc/PID/stat is process start time (the name may contain spaces).
                started = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
                matches.append((pid, path, started))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def send_signal(token, sig):
    expected = f"{TOKEN_ENV}={token}".encode()
    for pid, path, started in processes(token):
        try:
            # Prefer pidfds. Some training Python builds lack these bindings; in that
            # case recheck both the inherited token and start time immediately before kill.
            pidfds = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
            fd = os.pidfd_open(pid) if pidfds else None
            try:
                current_start = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
                if current_start == started and expected in (path / "environ").read_bytes().split(
                    b"\0"
                ):
                    if fd is not None:
                        signal.pidfd_send_signal(fd, sig)
                    else:
                        os.kill(pid, sig)
            finally:
                if fd is not None:
                    os.close(fd)
        except (FileNotFoundError, ProcessLookupError):
            pass


def cleanup(token, grace=5.0):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + grace
        # Rescan: children may fork or start a new session during cancellation.
        while True:
            send_signal(token, sig)
            if not processes(token):
                return 0
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token")
    args = parser.parse_args()
    raise SystemExit(cleanup(args.token))
