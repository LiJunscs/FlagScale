# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Check the real launcher/process group before any conversion side effects."""

import os
import sys


def require_world_one(environ=None, distributed=None):
    env = os.environ if environ is None else environ
    sizes = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
             "PMIX_SIZE", "SLURM_NTASKS")
    ranks = ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
             "PMIX_RANK", "SLURM_PROCID")
    for name, expected in [(key, 1) for key in sizes] + [(key, 0) for key in ranks]:
        if name in env:
            try:
                value = int(env[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid launcher setting {name}={env[name]!r}") from error
            if value != expected:
                raise ValueError(f"DSv4 conversion requires real world_size=1/rank=0; {name}={value}")
    if distributed is None:
        torch = sys.modules.get("torch")
        distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        if distributed.get_world_size() != 1 or distributed.get_rank() != 0:
            raise ValueError("DSv4 conversion refuses an initialized multi-rank process group")
