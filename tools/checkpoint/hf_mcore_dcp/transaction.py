# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Commit a conversion only after all writes and checks succeed."""
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from . import MappingError

@contextmanager
def _staged_output(path, overwrite, inputs):
    output = Path(path).expanduser().resolve()
    for item in inputs:
        source = Path(item).expanduser().resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise MappingError(f"Output overlaps an input: {output} / {source}")
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise MappingError(f"Output is not an empty directory: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        yield stage
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
