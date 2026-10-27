# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Generated tensor regions; no per-model shape inventory."""
from dataclasses import asdict, dataclass, field

@dataclass(frozen=True)
class Piece:
    hf: str
    runtime_key: str
    key: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    global_shape: tuple[int, ...]
    hf_index: int | None = None
    extra_key: str | None = None
    source_offset: tuple[int, ...] | None = None
    source_shape: tuple[int, ...] | None = None
    axes: tuple[int, ...] | None = None

@dataclass
class Profile:
    pieces: list[Piece]
    shapes: dict[str, tuple[int, ...]]

    name: str = "mapping"
    revision: str = "hf-mcore-v1"
    extra_keys: tuple[str, ...] = ()
    physical_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def keys(self):
        return {piece.key for piece in self.pieces}

    def report(self):
        return {"profile": self.name, "revision": self.revision,
                "physical_contract": {"non_homogeneous_layers": True,
                                      "singleton_local_shards": False},
                "pieces": [asdict(piece) for piece in self.pieces]}
