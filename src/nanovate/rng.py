"""Deterministic seeding: one config seed drives every random draw.

Each independent unit (layout, die, sample, worker) gets its own child generator
derived from (seed, *keys), so results do not depend on worker count or order.
"""
from __future__ import annotations

import zlib

import numpy as np


def _key_to_int(k: object) -> int:
    if isinstance(k, (int, np.integer)):
        return int(k) & 0xFFFFFFFF
    return zlib.crc32(str(k).encode())


def child_rng(seed: int, *keys: object) -> np.random.Generator:
    """Return a numpy Generator uniquely determined by ``seed`` and ``keys``."""
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key_to_int(k) for k in keys))
    return np.random.default_rng(ss)


def child_seed(seed: int, *keys: object) -> int:
    """Return a 32-bit integer seed (e.g. for torch) derived from ``seed`` and ``keys``."""
    return int(child_rng(seed, *keys).integers(0, 2**31 - 1))
