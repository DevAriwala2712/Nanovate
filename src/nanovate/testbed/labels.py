"""Labels from exact topology of the ground-truth wafer W vs CAD (never from intent).

A sample is accepted only if the topology class of W agrees with the injection intent:
  nuisance, notch, filament -> 'none'   (topology unchanged; label NUISANCE / NEAR_MISS)
  bridge                    -> 'bridge' (label KILLER, subclass bridge)
  open, particle            -> 'open'   (label KILLER, subclass open/particle)
Otherwise the sample is resampled and the resample is counted (AGENTS.md §7.6).
"""
from __future__ import annotations

import numpy as np

from ..topo_utils import Window, delta_betti, topo_class

EXPECTED = {
    "nuisance": "none",
    "notch": "none",
    "filament": "none",
    "bridge": "bridge",
    "open": "open",
    "particle": "open",
}

LABEL_OF_KIND = {
    "nuisance": "NUISANCE",
    "notch": "NEAR_MISS",
    "filament": "NEAR_MISS",
    "bridge": "KILLER",
    "open": "KILLER",
    "particle": "KILLER",
}


def topology_label(W_tile: np.ndarray, cad_tile: np.ndarray, win: Window) -> dict:
    """Exact Betti difference and topology class of W vs CAD on the central window."""
    d = delta_betti(W_tile, cad_tile, win)
    return {"d_beta0": d[0], "d_beta1": d[1], "topo": topo_class(d)}


# allowed total |d_beta0| + |d_beta1| per killer kind: one defect = one topology event,
# except a wide bridge capsule, which may short up to 2 neighbouring gaps.
MAX_EVENTS = {"bridge": 2, "open": 1, "particle": 1}


def consistent(kind: str, topo: dict) -> bool:
    """True if the measured topology class (and event count) matches ``kind``."""
    if EXPECTED[kind] != topo["topo"]:
        return False
    n = abs(topo["d_beta0"]) + abs(topo["d_beta1"])
    return n <= MAX_EVENTS[kind] if kind in MAX_EVENTS else True
