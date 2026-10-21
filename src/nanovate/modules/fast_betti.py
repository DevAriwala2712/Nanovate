"""B2 fast Betti path (NEXT_PATHS Track B2): Delta-beta at ONE field level without persistent homology. OFF BY DEFAULT.

beta0 = 4-connected conductor components (one labelling pass); beta1 = beta0 - chi, with the Euler characteristic chi
from local 2 x 2 pattern counts (``skimage.measure.euler_number``, connectivity 1 = conductor 4-, dielectric 8-connected,
the same convention as GUDHI's vertex cubical complex and ``topo_utils.betti_binary``).

Same splices as Module 4's v2 path: H0 on the field spliced with the CAD outside the relevance region, H1 on the field
filled with dielectric outside it; the reference is the binary CAD under the same splices.

Caveat that decides whether this can ever be exact: Module 4 counts persistence bars with lifetime >= tau over ALL levels,
not the Betti numbers of a single superlevel set. On a binary field the two coincide (tested); on a continuous field
they need not. ``experiments/b2_fast_betti.py`` measures how often they differ on Manhattan val. Nothing calls this
module from the pipeline.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import euler_number, label

from ..topo_utils import betti_binary


def betti_at_level(field: np.ndarray, level: float) -> tuple[int, int]:
    """(beta0, beta1) of the superlevel set {field >= level} (conductor 4-connected, holes 8-connected)."""
    m = np.asarray(field) >= level
    b0 = int(label(m, connectivity=1).max())
    return b0, b0 - int(euler_number(m, connectivity=1))


def fast_delta_beta(field: np.ndarray, cad: np.ndarray, reg: np.ndarray, level: float) -> tuple[int, int]:
    """Delta-beta at one level with Module 4's v2 splices (H0: CAD outside ``reg``; H1: dielectric outside ``reg``)."""
    c = cad.astype(bool)
    b0 = betti_at_level(np.where(reg, field, c.astype(float)), level)[0]
    b1 = betti_at_level(np.where(reg, field, 0.0), level)[1]
    r0 = betti_binary(c)[0]
    r1 = betti_binary(reg & c)[1]
    return b0 - r0, b1 - r1
