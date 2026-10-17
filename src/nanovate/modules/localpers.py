"""Region-local persistence (H0, H1) by union-find; no CAD/image splice (AGENTS.md §5.2, §7.5).

Filtration F = 1 - P on pixels (P = conductor probability in [0, 1]); conductor is the
sublevel set, 4-connected, dielectric 8-connected -- the same convention as the GUDHI
vertex construction in ``topology.py`` (pinned by tests that compare the two).

  H0: sublevel union-find (4-neighbours), elder rule. One pair per merged component:
      (birth F, death F). The one component that never dies is the "essential" one.
  H1: dual union-find on the superlevel set (descending F, 8-neighbours) with an
      "outside" node joined to every border pixel (so only enclosed holes survive).
      A hole born at F_hi and merged into the outside at F_lo is the sublevel H1
      interval (birth = F_lo, death = F_hi).

Localisation: a feature is *flagged* if its component, at the moment it dies, contains a
"core" pixel inside ``region``: a conductor-core pixel (F < core) for H0 and a dielectric-core
pixel (F > core) for H1. Only flagged features are counted, which mirrors
``topo_utils.betti_local`` (components / holes that intersect the CAD-derived region) and
needs no image/CAD splice, so there is no seam. Pixels with 0.5-ish values never flag, so a
feature whose peak is dimmer than ``core`` cannot be counted (documented limitation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class LocalPersistence:
    """Finite pairs (birth, death, flagged) for H0/H1 plus the essential H0 component.

    ``h0`` / ``h1``: float arrays (n, 2) of (birth, death) in F = 1 - P units;
    ``h0_flag`` / ``h1_flag``: bool arrays (n,). ``ess_flag_level`` is the filtration level at
    which the essential component's flag first became true (inf if never).
    """

    h0: np.ndarray
    h0_flag: np.ndarray
    h1: np.ndarray
    h1_flag: np.ndarray
    ess_flag_level: float

    def betti(self, tau: float) -> tuple[int, int]:
        """(beta_0, beta_1) of features with lifetime >= tau whose component meets the region."""
        life0 = self.h0[:, 1] - self.h0[:, 0]
        pers0 = life0 >= tau
        b0 = int(np.count_nonzero(pers0 & self.h0_flag))
        # essential component: flag evaluated just before the earliest persistent death
        d_first = float(self.h0[pers0, 1].min()) if pers0.any() else np.inf
        if self.ess_flag_level < d_first or (not pers0.any() and np.isfinite(self.ess_flag_level)):
            b0 += 1
        b1 = int(np.count_nonzero((self.h1[:, 1] - self.h1[:, 0] >= tau) & self.h1_flag))
        return b0, b1

    def diagrams(self) -> tuple[np.ndarray, np.ndarray]:
        """Flagged finite diagrams (H0, H1) for the bottleneck distance."""
        return self.h0[self.h0_flag], self.h1[self.h1_flag]


def _find(parent: list[int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _h0(F: np.ndarray, reg: np.ndarray, core: float) -> tuple[np.ndarray, np.ndarray, float]:
    H, W = F.shape
    n = H * W
    f = F.ravel().tolist()
    r = reg.ravel().tolist()
    order = np.argsort(F.ravel(), kind="stable").tolist()
    rank = [0] * n
    for k, v in enumerate(order):
        rank[v] = k
    parent = list(range(n))
    present = bytearray(n)
    flag = bytearray(n)
    birth = f  # birth F of the component whose origin pixel is the root key
    origin = list(range(n))  # origin pixel of each root (elder rule key = (F, rank))
    pairs: list[tuple[float, float, int]] = []
    first = order[0]
    ess_level = float("inf")
    for v in order:
        present[v] = 1
        fv = f[v]
        if fv < core and r[v]:
            flag[v] = 1
            if v == first:
                ess_level = fv
        y0, x0 = divmod(v, W)
        for dy, dx in _N4:
            yy, xx = y0 + dy, x0 + dx
            if 0 <= yy < H and 0 <= xx < W:
                u = yy * W + xx
                if not present[u]:
                    continue
                a, b = _find(parent, v), _find(parent, u)
                if a == b:
                    continue
                oa, ob = origin[a], origin[b]
                if (f[oa], rank[oa]) <= (f[ob], rank[ob]):
                    elder, young = a, b
                else:
                    elder, young = b, a
                oy = origin[young]
                if fv > birth[oy]:
                    pairs.append((birth[oy], fv, flag[young]))
                parent[young] = elder
                if flag[young] and not flag[elder]:
                    flag[elder] = 1
                    if _find(parent, first) == elder:
                        ess_level = fv
    arr = np.array(pairs, dtype=float).reshape(-1, 3)
    return arr[:, :2], arr[:, 2].astype(bool), ess_level


def _h1(F: np.ndarray, reg: np.ndarray, core: float) -> tuple[np.ndarray, np.ndarray]:
    H, W = F.shape
    n = H * W
    out = n  # outside node: birth +inf, always the elder
    f = F.ravel().tolist()
    r = reg.ravel().tolist()
    order = np.argsort(-F.ravel(), kind="stable").tolist()
    rank = [0] * n
    for k, v in enumerate(order):
        rank[v] = k
    parent = list(range(n + 1))
    present = bytearray(n + 1)
    present[out] = 1
    flag = bytearray(n + 1)
    origin = list(range(n + 1))
    pairs: list[tuple[float, float, int]] = []

    def union(v: int, u: int, level: float) -> None:
        a, b = _find(parent, v), _find(parent, u)
        if a == b:
            return
        if a == out or b == out:
            elder, young = (a, b) if a == out else (b, a)
        else:
            oa, ob = origin[a], origin[b]
            if (-f[oa], rank[oa]) <= (-f[ob], rank[ob]):
                elder, young = a, b
            else:
                elder, young = b, a
        if young != out:
            hi = f[origin[young]]
            if hi > level:
                pairs.append((level, hi, flag[young]))
        parent[young] = elder
        if flag[young]:
            flag[elder] = 1

    for v in order:
        present[v] = 1
        fv = f[v]
        if fv > core and r[v]:
            flag[v] = 1
        y0, x0 = divmod(v, W)
        if y0 == 0 or y0 == H - 1 or x0 == 0 or x0 == W - 1:
            union(v, out, fv)
        for dy, dx in _N8:
            yy, xx = y0 + dy, x0 + dx
            if 0 <= yy < H and 0 <= xx < W:
                u = yy * W + xx
                if present[u]:
                    union(v, u, fv)
    arr = np.array(pairs, dtype=float).reshape(-1, 3)
    return arr[:, :2], arr[:, 2].astype(bool)


def local_persistence(field: np.ndarray, region: np.ndarray, core: float = 0.5) -> LocalPersistence:
    """Region-local H0/H1 persistence of the conductor-probability ``field`` in [0, 1].

    Args:
        field: 2D conductor probability P (1 = conductor).
        region: 2D bool, same shape; features count only if they reach it
            (use ``topo_utils.relevance_region(cad, win)``).
        core: F-level separating conductor-core (F < core) from dielectric-core (F > core)
            pixels; only core pixels inside ``region`` flag a component.
    """
    F = np.clip(1.0 - np.asarray(field, dtype=float), 0.0, 1.0)
    reg = np.asarray(region, dtype=bool)
    h0, f0, ess = _h0(F, reg, core)
    h1, f1 = _h1(F, reg, core)
    return LocalPersistence(h0, f0, h1, f1, ess)
