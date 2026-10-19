"""FITTED parametric reference for periodic hole arrays (Track C Rung 1). NOT true CAD.

A hole array is modelled as a 2D lattice of circular holes (dielectric) in a conductor film:
centres o + i*a1 + j*a2 (i, j integer), one diameter d. Lengths in px of the image they are fitted on,
or in nm where the name says so. A fit can absorb a real defect (e.g. a global size change); a lattice of
equal circles cannot represent a missing, merged or extra hole, which is why this model is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


@dataclass(frozen=True)
class Lattice:
    """Hole lattice in px: basis vectors a1, a2 (row, col), origin o (row, col), diameter d."""

    a1: tuple[float, float]
    a2: tuple[float, float]
    o: tuple[float, float]
    d: float

    def scaled(self, k: float) -> "Lattice":
        """Same lattice in a frame whose px are k times smaller (k = old_px_nm / new_px_nm)."""
        f = lambda v: (v[0] * k, v[1] * k)  # noqa: E731
        return Lattice(f(self.a1), f(self.a2), f(self.o), self.d * k)


def detect_holes(img: np.ndarray, smooth: float = 2.0) -> list:
    """Dark, roughly round blobs not touching the border; returns skimage regionprops."""
    sm = ndi.gaussian_filter(np.asarray(img, float), smooth)
    lab = label(sm < threshold_otsu(sm))
    rp = [r for r in regionprops(lab) if r.area >= 20]
    H, W = img.shape
    rp = [r for r in rp if r.bbox[0] > 0 and r.bbox[1] > 0 and r.bbox[2] < H and r.bbox[3] < W]
    if not rp:
        return []
    ref = np.percentile([r.area for r in rp], 90)     # the lattice holes are the large blobs; pits / debris are small
    return [r for r in rp if 0.4 * ref <= r.area <= 2.5 * ref and r.eccentricity < 0.75]


def _basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two shortest non-collinear neighbour vectors among hole centres (row, col)."""
    D = c[None] - c[:, None]
    L = np.hypot(D[..., 0], D[..., 1])
    np.fill_diagonal(L, np.inf)
    v = D[np.arange(len(c)), L.argmin(1)]
    v = np.where((v[:, 1] < 0)[:, None] | ((v[:, 1] == 0) & (v[:, 0] < 0))[:, None], -v, v)   # canonical sign
    a1 = np.median(v[np.hypot(*v.T) < 1.3 * np.median(np.hypot(*v.T))], axis=0)
    cand = D.reshape(-1, 2)
    n = np.hypot(*cand.T)
    ok = (n > 0.5 * np.hypot(*a1)) & (np.abs(np.cross(cand, a1)) / (n * np.hypot(*a1) + 1e-9) > 0.5)
    cand = cand[ok]; n = n[ok]
    a2 = cand[np.argsort(n)[: max(3, len(n) // 50)]]
    a2 = np.where((a2[:, 0] < 0)[:, None], -a2, a2)
    return a1, np.median(a2, axis=0)


def fit_lattice(centres: np.ndarray, diam: float, a1: np.ndarray | None = None, a2: np.ndarray | None = None) -> tuple[Lattice, np.ndarray]:
    """Least-squares lattice through hole centres; basis estimated unless given. Returns (lattice, residual px per hole)."""
    c = np.asarray(centres, float)
    if a1 is None or a2 is None:
        a1, a2 = _basis(c)
        fix = False
    else:
        fix = True
    A = np.stack([a1, a2], 1)
    ij = np.rint(np.linalg.solve(A, (c - c[0]).T).T)
    if fix:
        o = np.mean(c - ij @ A.T, axis=0)
    else:
        X = np.c_[np.ones(len(c)), ij]
        coef, *_ = np.linalg.lstsq(X, c, rcond=None)
        o, a1, a2 = coef[0], coef[1], coef[2]
        A = np.stack([a1, a2], 1)
    res = np.hypot(*(c - (o + ij @ A.T)).T)
    return Lattice(tuple(map(float, a1)), tuple(map(float, a2)), tuple(map(float, o)), float(diam)), res


def render(lat: Lattice, shape: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Binary conductor mask (True = film) of ``shape``; ``offset`` = (row, col) of the crop's corner in the lattice frame."""
    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    rr += offset[0]; cc += offset[1]
    A = np.array([lat.a1, lat.a2]).T
    Ai = np.linalg.inv(A)
    q = Ai @ np.stack([rr.ravel() - lat.o[0], cc.ravel() - lat.o[1]])
    fr = q - np.rint(q)
    best = np.full(H * W, np.inf)
    for di in (-1, 0, 1):          # nearest lattice point (non-orthogonal lattices need neighbours)
        for dj in (-1, 0, 1):
            p = A @ (fr - np.array([[di], [dj]]))
            best = np.minimum(best, np.hypot(p[0], p[1]))
    return (best > lat.d / 2).reshape(H, W)
