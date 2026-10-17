"""CAD -> wafer W perturbations: nuisance (LER, grain) and defects.

All functions work on the padded canvas (see layouts.py). Defects are placed only inside
the *placement region* = central comparison window shrunk by a buffer, so they never sit
in the boundary margin. Every defect returns (W_new, mask, params); W is never trusted
for labelling by intent alone: labels.py re-derives topology from W.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import lfilter
from scipy.spatial import cKDTree
from skimage.measure import label as sk_label

from ..config import DefectCfg, LayoutCfg, NuisanceCfg
from .layouts import rasterize

_C4 = ndi.generate_binary_structure(2, 1)


# --------------------------------------------------------------------------- nuisance
def _resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed polygon at ~``step`` nm arc-length spacing (corners preserved)."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        L = float(np.hypot(*(b - a)))
        k = max(1, int(np.ceil(L / step)))
        t = np.arange(k)[:, None] / k
        out.append(a + t * (b - a))
    return np.concatenate(out, 0)


def ou_process(
    n: int, step: float, sigma: float, xi: float, rng: np.random.Generator, periodic: bool = False
) -> np.ndarray:
    """Stationary Ornstein-Uhlenbeck samples (nm) at arc-length spacing ``step``.

    Open: <e(s) e(s')> = sigma^2 exp(-|s - s'| / xi) (exact AR(1) recursion).
    Periodic (closed contour of perimeter P = n * step): the OU process on a circle,
    <e e'> = sigma^2 cosh((P/2 - |ds|)/xi) / cosh(P/(2 xi)) with |ds| the arc distance,
    sampled exactly by circulant embedding. Needed so a contour has no jump where it closes.
    """
    if sigma <= 0:
        return np.zeros(n)
    if periodic:
        P = n * step
        ds = np.minimum(np.arange(n), n - np.arange(n)) * step
        c = sigma**2 * np.cosh((P / 2 - ds) / xi) / np.cosh(P / (2 * xi))
        lam = np.clip(np.fft.fft(c).real, 0, None)
        z = rng.standard_normal(n)
        return np.fft.ifft(np.sqrt(lam) * np.fft.fft(z)).real
    rho = np.exp(-step / xi)
    z = rng.standard_normal(n)
    z[0] /= np.sqrt(1 - rho**2)  # stationary start
    return lfilter([np.sqrt(1 - rho**2) * sigma], [1.0, -rho], z)


def _union_boundary_points(polygons: list[np.ndarray], step: float, rng: np.random.Generator, cfg: NuisanceCfg):
    """Resampled contour points of every polygon that lie on the *union* boundary, with
    their OU outward displacements (one independent OU process per polygon contour)."""
    from matplotlib.path import Path

    pts_all, eta_all = [], []
    paths = [Path(p) for p in polygons]
    lo = np.array([p.min(0) for p in polygons])
    hi = np.array([p.max(0) for p in polygons])
    for i, poly in enumerate(polygons):
        q = _resample_closed(poly, step)
        area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
        t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
        nrm = np.stack([t[:, 1], -t[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        if area < 0:
            nrm = -nrm
        eta = ou_process(len(q), step, cfg.ler_sigma, cfg.ler_xi, rng, periodic=True)
        if cfg.ler_aa > 0 and len(q) > 4:
            # anti-alias: sub-pixel OU roughness cannot be represented on a 1 nm raster and
            # otherwise produces single-pixel islands; rescale to keep the edge std = sigma
            eta = ndi.gaussian_filter1d(eta, cfg.ler_aa / step, mode="wrap") / np.sqrt(_aa_gain(step, cfg))
        probe = q + 0.3 * nrm  # just outside this polygon
        keep = np.ones(len(q), bool)
        for j in np.nonzero(np.all(lo <= hi[i] + 1, 1) & np.all(hi >= lo[i] - 1, 1))[0]:
            if j != i:
                keep &= ~paths[j].contains_points(probe)
        pts_all.append(q[keep])
        eta_all.append(eta[keep])
    return np.concatenate(pts_all), np.concatenate(eta_all)


def _aa_gain(step: float, cfg: NuisanceCfg) -> float:
    """Variance factor of a unit-variance OU after the arc-length anti-alias Gaussian."""
    h = int(np.ceil(5 * cfg.ler_aa / step))
    lag = np.arange(-h, h + 1)
    kern = np.exp(-0.5 * (lag * step / cfg.ler_aa) ** 2)
    kern /= kern.sum()
    acf = np.exp(-step / cfg.ler_xi) ** np.abs(lag[:, None] - lag[None, :])
    return float(kern @ acf @ kern)


def remove_specks(W: np.ndarray, cad: np.ndarray, max_px: int) -> np.ndarray:
    """Drop conductor islands and dielectric pinholes of <= max_px pixels that are absent
    from CAD (unphysical raster specks created by LER on a 1 nm grid)."""
    if max_px <= 0:
        return W
    W = W.copy()
    for target, val in ((W, False), (~W, True)):
        lab = sk_label(target, connectivity=1 if val is False else 2)
        sizes = np.bincount(lab.ravel())
        small = np.nonzero(sizes <= max_px)[0]
        small = small[small > 0]
        if small.size:
            m = np.isin(lab, small)
            W[m] = val
    return W


def apply_ler(polygons: list[np.ndarray], N: int, cfg: NuisanceCfg, rng: np.random.Generator) -> np.ndarray:
    """Line-edge roughness as a level set (no self-intersections at corners).

    Each union-boundary point carries an OU outward displacement eta(s) (nm). A pixel x
    near the boundary takes the displacement of its nearest boundary point s*(x), and
    W = {phi(x) < eta(s*(x))} with phi the signed distance to CAD (negative inside).
    Far from any edge (> 4 sigma + 3 nm) W equals CAD.
    """
    cad = rasterize(polygons, N)
    if cfg.ler_sigma <= 0:
        return cad
    pts, eta = _union_boundary_points(polygons, cfg.ler_step, rng, cfg)
    band_w = 4 * cfg.ler_sigma + 3
    near = (ndi.distance_transform_edt(cad) <= band_w + 1) & (ndi.distance_transform_edt(~cad) <= band_w + 1)
    near |= ndi.binary_dilation(cad ^ ndi.binary_erosion(cad), iterations=int(band_w) + 1)
    rr, cc = np.nonzero(near)
    d, k = cKDTree(pts).query(np.stack([cc, rr], 1).astype(float))
    phi = np.where(cad[rr, cc], -d, d)
    W = cad.copy()
    W[rr, cc] = phi < eta[k]
    return remove_specks(W, cad, cfg.min_speck)


def grain_field(N: int, cfg: NuisanceCfg, rng: np.random.Generator) -> np.ndarray:
    """Voronoi grain offsets ~ U(-a, a) per cell (fraction of conductor contrast), (N, N)."""
    if cfg.grain_contrast <= 0:
        return np.zeros((N, N), np.float32)
    n_seeds = max(1, int(rng.poisson(N * N / cfg.grain_size**2)))
    seeds = rng.uniform(0, N, size=(n_seeds, 2))
    offs = rng.uniform(-cfg.grain_contrast, cfg.grain_contrast, n_seeds)
    rr, cc = np.mgrid[0:N, 0:N]
    _, idx = cKDTree(seeds).query(np.stack([cc.ravel(), rr.ravel()], 1))
    return offs[idx].reshape(N, N).astype(np.float32)


# --------------------------------------------------------------------------- helpers
def placement_region(lcfg: LayoutCfg, buf: float) -> tuple[int, int]:
    """[lo, hi) canvas index range (both axes) where defect centres may be placed."""
    lo = int(lcfg.pad + lcfg.margin + np.ceil(buf))
    hi = int(lcfg.pad + lcfg.tile - lcfg.margin - np.ceil(buf))
    return lo, hi


def _in_region(rc: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return (rc[:, 0] >= lo) & (rc[:, 0] < hi) & (rc[:, 1] >= lo) & (rc[:, 1] < hi)


def capsule(N: int, p: np.ndarray, q: np.ndarray, r: float) -> np.ndarray:
    """Pixels within distance ``r`` of segment p-q (p, q as (row, col))."""
    lo = np.floor(np.minimum(p, q) - r - 1).astype(int).clip(0, N)
    hi = np.ceil(np.maximum(p, q) + r + 2).astype(int).clip(0, N)
    rr, cc = np.mgrid[lo[0]:hi[0], lo[1]:hi[1]]
    pts = np.stack([rr, cc], -1).astype(float)
    d = q - p
    L2 = float(d @ d)
    t = np.zeros(rr.shape) if L2 == 0 else np.clip(((pts - p) @ d) / L2, 0, 1)
    proj = p + t[..., None] * d
    m = np.zeros((N, N), bool)
    m[lo[0]:hi[0], lo[1]:hi[1]] = np.linalg.norm(pts - proj, axis=-1) <= r
    return m


def _boundary_pixels(W: np.ndarray) -> np.ndarray:
    return W & ~ndi.binary_erosion(W, _C4, border_value=1)


def _nearest_other(lab: np.ndarray, p: np.ndarray, max_d: float) -> tuple[np.ndarray, float] | None:
    """Nearest conductor pixel of a different component than ``p`` within ``max_d``."""
    N = lab.shape[0]
    R = int(np.ceil(max_d)) + 1
    r0, c0 = int(p[0]), int(p[1])
    sl = (slice(max(0, r0 - R), min(N, r0 + R + 1)), slice(max(0, c0 - R), min(N, c0 + R + 1)))
    sub = lab[sl]
    own = lab[r0, c0]
    rr, cc = np.nonzero((sub > 0) & (sub != own))
    if rr.size == 0:
        return None
    rr = rr + sl[0].start
    cc = cc + sl[1].start
    d = np.hypot(rr - r0, cc - c0)
    k = int(np.argmin(d))
    if d[k] > max_d:
        return None
    return np.array([rr[k], cc[k]], float), float(d[k])


def _cross_section(W: np.ndarray, p: np.ndarray, n: np.ndarray, smax: int = 30) -> tuple[float, float]:
    """Extent [s1, s2] of conductor along p + s*n that contains s=0 (1-px steps)."""
    N = W.shape[0]

    def inside(s: float) -> bool:
        r, c = np.rint(p + s * n).astype(int)
        return 0 <= r < N and 0 <= c < N and bool(W[r, c])

    s1 = 0
    while s1 > -smax and inside(s1 - 1):
        s1 -= 1
    s2 = 0
    while s2 < smax and inside(s2 + 1):
        s2 += 1
    return float(s1), float(s2)


def _band(N: int, p: np.ndarray, n: np.ndarray, s_lo: float, s_hi: float, half_len: float) -> np.ndarray:
    """Oriented rectangle: s in [s_lo, s_hi] along n, |t| <= half_len along the tangent."""
    t = np.array([-n[1], n[0]])
    R = int(np.ceil(max(abs(s_lo), abs(s_hi)) + half_len + 2))
    r0, c0 = int(round(p[0])), int(round(p[1]))
    rr, cc = np.mgrid[max(0, r0 - R):min(N, r0 + R + 1), max(0, c0 - R):min(N, c0 + R + 1)]
    v = np.stack([rr - p[0], cc - p[1]], -1)
    s = v @ n
    u = v @ t
    m = np.zeros((N, N), bool)
    m[rr, cc] = (s >= s_lo - 0.5) & (s <= s_hi + 0.5) & (np.abs(u) <= half_len)
    return m


def _local_piece(W: np.ndarray, cut: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Part of (W & cut) 4-connected to pixel p (so cuts never touch other wires)."""
    lab = sk_label(W & cut, connectivity=1)
    k = lab[int(round(p[0])), int(round(p[1]))]
    return lab == k if k > 0 else np.zeros_like(W)


def _interior_point(W: np.ndarray, rng: np.random.Generator, lo: int, hi: int, min_depth: float):
    """Random near-medial conductor pixel and the unit normal across its trace.

    The normal is the smoothed-edge gradient at the nearest boundary point (pointing into
    the conductor), which is robust where the raw nearest-pixel vector is diagonal.
    """
    edt, ind = ndi.distance_transform_edt(W, return_indices=True)
    cand = np.argwhere(W & (edt >= min_depth))
    cand = cand[_in_region(cand, lo, hi)]
    if len(cand) == 0:
        return None
    p = cand[rng.integers(len(cand))].astype(float)
    b = (ind[0][int(p[0]), int(p[1])], ind[1][int(p[0]), int(p[1])])
    sm = ndi.gaussian_filter(W.astype(np.float32), 2.0)
    gy, gx = np.gradient(sm)
    n = np.array([gy[b], gx[b]], float)
    if np.linalg.norm(n) < 1e-6:
        return None
    n /= np.linalg.norm(n)
    return p, n


# --------------------------------------------------------------------------- defects
def inject_bridge(W, rng, dcfg: DefectCfg, lcfg: LayoutCfg):
    """Conductive capsule joining two components across a gap < bridge_max_gap."""
    N = W.shape[0]
    r = rng.uniform(*dcfg.bridge_radius)
    lo, hi = placement_region(lcfg, dcfg.bridge_max_gap / 2 + r)
    lab = sk_label(W, connectivity=1)
    cand = np.argwhere(_boundary_pixels(W))
    cand = cand[_in_region(cand, lo, hi)]
    for _ in range(40):
        if len(cand) == 0:
            break
        p = cand[rng.integers(len(cand))].astype(float)
        hit = _nearest_other(lab, p, dcfg.bridge_max_gap)
        if hit is None:
            continue
        q, d = hit
        cap = capsule(N, p, q, r)
        return W | cap, cap & ~W, {"radius": r, "gap": d - 1, "p": p.tolist(), "q": q.tolist()}
    return None


def inject_open(W, rng, dcfg: DefectCfg, lcfg: LayoutCfg):
    """Void cutting fully across a trace."""
    N = W.shape[0]
    L = rng.uniform(*dcfg.open_length)
    lo, hi = placement_region(lcfg, 12 + L)
    pt = _interior_point(W, rng, lo, hi, min_depth=5)
    if pt is None:
        return None
    p, n = pt
    s1, s2 = _cross_section(W, p, n)
    cut = _band(N, p, n, s1 - 6, s2 + 6, L / 2)
    rm = _local_piece(W, cut, p)
    return W & ~rm, rm, {"length": L, "width": s2 - s1 + 1, "p": p.tolist()}


def inject_particle(W, rng, dcfg: DefectCfg, lcfg: LayoutCfg):
    """Isolated conductive blob in dielectric (clearance >= 2 px to any conductor)."""
    N = W.shape[0]
    r = rng.uniform(*dcfg.particle_radius)
    lo, hi = placement_region(lcfg, r + 2)
    edt = ndi.distance_transform_edt(~W)
    for rr_ in (r, r * 0.8, max(2.5, r * 0.6)):
        cand = np.argwhere(edt >= rr_ + 3)
        cand = cand[_in_region(cand, lo, hi)]
        if len(cand):
            p = cand[rng.integers(len(cand))].astype(float)
            blob = capsule(N, p, p, rr_)
            return W | blob, blob & ~W, {"radius": rr_, "p": p.tolist()}
    return None


def inject_notch(W, rng, dcfg: DefectCfg, lcfg: LayoutCfg):
    """Near-miss constriction: bite from one side leaving ``notch_remaining`` px of wire."""
    N = W.shape[0]
    L = rng.uniform(*dcfg.notch_length)
    keep = int(rng.integers(dcfg.notch_remaining[0], dcfg.notch_remaining[1] + 1))
    lo, hi = placement_region(lcfg, 12 + L)
    pt = _interior_point(W, rng, lo, hi, min_depth=5)
    if pt is None:
        return None
    p, n = pt
    if rng.random() < 0.5:
        n = -n
    s1, s2 = _cross_section(W, p, n)
    if (s2 - s1 + 1) - keep < 2:
        return None
    cut = _band(N, p, n, s1 - 6, s2 - keep, L / 2)
    rm = _local_piece(W, cut, p + s1 * n)  # piece containing the bitten edge pixel
    return W & ~rm, rm, {"length": L, "remaining": keep, "width": s2 - s1 + 1, "p": p.tolist()}


def inject_filament(W, rng, dcfg: DefectCfg, lcfg: LayoutCfg):
    """Near-miss filament: capsule from one wire stopping 1-3 nm short of its neighbour."""
    N = W.shape[0]
    rf = rng.uniform(*dcfg.filament_radius)
    gap = int(rng.integers(dcfg.filament_gap[0], dcfg.filament_gap[1] + 1))
    lo, hi = placement_region(lcfg, dcfg.bridge_max_gap / 2 + rf)
    lab = sk_label(W, connectivity=1)
    cand = np.argwhere(_boundary_pixels(W))
    cand = cand[_in_region(cand, lo, hi)]
    for _ in range(60):
        if len(cand) == 0:
            break
        p = cand[rng.integers(len(cand))].astype(float)
        hit = _nearest_other(lab, p, dcfg.bridge_max_gap)
        if hit is None:
            continue
        q, d = hit
        if d < gap + 2 * rf + 3:
            continue
        u = (q - p) / d
        own = lab[int(p[0]), int(p[1])]
        dist_other = ndi.distance_transform_edt(~((lab > 0) & (lab != own)))
        clear = np.inf
        # walk the tip forward until the clearance to the neighbour equals `gap` px
        for back in np.arange(gap + rf + 2.0, gap + rf - 1.0, -0.25):
            fil = capsule(N, p - u, q - u * back, rf)
            add = fil & ~W
            if not add.any():
                continue
            clear = float(dist_other[add].min()) - 1.0  # dielectric px between filament and neighbour
            if clear <= gap:
                break
        if abs(clear - gap) > 0.75 or clear < 1:
            continue
        return W | fil, add, {"radius": rf, "gap_target": gap, "gap": clear, "p": p.tolist(), "q": q.tolist()}
    return None


DEFECTS = {
    "bridge": inject_bridge,
    "open": inject_open,
    "particle": inject_particle,
    "notch": inject_notch,
    "filament": inject_filament,
}
