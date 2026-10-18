"""v3 Phase 1 diagnostics: why does the frozen v2 verifier miss killers on noise-free images?

Three tools, none of which changes the verifier:
  * ``verify_scaled``: the frozen v2 singleton verifier (template field, splice region, dielectric hole fill)
    with every parameter that is defined in pixels rescaled for s sub-pixels per nm, returning the
    intermediate field and persistence diagrams. With s = 1 it equals ``pipeline.verify_patch`` (test).
  * ``delta_beta_curve``: Delta-beta of the same diagrams at every lifetime threshold tau of a grid
    (the persistence is computed once; only the lifetime cut-off moves).
  * ``defect_saddle``: the field value at the defect that decides the call. For a merge (bridge, filament)
    the highest level at which the two wires are connected through the field; for a cut (open, notch) the
    level at which the two halves of the wire stay connected; for a particle its peak and the level at
    which it joins a wire. A merge is counted when saddle > 1 - tau, a cut when saddle < 1 - tau.

Pixel-defined v2 parameters and their rescaling at s sub-pixels per nm (lengths held fixed in nm):
  pre_smooth px * s, region_tol px * s, window margin px * s, calibration feature sigmas px * s,
  SDF-energy smoothing (1 px) * s, gamma (1/px) / s, cap (px) * s, peak-z radius (20 px) * s.
  tau, d*, e*, z* are in field / contrast units and unchanged. Synthetic data only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label

from .modules.align_sdf import align_to_template, energy_map, peak_z
from .modules.calibration import Calibration, features
from .modules.fields import build_field
from .modules.topology import (
    _reference, betti_from_intervals, classify_verdict, compute_bottleneck_distances, compute_persistence_intervals,
)
from .pipeline import PipelineCfg
from .topo_utils import Window


@dataclass
class ScaledResult:
    verdict: str
    delta_beta: tuple[int, int]
    beta_ref: tuple[int, int]
    bottleneck: float
    energy: float
    peak_z: float
    field: np.ndarray          # spliced field used for H0 (CAD outside the relevance region)
    diagrams: tuple            # (d0_img, d1_img, d0_ref, d1_ref)
    region: np.ndarray


def scaled_window(win: Window, s: int) -> Window:
    return Window(win.size * s, win.margin * s)


def verify_scaled(image: np.ndarray, cad: np.ndarray, cal: Calibration, win: Window, pc: PipelineCfg, s: int = 1) -> ScaledResult:
    """Frozen v2 singleton verifier at s sub-pixels per nm. ``win`` and ``cal`` must already be at s."""
    if (pc.field_option, pc.region_mode, pc.h1_fill) != ("template", "splice", "dielectric"):
        raise ValueError("verify_scaled implements the frozen v2 configuration only")
    template = cal.predict(cad)
    aligned, _ = align_to_template(image, template) if pc.align else (np.asarray(image, float), (0.0, 0.0))
    emap = energy_map(image, cad, cal, win, pc.gamma / s, pc.cap * s, smooth=1.0 * s, aligned=aligned)
    energy, pz = float(emap.max()), peak_z(emap, radius=20 * s)
    field, _ = build_field(aligned, option="template", levels=cal.levels, pre_smooth=pc.pre_smooth * s,
                           template=template, cad_smooth=features(cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
    tol = pc.region_tol * s
    reg, d0r, d1r = _reference(cad, win, tol, "dielectric")
    f_eval = np.where(reg, field, cad.astype(float))
    d0i, _ = compute_persistence_intervals(f_eval)
    d1i = compute_persistence_intervals(np.where(reg, field, 0.0))[1]
    bi, br = betti_from_intervals(d0i, d1i, pc.tau), betti_from_intervals(d0r, d1r, pc.tau)
    db = (bi[0] - br[0], bi[1] - br[1])
    b0, b1 = compute_bottleneck_distances(d0i, d1i, d0r, d1r)
    bot = max(b0, b1)
    verdict = classify_verdict(db, bot, energy, pc.d_star, pc.e_star, pz, pc.z_star)
    return ScaledResult(verdict, db, br, bot, energy, pz, f_eval, (d0i, d1i, d0r, d1r), reg)


def _lifetimes(d: np.ndarray) -> np.ndarray:
    return (d[:, 1] - d[:, 0]) if len(d) else np.empty(0)


def delta_beta_curve(diagrams: tuple, taus: np.ndarray) -> np.ndarray:
    """(len(taus), 2) int array of Delta-beta at each lifetime threshold (same rule as ``betti_from_intervals``)."""
    d0i, d1i, d0r, d1r = diagrams
    taus = np.asarray(taus, float)[:, None]
    cnt = [((_lifetimes(d)[None, :] >= taus).sum(1) if len(d) else np.zeros(len(taus), int)) for d in (d0i, d1i, d0r, d1r)]
    return np.stack([cnt[0] - cnt[2], cnt[1] - cnt[3]], 1).astype(int)


# ------------------------------------------------------------------ saddle values at the defect
def saddle(P: np.ndarray, A: np.ndarray, B: np.ndarray, region: np.ndarray) -> float:
    """Largest t such that A and B are 4-connected inside {P >= t} & region (nan if never)."""
    vals = np.unique(P[region])
    lo, hi = 0, len(vals) - 1          # invariant: connected at vals[lo] (checked), search the largest

    def conn(t: float) -> bool:
        lab = sk_label((P >= t) & region, connectivity=1)
        la, lb = np.unique(lab[A & (lab > 0)]), np.unique(lab[B & (lab > 0)])
        return bool(np.intersect1d(la, lb).size)

    if len(vals) == 0 or not conn(vals[0]):
        return float("nan")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if conn(vals[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(vals[lo])


def _box(mask: np.ndarray, pts: list, pad: int) -> tuple[slice, slice]:
    rr, cc = np.nonzero(mask)
    rs = list(rr) + [int(p[0]) for p in pts]
    cs = list(cc) + [int(p[1]) for p in pts]
    H, W = mask.shape
    return slice(max(0, min(rs) - pad), min(H, max(rs) + pad + 1)), slice(max(0, min(cs) - pad), min(W, max(cs) + pad + 1))


def _to_tile(p, pad: int, s: int) -> np.ndarray:
    return (np.asarray(p, float) - pad) * s + (s - 1) / 2.0


def defect_saddle(kind: str, P: np.ndarray, cad: np.ndarray, W: np.ndarray, mask: np.ndarray, params: dict,
                  pad: int, s: int = 1) -> dict:
    """Decisive field values at the defect (tile coords at s sub-pixels per nm). See module docstring.

    Returns {'saddle': float, 'peak': float (particle only), 'geom': 'ok' | reason}. ``params`` are the
    testbed's defect params (canvas coords at 1 nm/px); ``pad`` is the canvas pad (1 nm px).
    """
    out = {"saddle": float("nan"), "peak": float("nan"), "geom": "ok"}
    if kind == "nuisance" or not mask.any():
        out["geom"] = "no_defect"
        return out
    c = cad.astype(bool)
    if kind in ("bridge", "filament"):
        p, q = _to_tile(params["p"], pad, s), _to_tile(params["q"], pad, s)
        sl = _box(mask, [p, q], 10 * s)
        cb, Pb = c[sl], P[sl]
        lab = sk_label(cb, connectivity=1)
        off = np.array([sl[0].start, sl[1].start], float)

        def comp_near(pt, exclude=0):
            ids = np.argwhere((lab > 0) & (lab != exclude))
            if not len(ids):
                return 0
            k = np.argmin(np.hypot(*(ids - (pt - off)).T))
            return int(lab[tuple(ids[k])])

        a = comp_near(p)
        b = comp_near(q, exclude=a)
        if a == 0 or b == 0:
            out["geom"] = "no_pair"
            return out
        core = ndi.binary_erosion(cb, iterations=s)
        A, B = (lab == a) & core, (lab == b) & core
        out["saddle"] = saddle(Pb, A if A.any() else lab == a, B if B.any() else lab == b, np.ones_like(cb))
        return out
    if kind in ("open", "notch"):
        p = _to_tile(params["p"], pad, s)
        sl = _box(mask, [p], 12 * s)
        cb, Pb, mb = c[sl], P[sl], mask[sl]
        lab = sk_label(cb, connectivity=1)
        ids = np.unique(lab[ndi.binary_dilation(mb, iterations=2 * s) & (lab > 0)])
        if ids.size == 0:
            out["geom"] = "no_wire"
            return out
        comp = np.isin(lab, ids)
        far = ndi.distance_transform_edt(~mb) > 6 * s
        sides = sk_label(comp & far, connectivity=1)
        sizes = np.bincount(sides.ravel())[1:]
        if len(sizes) < 2:
            out["geom"] = "one_side"
            return out
        top = np.argsort(sizes)[::-1][:2] + 1
        region = ndi.binary_dilation(comp, iterations=3 * s)
        out["saddle"] = saddle(Pb, sides == top[0], sides == top[1], region)
        return out
    if kind == "particle":
        sl = _box(mask, [], 12 * s)
        cb, Pb, mb = c[sl], P[sl], mask[sl]
        out["peak"] = float(Pb[mb].max())
        out["saddle"] = saddle(Pb, mb, cb, np.ones_like(cb)) if cb.any() else 0.0
        return out
    raise ValueError(kind)


def margin(kind: str, sad: dict, tau: float) -> float:
    """Signed field margin toward a *change* call (> 0: the frozen rule sees the change). Field units.

    merge (bridge, filament): saddle - (1 - tau); cut (open, notch): (1 - tau) - saddle;
    particle: (peak - saddle) - tau (its H0 lifetime minus tau).
    """
    if kind in ("bridge", "filament"):
        return sad["saddle"] - (1 - tau)
    if kind in ("open", "notch"):
        return (1 - tau) - sad["saddle"]
    if kind == "particle":
        return (sad["peak"] - sad["saddle"]) - tau
    return float("nan")
