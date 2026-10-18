"""v3 Module 4 rule: located topology events instead of a Delta-beta sum (docs/V3_PHASE2_DESIGN.md).

The field, alignment, relevance region and splice are exactly v2's. What changes is how a patch is called:

  * Seeds: the CAD components (4-connected). Their cores are the components eroded by 1 px.
  * Merge events, per seed pair (CAD components closer than ``pair_dist`` px): the observed merge level
    s_ij is the highest field level at which the two cores are 4-connected through the spliced field
    (a maximin path, by morphological reconstruction inside a box around seed i); the expected level e_ij
    is the same quantity on the field of a defect-free image (the template-anchored field of T(C) itself,
    zero residual). A bridge between i and j is called iff s_ij - e_ij >= m_merge, and, when ``normalise``
    is on, also s_ij / body_ij >= m_norm, where body_ij is the brightest non-CAD field value within
    ``body_radius`` px of the saddle (a grain-dimmed bridge is uniformly dim, so s ~ body; a filament
    1-3 nm short has a bright body and a dip, so s << body).
  * Splits, new components and holes are read from the persistence pairs *with their locations*
    (GUDHI vertex pairs): each H0 bar is assigned to the seed containing its birth vertex (or to the
    dielectric), each H1 bar to the CAD hole containing its death vertex. v2 counts them with lifetime
    >= tau; the per-neck option (``m_cut``) instead calls a split when the expected field at the neck
    minus the observed one is >= m_cut. A new component whose merge point lies where the expected field
    is >= ``e_edge`` (a CAD wire edge) is treated as a dimmed protrusion, not a particle (option).
  * Verdict from the event set (no summing, so a seen bridge is never cancelled by an unrelated split):
    any bridge or new-hole event -> KILLER_BRIDGE; else any split, particle or lost-hole event ->
    KILLER_OPEN/PARTICLE; else v2's near-miss gate (bottleneck >= d*, energy >= e*, peak-z >= z*); else NUISANCE.

``evidence`` extracts everything once per patch (so a parameter grid is cheap); ``decide`` applies a rule.
``v2_events`` reads v2's own counts back from the located bars (their sum equals v2's Delta-beta; tested).
Units: field in [0, 1]; lengths px (= nm at 1 nm/px).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import gudhi
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from skimage.morphology import reconstruction

from .topology import KILLER_BRIDGE, KILLER_OPEN_PARTICLE, NEAR_MISS, NUISANCE, classify_verdict

_C4 = ndi.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class EventRuleCfg:
    """v3 decision rule settings, passed alongside the (unchanged) v2 ``PipelineCfg``."""

    rule: str = "v3_events"            # 'v2_global_tau' (reproduces v2 exactly) | 'v3_events'
    pair_dist: float = 30.0            # px; seed pairs considered for a bridge (a priori, ~2 pitches; not searched)
    m_merge: float = 0.5               # field units; observed minus expected merge level (selected on val)
    normalise: bool = False            # also require s / body >= m_norm (grain-dimmed bridges)
    m_norm: float = 0.0
    body_radius: float = 6.0           # px, a priori
    m_cut: float | None = None         # None: v2 lifetime-tau splits; else per-neck expected-minus-observed margin
    e_edge: float | None = None        # None: off; else expected field at a new component's merge point marking a protrusion

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    """Everything a rule needs, extracted once per patch."""

    pairs: list[dict]                  # i, j, d, s, e, body
    h0: np.ndarray                     # (n, 5): f_birth, f_death (nan = essential), seed (0 = dielectric), f_expected_at_death, essential
    h1: np.ndarray                     # (m, 3): f_birth, f_death, cad_hole_id (0 = none)
    cad_holes: np.ndarray              # ids of CAD holes (dielectric fill outside the region)
    n_seeds: int
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ persistence with locations
def located_pairs(f: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex pairs of the sublevel filtration of 1 - f in dimension ``dim``.

    Returns (regular (k, 2) [birth_idx, death_idx], essential (e,) birth_idx) as (row, col) flat C indices.
    """
    F = np.clip(1.0 - f, 0.0, 1.0)
    cc = gudhi.CubicalComplex(vertices=F)
    cc.compute_persistence()
    reg, ess = cc.vertices_of_persistence_pairs()
    shape = f.shape

    def conv(a):  # GUDHI vertex indices are column-major
        a = np.asarray(a, dtype=np.int64)
        if a.size == 0:
            return a
        r, c = np.unravel_index(a, shape, order="F")
        return np.ravel_multi_index((r, c), shape)

    r = reg[dim] if dim < len(reg) else np.empty((0, 2), np.int64)
    e = ess[dim] if dim < len(ess) else np.empty(0, np.int64)
    r = np.asarray(r, np.int64).reshape(-1, 2)
    return np.stack([conv(r[:, 0]), conv(r[:, 1])], 1) if len(r) else np.empty((0, 2), np.int64), conv(np.asarray(e).ravel())


# ------------------------------------------------------------------ merge levels per seed pair
def _pair_list(lab: np.ndarray, n: int, active: np.ndarray, pair_dist: float):
    edts, pairs = {}, []
    for i in range(1, n + 1):
        if active[i]:
            edts[i] = ndi.distance_transform_edt(lab != i)
    for i in sorted(edts):
        for j in sorted(edts):
            if j <= i:
                continue
            d = float(edts[i][lab == j].min())
            if d <= pair_dist:
                pairs.append((i, j, d))
    return pairs, edts


def merge_levels(f: np.ndarray, lab: np.ndarray, cores: dict, pairs: list, edts: dict, pair_dist: float,
                 cad: np.ndarray, body_radius: float) -> list[dict]:
    """Maximin merge level s_ij between seed cores, its saddle location and the body level near it."""
    H, W = f.shape
    out = []
    by_i: dict[int, list] = {}
    for i, j, d in pairs:
        by_i.setdefault(i, []).append((j, d))
    conductor_zone = ndi.binary_dilation(cad, _C4)
    for i, lst in by_i.items():
        rr, cc = np.nonzero(lab == i)
        pad = int(np.ceil(pair_dist)) + 3
        sl = (slice(max(0, rr.min() - pad), min(H, rr.max() + pad + 1)), slice(max(0, cc.min() - pad), min(W, cc.max() + pad + 1)))
        fb = f[sl]
        marker = np.where(cores[i][sl], fb, 0.0)
        b = reconstruction(marker, fb, method="dilation", footprint=_C4)
        for j, d in lst:
            cj = cores[j][sl]
            if not cj.any():
                out.append(dict(i=i, j=j, d=d, s=float("nan"), body=float("nan"), pass_rc=None))
                continue
            s = float(b[cj].max())
            cand = (fb == s) & (b >= s)
            if cand.any():
                score = (edts[i][sl] + edts[j][sl])
                k = np.argmin(np.where(cand, score, np.inf))
                pr, pc_ = np.unravel_index(k, fb.shape)
                y0, x0 = pr + sl[0].start, pc_ + sl[1].start
                R = int(np.ceil(body_radius))
                ys, xs = slice(max(0, y0 - R), min(H, y0 + R + 1)), slice(max(0, x0 - R), min(W, x0 + R + 1))
                yy, xx = np.mgrid[ys, xs]
                near = ((yy - y0) ** 2 + (xx - x0) ** 2 <= body_radius ** 2) & ~conductor_zone[ys, xs]
                body = float(f[ys, xs][near].max()) if near.any() else s
                out.append(dict(i=i, j=j, d=d, s=s, body=max(body, s), pass_rc=(int(y0), int(x0))))
            else:
                out.append(dict(i=i, j=j, d=d, s=s, body=s, pass_rc=None))
    return out


# ------------------------------------------------------------------ CAD-only quantities (cached)
_CACHE: OrderedDict = OrderedDict()


def _cad_info(cad: np.ndarray, reg: np.ndarray, f_exp_eval: np.ndarray, pair_dist: float, body_radius: float, key_extra) -> dict:
    key = (hashlib.sha1(np.ascontiguousarray(cad, dtype=bool).tobytes()).hexdigest(),
           hashlib.sha1(np.ascontiguousarray(reg).tobytes()).hexdigest(), pair_dist, body_radius, key_extra)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    c = cad.astype(bool)
    lab, n = sk_label(c, connectivity=1, return_num=True)
    active = np.zeros(n + 1, bool)
    active[np.unique(lab[reg & c])] = True
    active[0] = False
    cores = {}
    for i in range(1, n + 1):
        comp = lab == i
        core = ndi.binary_erosion(comp, _C4)
        cores[i] = core if core.any() else comp
    pairs, edts = _pair_list(lab, n, active, pair_dist)
    exp = {(p["i"], p["j"]): p["s"] for p in merge_levels(f_exp_eval, lab, cores, pairs, edts, pair_dist, c, body_radius)}
    hfill = np.where(reg, c, False)
    diel = sk_label(~hfill, connectivity=2)
    border = np.unique(np.concatenate([diel[0], diel[-1], diel[:, 0], diel[:, -1]]))
    holes = np.setdiff1d(np.unique(diel), np.r_[border, 0])
    hole_lab = np.where(np.isin(diel, holes), diel, 0)
    info = dict(lab=lab, n=n, cores=cores, pairs=pairs, edts=edts, exp=exp, hole_lab=hole_lab, holes=holes)
    _CACHE[key] = info
    if len(_CACHE) > 32:
        _CACHE.popitem(last=False)
    return info


def evidence(field_img: np.ndarray, cad: np.ndarray, reg: np.ndarray, f_expected: np.ndarray, pair_dist: float = 30.0,
             body_radius: float = 6.0) -> Evidence:
    """Located evidence for one patch. ``field_img`` and ``f_expected`` are pre-smoothed fields in [0, 1]
    (image, and the defect-free template field); ``reg`` is the v2 relevance region (splice as in v2)."""
    c = cad.astype(bool)
    cadf = c.astype(float)
    f_eval = np.where(reg, field_img, cadf)
    fx_eval = np.where(reg, f_expected, cadf)
    info = _cad_info(c, reg, fx_eval, pair_dist, body_radius, hashlib.sha1(np.ascontiguousarray(fx_eval).tobytes()).hexdigest())
    lab = info["lab"]
    pairs = merge_levels(f_eval, lab, info["cores"], info["pairs"], info["edts"], pair_dist, c, body_radius)
    for p in pairs:
        p["e"] = info["exp"].get((p["i"], p["j"]), float("nan"))
    fe, fx, lf = f_eval.ravel(), fx_eval.ravel(), lab.ravel()
    reg0, ess0 = located_pairs(f_eval, 0)
    h0 = [(fe[b], fe[d], lf[b], fx[d], 0.0) for b, d in reg0] + [(fe[b], np.nan, lf[b], np.nan, 1.0) for b in ess0]
    f_h1 = np.where(reg, field_img, 0.0)
    reg1, _ = located_pairs(f_h1, 1)
    fh, hl = f_h1.ravel(), info["hole_lab"].ravel()
    h1 = [(fh[b], fh[d], hl[d]) for b, d in reg1]
    return Evidence(pairs, np.array(h0, float).reshape(-1, 5), np.array(h1, float).reshape(-1, 3), info["holes"], info["n"])


# ------------------------------------------------------------------ rules
def _h0_counts(ev: Evidence, tau: float):
    life = np.where(ev.h0[:, 4] == 1, np.inf, ev.h0[:, 0] - ev.h0[:, 1])
    counted = life >= tau
    seeds = ev.h0[:, 2].astype(int)
    return counted, seeds


def _hole_events(ev: Evidence, tau: float) -> tuple[list, list]:
    counted = (ev.h1[:, 0] - ev.h1[:, 1]) >= tau if len(ev.h1) else np.zeros(0, bool)
    hid = ev.h1[:, 2].astype(int) if len(ev.h1) else np.zeros(0, int)
    new = [dict(kind="new_hole", hole=0)] * int((counted & (hid == 0)).sum())
    lost = []
    for h in ev.cad_holes:
        k = int((counted & (hid == h)).sum())
        if k == 0:
            lost.append(dict(kind="lost_hole", hole=int(h)))
        new += [dict(kind="new_hole", hole=int(h))] * max(0, k - 1)
    return new, lost


def v2_events(ev: Evidence, tau: float) -> dict[str, list]:
    """v2's counts read from located bars: merges (seeds with no counted bar), splits, particles, holes."""
    counted, seeds = _h0_counts(ev, tau)
    per = np.bincount(seeds[counted], minlength=ev.n_seeds + 1)
    merges = [dict(kind="merge", seed=int(k)) for k in range(1, ev.n_seeds + 1) if per[k] == 0]
    splits = [dict(kind="split", seed=int(k)) for k in range(1, ev.n_seeds + 1) for _ in range(max(0, per[k] - 1))]
    parts = [dict(kind="particle")] * int(per[0])
    new, lost = _hole_events(ev, tau)
    return dict(bridge=merges + new, open=splits + parts + lost)


def v2_delta_beta(ev: Evidence, tau: float) -> tuple[int, int]:
    e = v2_events(ev, tau)
    nm = sum(x["kind"] == "merge" for x in e["bridge"]); nh = len(e["bridge"]) - nm
    nl = sum(x["kind"] == "lost_hole" for x in e["open"]); ns = len(e["open"]) - nl
    return ns - nm, nh - nl


def v3_events(ev: Evidence, rc: EventRuleCfg, tau: float) -> dict[str, list]:
    bridge = []
    for p in ev.pairs:
        if np.isnan(p["s"]) or np.isnan(p["e"]):
            continue
        if p["s"] - p["e"] >= rc.m_merge and (not rc.normalise or p["s"] / max(p["body"], 1e-9) >= rc.m_norm):
            bridge.append(dict(kind="bridge", seeds=(p["i"], p["j"]), s=p["s"], e=p["e"], body=p["body"], margin=p["s"] - p["e"] - rc.m_merge))
    counted, seeds = _h0_counts(ev, tau)
    opens = []
    if rc.m_cut is None:
        per = np.bincount(seeds[counted], minlength=ev.n_seeds + 1)
        opens += [dict(kind="split", seed=int(k)) for k in range(1, ev.n_seeds + 1) for _ in range(max(0, per[k] - 1))]
    else:
        fin = ev.h0[:, 4] == 0
        drop = (ev.h0[:, 3] - ev.h0[:, 1] >= rc.m_cut) & (ev.h0[:, 0] - ev.h0[:, 1] >= rc.m_cut)
        opens += [dict(kind="split", seed=int(k)) for k in seeds[fin & drop & (seeds > 0)]]
    part = counted & (seeds == 0)
    if rc.e_edge is not None:
        part &= ~(ev.h0[:, 3] >= rc.e_edge)
    opens += [dict(kind="particle")] * int(part.sum())
    new, lost = _hole_events(ev, tau)
    return dict(bridge=bridge + new, open=opens + lost)


def decide(ev: Evidence, rc: EventRuleCfg, tau: float, bottleneck: float, energy: float, pz: float,
           d_star: float, e_star: float, z_star: float) -> tuple[str, dict]:
    if rc.rule == "v2_global_tau":  # v2's ladder on the Delta-beta sum (cancellation included, as in v2)
        return classify_verdict(v2_delta_beta(ev, tau), bottleneck, energy, d_star, e_star, pz, z_star), v2_events(ev, tau)
    if rc.rule != "v3_events":
        raise ValueError(f"unknown rule {rc.rule!r}")
    ev_set = v3_events(ev, rc, tau)
    if ev_set["bridge"]:
        verdict = KILLER_BRIDGE
    elif ev_set["open"]:
        verdict = KILLER_OPEN_PARTICLE
    elif bottleneck >= d_star and energy >= e_star and pz >= z_star:
        verdict = NEAR_MISS
    else:
        verdict = NUISANCE
    return verdict, ev_set
