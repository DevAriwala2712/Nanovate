"""Plain-English explanations and a display-only site locator for one analysis (no LLM; fixed templates).

``explain`` turns the reason fields of the three methods into sentences; every number in a sentence is
formatted from those fields. ``locate`` marks where the image differs from CAD in the way Δβ says it does,
for display only: it thresholds the conductor field at 0.5 and looks for difference blobs that join or cut
conductors. It never feeds a verdict, and when it finds no site consistent with Δβ it says so and marks nothing.
Units: px (= nm at 1 nm/px), boxes as [row0, col0, row1, col1] in the 256 x 256 tile frame, end-exclusive.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label

from ..topo_utils import Window

LOCATOR_METHOD = ("display-only heuristic: conductor field thresholded at 0.5, difference blobs vs CAD that join "
                  "or cut conductors; not used by any verdict")
MIN_BLOB = 3      # px, smaller difference blobs are ignored by the locator
MAX_SITES = 5


def _n(x, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ge(a, b) -> bool:
    return a is not None and b is not None and a >= b


def explain_verifier(verdict: str, r: dict, recovered_shift=None, applied_shift=None) -> list[str]:
    """Sentences for the verifier verdict, built only from its reason fields."""
    d0, d1 = r["delta_betti"]
    (ref0, ref1), (img0, img1) = r["beta_ref"], r["beta_img"]
    out = []
    if d0 < 0:
        what = "two separate CAD conductors are connected" if d0 == -1 else "separate CAD conductors are connected"
        out.append(f"β₀ dropped by {-d0} (CAD {ref0} conductors, image {img0}): {what} in the image → bridge.")
    if d0 > 0:
        out.append(f"β₀ rose by {d0} (CAD {ref0} conductors, image {img0}): a CAD conductor is split, "
                   "or an isolated conductive blob appeared → open or particle.")
    if d1 > 0:
        out.append(f"β₁ rose by {d1} (CAD {ref1} enclosed holes, image {img1}): conductors are joined around a gap, "
                   "closing a new dielectric loop → bridge.")
    if d1 < 0:
        out.append(f"β₁ dropped by {-d1} (CAD {ref1} enclosed holes, image {img1}): a conductor loop of the CAD is broken, "
                   "opening an enclosed hole → open.")
    if d0 or d1:
        out.append(f"Verdict {verdict} comes from the Δβ rule alone (persistence lifetime τ = {r['tau']:g}); "
                   "the near-miss values are not used for killers.")
    else:
        out.append(f"Topology matches CAD: Δβ = (0, 0) at persistence lifetime τ = {r['tau']:g} "
                   f"(β = ({img0}, {img1}) in both).")
        if not r.get("near_miss_enabled", True) or r.get("d_star") is None:
            out.append("The near-miss rule is off in this configuration → NUISANCE.")
        else:
            conds = [("bottleneck distance", r["bottleneck_max"], r["d_star"], "d*", 3),
                     ("max SDF energy", r["max_sdf_energy"], r["e_star"], "e*", 2),
                     ("peak-z", r.get("peak_z"), r.get("z_star"), "z*", 1)]
            conds = [c for c in conds if c[2] is not None]
            parts = [f"{name} {_n(v, nd)} {'≥' if _ge(v, t) else '<'} {sym} {_n(t, nd)}" for name, v, t, sym, nd in conds]
            if all(_ge(v, t) for _, v, t, _, _ in conds):
                out.append("Near-miss rule: " + ", ".join(parts) + " → all hold → NEAR_MISS "
                           "(the image is close to changing topology near the CAD edges: a pinch or a filament).")
            else:
                failed = [p for p, c in zip(parts, conds) if not _ge(c[1], c[2])]
                out.append("Near-miss rule: " + ", ".join(parts) + f" → {'; '.join(failed)} fails → NUISANCE "
                           "(roughness, grain or noise without a topology change).")
    if r.get("singleton"):
        out.append("Singleton path: no same-layout stack was available, so Module 2 (low-rank denoising) was not used.")
    if recovered_shift is not None:
        s = f"Module 3 recovered an acquisition shift of ({recovered_shift[0]:+.2f}, {recovered_shift[1]:+.2f}) px (row, col)"
        if applied_shift is not None:
            s += f"; the simulator applied ({applied_shift[0]:+.2f}, {applied_shift[1]:+.2f}) px"
        out.append(s + ".")
    return out


def event_delta(r: dict) -> tuple[int, int]:
    """(Δβ₀, Δβ₁) implied by the v3 located events, for the display locator only: merges/bridges and new holes vs
    splits, particles and lost holes. The v3 verdict itself is decided from the events, not from this sum."""
    ev = r["events"]
    n_merge = sum(e["kind"] in ("bridge", "merge") for e in ev["bridge"])
    n_lost = sum(e["kind"] == "lost_hole" for e in ev["open"])
    return (len(ev["open"]) - n_lost) - n_merge, (len(ev["bridge"]) - n_merge) - n_lost


def explain_verifier_v3(verdict: str, r: dict, recovered_shift=None, applied_shift=None) -> list[str]:
    """Sentences for a v3 verdict (located-event rule), built only from its reason fields."""
    ev = r["events"]
    rc = r.get("rule_cfg") or {}
    bridges = [e for e in ev["bridge"] if e["kind"] == "bridge"]
    new_holes = [e for e in ev["bridge"] if e["kind"] != "bridge"]
    splits = [e for e in ev["open"] if e["kind"] == "split"]
    parts = [e for e in ev["open"] if e["kind"] == "particle"]
    lost = [e for e in ev["open"] if e["kind"] == "lost_hole"]
    out = []
    for b in bridges[:3]:
        out.append(f"Merge event: between CAD conductors {b['seeds'][0]} and {b['seeds'][1]} the observed merge level {_n(b['s'])} "
                   f"exceeds the template-expected level {_n(b['e'])} by at least the margin {_n(rc.get('m_merge'), 2)} → bridge.")
    if len(bridges) > 3:
        out.append(f"… and {len(bridges) - 3} more merge event(s).")
    if new_holes:
        out.append(f"{len(new_holes)} new enclosed dielectric hole(s) (persistence lifetime ≥ τ = {r['tau']:g}): conductors joined around a gap → bridge.")
    if splits:
        out.append(f"{len(splits)} split event(s): a CAD conductor breaks into separate pieces (lifetime ≥ τ = {r['tau']:g}) → open.")
    if parts:
        out.append(f"{len(parts)} isolated conductive blob(s) in the dielectric (lifetime ≥ τ = {r['tau']:g}) → particle.")
    if lost:
        out.append(f"{len(lost)} CAD hole(s) lost: a conductor loop is broken → open.")
    if bridges or new_holes or splits or parts or lost:
        out.append(f"Verdict {verdict} comes from the located events (v3 rule: a seen bridge is never cancelled by an open); "
                   "the near-miss values are not used for killers.")
    else:
        out.append(f"No located merge, split, particle or hole event (v3 rule, τ = {r['tau']:g}): topology matches CAD.")
        conds = [("bottleneck distance", r.get("bottleneck_max"), r.get("d_star"), "d*", 3),
                 ("max SDF energy", r.get("max_sdf_energy"), r.get("e_star"), "e*", 2),
                 ("peak-z", r.get("peak_z"), r.get("z_star"), "z*", 1)]
        conds = [c for c in conds if c[2] is not None]
        parts_t = [f"{name} {_n(v, nd)} {'≥' if _ge(v, t) else '<'} {sym} {_n(t, nd)}" for name, v, t, sym, nd in conds]
        if conds and all(_ge(v, t) for _, v, t, _, _ in conds):
            out.append("Near-miss rule: " + ", ".join(parts_t) + " → all hold → NEAR_MISS "
                       "(the image is close to changing topology near the CAD edges: a pinch or a filament).")
        else:
            failed = [p for p, c in zip(parts_t, conds) if not _ge(c[1], c[2])]
            out.append("Near-miss rule: " + ", ".join(parts_t) + f" → {'; '.join(failed) or 'off'} fails → NUISANCE "
                       "(roughness, grain or noise without a topology change).")
    if r.get("singleton"):
        out.append("Singleton path: no same-layout stack was available, so Module 2 (low-rank denoising) was not used.")
    if recovered_shift is not None:
        s = f"Module 3 recovered an acquisition shift of ({recovered_shift[0]:+.2f}, {recovered_shift[1]:+.2f}) px (row, col)"
        if applied_shift is not None:
            s += f"; the simulator applied ({applied_shift[0]:+.2f}, {applied_shift[1]:+.2f}) px"
        out.append(s + ".")
    return out


def explain_cnn(c: dict) -> list[str]:
    rel = "≥" if c["flagged"] else "<"
    out = [f"CNN: killer probability {_n(c['prob'])} {rel} threshold {_n(c['threshold'])} (chosen on Manhattan val) → "
           f"{'killer' if c['flagged'] else 'not a killer'}. It gives no reason and has no near-miss bin."]
    if c.get("domain_note"):
        out.append(f"CNN caveat: {c['domain_note']}.")
    return out


def explain_d2d(d: dict) -> list[str]:
    if not d.get("available"):
        return [f"Die-to-die: {d['why']}."]
    out = []
    for m in ("single", "double"):
        x = d.get(m, {})
        if "flagged" not in x:
            out.append(f"Die-to-die ({m}): {x.get('why', 'not available')}.")
            continue
        ru = x["rule"]
        out.append(f"Die-to-die ({m}): the largest region above θ = {ru['theta']:g} (smoothing σ {ru['sigma']:g} px) covers "
                   f"{x['max_area']} px {'≥' if x['flagged'] else '<'} minimum area {ru['min_area']} px → "
                   f"{'flagged' if x['flagged'] else 'not flagged'}. It measures pixel difference, not connectivity.")
    return out


def explain_truth(t: dict | None) -> list[str]:
    if t is None:
        return ["No ground truth exists for this input, so no verdict is marked correct or wrong."]
    return [f"Ground truth (exact topology of the simulated wafer W vs CAD): {t['label']} "
            f"(injected {t['kind']}, Δβ ({t['d_beta'][0]}, {t['d_beta'][1]}))."]


def explain(resp: dict) -> dict:
    """All explanation blocks for one /verify response body (before encoding)."""
    v = resp["verifier"]
    applied = resp["truth"]["applied_shift"] if resp.get("truth") else None
    loc = resp.get("locator") or {}
    loc_text = []
    if loc.get("sites"):
        loc_text.append(f"Marked {len(loc['sites'])} site(s) where the image {'joins' if loc['sites'][0]['type'] == 'join' else 'cuts or adds'} "
                        f"conductors ({LOCATOR_METHOD}).")
    elif loc.get("applicable"):
        loc_text.append("The display locator found no site consistent with Δβ in the thresholded field, so no site is marked "
                        "(the verdict itself comes from persistence, not from this locator).")
    if loc.get("energy_peak"):
        p = loc["energy_peak"]
        loc_text.append(f"The SDF tension energy peaks at (row {p['row']}, col {p['col']}) with {_n(p['value'], 2)}.")
    fn = explain_verifier_v3 if v["reason"].get("rule") == "v3_events" else explain_verifier
    return {"verifier": fn(v["verdict"], v["reason"], v.get("recovered_shift"), applied),
            "cnn": explain_cnn(resp["cnn"]), "d2d": explain_d2d(resp["d2d"]), "truth": explain_truth(resp.get("truth")),
            "locator": loc_text}


# ---- display-only locator -------------------------------------------------------------------------------
def _box(mask: np.ndarray) -> list[int]:
    rr, cc = np.nonzero(mask)
    return [int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1]


def _contacts(blob: np.ndarray, lab: np.ndarray) -> int:
    """Number of distinct labelled components (of ``lab``) touching the 1-px dilation of ``blob``."""
    ring = ndi.binary_dilation(blob, structure=ndi.generate_binary_structure(2, 1)) & ~blob
    ids = np.unique(lab[ring])
    return int(np.count_nonzero(ids))


def _contact_regions(blob: np.ndarray, other: np.ndarray) -> int:
    """Number of separate contact patches between ``blob`` and mask ``other`` (a loop closure touches twice)."""
    ring = ndi.binary_dilation(blob, structure=ndi.generate_binary_structure(2, 1)) & ~blob & other
    return int(sk_label(ring, connectivity=2).max())


def locate(field: np.ndarray, cad: np.ndarray, delta_betti, energy: np.ndarray, win: Window) -> dict:
    """Sites consistent with Δβ (display only) plus the SDF-energy peak. See the module docstring."""
    d0, d1 = int(delta_betti[0]), int(delta_betti[1])
    img = np.asarray(field) >= 0.5
    cad = np.asarray(cad, bool)
    inner = np.zeros_like(cad)
    inner[win.sl] = True
    cad_lab = sk_label(cad, connectivity=1)
    img_lab = sk_label(img, connectivity=1)
    sites: list[dict] = []
    if d0 < 0 or d1 > 0:                      # bridge-type: extra conductor joining CAD conductors (or a loop)
        extra = sk_label(img & ~cad, connectivity=1)
        for i in range(1, int(extra.max()) + 1):
            b = extra == i
            if b.sum() < MIN_BLOB or not (b & inner).any():
                continue
            if _contacts(b, cad_lab) >= 2 or (d1 > 0 and _contact_regions(b, cad) >= 2):
                sites.append({"type": "join", "box": _box(b), "area": int(b.sum())})
    if d0 > 0 or d1 < 0:                      # open-type: missing conductor cutting a CAD conductor; particle
        missing = sk_label(cad & ~img, connectivity=1)
        for i in range(1, int(missing.max()) + 1):
            b = missing == i
            if b.sum() < MIN_BLOB or not (b & inner).any():
                continue
            if _contacts(b, img_lab) >= 2 or (d1 < 0 and _contact_regions(b, img) >= 2):
                sites.append({"type": "cut", "box": _box(b), "area": int(b.sum())})
        for i in range(1, int(img_lab.max()) + 1):
            b = img_lab == i
            if b.sum() >= MIN_BLOB and (b & inner).any() and not (b & cad).any():
                sites.append({"type": "particle", "box": _box(b), "area": int(b.sum())})
    sites = sorted(sites, key=lambda s: -s["area"])[:MAX_SITES]
    m = win.margin
    e = np.asarray(energy)
    r, c = np.unravel_index(int(np.argmax(e)), e.shape)
    return {"applicable": bool(d0 or d1), "sites": sites, "agrees_with_delta_beta": (bool(sites) if (d0 or d1) else None),
            "energy_peak": {"row": int(r) + m, "col": int(c) + m, "value": float(e[r, c])}, "method": LOCATOR_METHOD}
