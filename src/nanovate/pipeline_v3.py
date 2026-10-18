"""v3 end-to-end verdict for one patch: v2's Modules 1-3 and field, then the located-event rule (modules/events.py).

``pipeline.py`` (v1/v2) is not modified. With ``EventRuleCfg(rule="v2_global_tau")`` this calls
``pipeline.verify_patch`` unchanged, so v2 is reproduced exactly. Singleton path only (as in v2's experiments).
Units: images e-/px, lengths px (= nm at 1 nm/px).
"""
from __future__ import annotations

import time

import numpy as np

from .modules.align_sdf import align_to_template, energy_map, peak_z
from .modules.calibration import Calibration, features
from .modules.events import EventRuleCfg, Evidence, decide, evidence, v2_delta_beta
from .modules.fields import build_field
from .modules.topology import _reference, compute_bottleneck_distances
from .pipeline import PipelineCfg, PipelineResult, verify_patch
from .topo_utils import Window


def patch_evidence(image: np.ndarray, cad: np.ndarray, cal: Calibration, win: Window, pc: PipelineCfg,
                   rc: EventRuleCfg = EventRuleCfg()) -> tuple[Evidence, dict, dict]:
    """(evidence, scalars {bottleneck, energy, peak_z, delta_betti_v2}, timings ms) for one singleton patch."""
    if (pc.field_option, pc.region_mode, pc.h1_fill) != ("template", "splice", "dielectric"):
        raise ValueError("the v3 rule is defined on the v2 field (template, splice, dielectric holes)")
    t = {}
    t0 = time.perf_counter()
    template = cal.predict(cad)
    aligned, shift = align_to_template(image, template) if pc.align else (np.asarray(image, float), (0.0, 0.0))
    t["align"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    emap = energy_map(image, cad, cal, win, pc.gamma, pc.cap, aligned=aligned)
    energy, pz = float(emap.max()), peak_z(emap)
    t["sdf_energy"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    kw = dict(levels=cal.levels, pre_smooth=pc.pre_smooth, cad_smooth=features(cad, *cal.sigmas)[..., 1],
              contrast=cal.mu_cond - cal.mu_diel)
    fimg, _ = build_field(aligned, option="template", template=template, **kw)
    fexp, _ = build_field(template, option="template", template=template, **kw)   # defect-free: zero residual
    reg, d0r, d1r = _reference(cad, win, pc.region_tol, "dielectric")
    ev = evidence(fimg, cad, reg, fexp, rc.pair_dist, rc.body_radius)
    F = lambda a: np.c_[1 - a[:, 0], np.where(np.isnan(a[:, 1]), np.inf, 1 - a[:, 1])] if len(a) else np.empty((0, 2))  # noqa: E731
    d0i = F(ev.h0[:, :2]) if len(ev.h0) else np.empty((0, 2))
    d1i = F(ev.h1[:, :2]) if len(ev.h1) else np.empty((0, 2))
    b0, b1 = compute_bottleneck_distances(d0i, d1i, d0r, d1r)
    t["topology"] = (time.perf_counter() - t0) * 1e3
    return ev, dict(bottleneck=max(b0, b1), energy=energy, peak_z=pz, delta_betti_v2=v2_delta_beta(ev, pc.tau), shift=shift), t


def verify_patch_v3(image: np.ndarray, cad: np.ndarray, cal: Calibration, win: Window, pc: PipelineCfg,
                    rc: EventRuleCfg = EventRuleCfg()) -> PipelineResult:
    """Singleton verdict with the v3 event rule; ``rc.rule == 'v2_global_tau'`` is exactly ``verify_patch``."""
    if rc.rule == "v2_global_tau":
        return verify_patch(image, cad, cal, win, pc)
    ev, sc, t = patch_evidence(image, cad, cal, win, pc, rc)
    verdict, events = decide(ev, rc, pc.tau, sc["bottleneck"], sc["energy"], sc["peak_z"], pc.d_star, pc.e_star, pc.z_star)
    reason = dict(verdict=verdict, rule=rc.rule, events=events, n_bridge_events=len(events["bridge"]), n_open_events=len(events["open"]),
                  delta_betti=sc["delta_betti_v2"], bottleneck_max=sc["bottleneck"], max_sdf_energy=sc["energy"], peak_z=sc["peak_z"],
                  pairs=[{k: v for k, v in p.items() if k != "pass_rc"} for p in ev.pairs], tau=pc.tau, d_star=pc.d_star,
                  e_star=pc.e_star, z_star=pc.z_star, singleton=True, rule_cfg=rc.to_dict())
    return PipelineResult(verdict, reason, True, sc["shift"], t)
