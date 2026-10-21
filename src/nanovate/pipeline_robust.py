"""Rung 1b robustness candidates on top of the frozen v3 rule (docs/REAL_DATA_RUNG1B_PREREG.md, Part A).

C1 calibration guard: leave-one-scribe-patch-out contrast consistency -> verdict UNVERIFIED_CALIBRATION if it fails.
C2 template clamp:    T(C) clipped to [q0.5, q99.5] of the scribe-patch intensities (every module sees the clamped T).
C3 bounded alignment: a Module-3 shift above ``max_shift`` px on either axis -> shift (0, 0), flagged.

``pipeline.py`` and ``pipeline_v3.py`` are not modified; with every candidate off, ``verify_patch_robust`` equals
``verify_patch_v3`` (tested). Units: images e-/px (or grey levels on real data), lengths px.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass

import numpy as np

from .modules.align_sdf import align_to_template, energy_map, peak_z
from .modules.calibration import Calibration, calibrate, features
from .modules.events import EventRuleCfg, decide, evidence, v2_delta_beta
from .modules.fields import build_field
from .modules.topology import _reference, compute_bottleneck_distances
from .pipeline import PipelineCfg, PipelineResult
from .topo_utils import Window

UNVERIFIED = "UNVERIFIED_CALIBRATION"


@dataclass(frozen=True)
class RobustCfg:
    guard: bool = False                          # C1
    guard_rel_spread: float = 0.5                # C1: max |c_loo - c| / |c|
    clamp: tuple[float, float] | None = None     # C2: percentiles, e.g. (0.5, 99.5)
    max_shift: float | None = None               # C3: px per axis, e.g. 12.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ClampedCalibration(Calibration):
    """Calibration whose template is clipped to [lo, hi] (C2). lo/hi = -inf/inf is a plain Calibration."""

    lo: float = -np.inf
    hi: float = np.inf

    def predict(self, cad: np.ndarray) -> np.ndarray:
        return np.clip(super().predict(cad), self.lo, self.hi)


@dataclass(frozen=True)
class RobustCalibration:
    cal: Calibration
    reliable: bool
    contrast: float
    loo_contrasts: tuple
    clamp_range: tuple | None


def calibrate_robust(patches: list[tuple[np.ndarray, np.ndarray]], rcfg: RobustCfg, **kw) -> RobustCalibration:
    """Module 1 plus the C1 reliability check and the C2 clamp range (both from the scribe patches only)."""
    cal = calibrate(patches, **kw)
    c = cal.mu_cond - cal.mu_diel
    loo = []
    if len(patches) >= 2:
        for k in range(len(patches)):
            ck = calibrate([p for i, p in enumerate(patches) if i != k], **kw)
            loo.append(float(ck.mu_cond - ck.mu_diel))
    reliable = bool(loo) and all(np.sign(x) == np.sign(c) and abs(x - c) <= rcfg.guard_rel_spread * abs(c) for x in loo)
    rng = None
    if rcfg.clamp is not None:
        ys = np.concatenate([np.asarray(im, float).ravel() for _, im in patches])
        rng = (float(np.percentile(ys, rcfg.clamp[0])), float(np.percentile(ys, rcfg.clamp[1])))
        cal = ClampedCalibration(**{f.name: getattr(cal, f.name) for f in dataclasses.fields(Calibration)}, lo=rng[0], hi=rng[1])
    return RobustCalibration(cal, reliable, float(c), tuple(loo), rng)


def verify_patch_robust(image: np.ndarray, cad: np.ndarray, rcal: RobustCalibration, win: Window, pc: PipelineCfg,
                        rc: EventRuleCfg, rcfg: RobustCfg) -> PipelineResult:
    """v3 singleton verdict with the enabled candidates; reason carries the unguarded verdict and every flag."""
    cal = rcal.cal
    t = {}
    t0 = time.perf_counter()
    template = cal.predict(cad)
    aligned, shift = align_to_template(image, template) if pc.align else (np.asarray(image, float), (0.0, 0.0))
    align_rejected = False
    if rcfg.max_shift is not None and max(abs(shift[0]), abs(shift[1])) > rcfg.max_shift:
        aligned, shift, align_rejected = np.asarray(image, float), (0.0, 0.0), True
    t["align"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    emap = energy_map(image, cad, cal, win, pc.gamma, pc.cap, aligned=aligned)
    energy, pz = float(emap.max()), peak_z(emap)
    t["sdf_energy"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    kw = dict(levels=cal.levels, pre_smooth=pc.pre_smooth, cad_smooth=features(cad, *cal.sigmas)[..., 1],
              contrast=cal.mu_cond - cal.mu_diel)
    fimg, _ = build_field(aligned, option="template", template=template, **kw)
    fexp, _ = build_field(template, option="template", template=template, **kw)
    reg, d0r, d1r = _reference(cad, win, pc.region_tol, "dielectric")
    ev = evidence(fimg, cad, reg, fexp, rc.pair_dist, rc.body_radius)
    F = lambda a: np.c_[1 - a[:, 0], np.where(np.isnan(a[:, 1]), np.inf, 1 - a[:, 1])] if len(a) else np.empty((0, 2))  # noqa: E731
    d0i = F(ev.h0[:, :2]) if len(ev.h0) else np.empty((0, 2))
    d1i = F(ev.h1[:, :2]) if len(ev.h1) else np.empty((0, 2))
    b0, b1 = compute_bottleneck_distances(d0i, d1i, d0r, d1r)
    bott = max(b0, b1)
    verdict, events = decide(ev, rc, pc.tau, bott, energy, pz, pc.d_star, pc.e_star, pc.z_star)
    t["topology"] = (time.perf_counter() - t0) * 1e3
    unguarded = verdict
    if rcfg.guard and not rcal.reliable:
        verdict = UNVERIFIED
    reason = dict(verdict=verdict, verdict_unguarded=unguarded, rule=rc.rule, events=events, n_bridge_events=len(events["bridge"]),
                  n_open_events=len(events["open"]), delta_betti=v2_delta_beta(ev, pc.tau), bottleneck_max=bott,
                  max_sdf_energy=energy, peak_z=pz, singleton=True, align_rejected=align_rejected,
                  calibration_reliable=rcal.reliable, contrast=rcal.contrast, loo_contrasts=list(rcal.loo_contrasts),
                  clamp_range=rcal.clamp_range, robust_cfg=rcfg.to_dict(), rule_cfg=rc.to_dict())
    return PipelineResult(verdict, reason, True, shift, t)
