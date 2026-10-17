"""End-to-end verdict for one patch (singleton) or one same-layout stack (AGENTS.md §4, §5).

Module 1 (calibration) is fitted once per layer from the scribe patches and passed in as ``cal``.
Per patch: Module 3 alignment to the calibrated template -> (Module 2 low-rank denoise when a stack
of same-layout peers exists, else per-patch Gaussian pre-smoothing and a ``singleton=True`` flag)
-> Module 3 SDF tension energy -> Module 4 topology verdict. Every result carries its reason
(Delta-beta, bottleneck distances, max SDF energy) and per-module wall times in ms (CPU).

The configuration is a frozen dataclass so one run's settings can be saved next to its results.
Units: images e-/px, lengths px (= nm at 1 nm/px).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .modules.align_sdf import align_to_template, energy_map, peak_z
from .modules.calibration import Calibration, features
from .modules.lowrank import denoise_stack
from .modules.topology import verify_topology
from .topo_utils import Window


@dataclass(frozen=True)
class PipelineCfg:
    """Frozen verifier settings (selected on Manhattan val only; see results/calib/decisions.json)."""

    field_option: str = "calibrated"   # 'percentile' | 'calibrated' | 'template'
    tau: float = 0.45                  # persistence lifetime threshold, in field units
    pre_smooth: float = 1.0            # Gaussian sigma (px) before topology (singleton path only)
    region_mode: str = "splice"        # 'splice' | 'pairs'
    region_tol: int = 8                # px clearance of the relevance region from out-of-window wires
    use_stack: bool = False            # Module 2 when peers are available
    align: bool = True                 # Module 3 alignment (False only to measure its value, Exp 4)
    d_star: float = float("inf")       # bottleneck threshold for NEAR_MISS (inf = rule off)
    e_star: float = float("inf")       # SDF-energy threshold for NEAR_MISS (inf = rule off)
    gamma: float = 0.2                 # SDF energy exponent, 1/px (fixed a priori)
    cap: float = 10.0                  # |phi| cap in the energy, px
    h1_fill: str = "cad"               # 'cad' (v1) | 'dielectric' (v2: no splice-seam holes)
    z_star: float = float("-inf")      # peak-z threshold for NEAR_MISS (-inf = v1 rule, no z condition)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineResult:
    verdict: str
    reason: dict
    singleton: bool
    shift: tuple[float, float]
    timings_ms: dict = field(default_factory=dict)


def _topology(field_img, cad, cal, win, cfg, template, energy, singleton, pre_smooth, pz=None):
    kw = {}
    if cfg.field_option == "template":
        kw = dict(template=template, cad_smooth=features(cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
    return verify_topology(
        field_img, cad, win, tau=cfg.tau, field_option=cfg.field_option, levels=cal.levels,
        pre_smooth=pre_smooth, d_star=cfg.d_star, e_star=cfg.e_star, sdf_energy=energy, singleton=singleton,
        region_tol=cfg.region_tol, region_mode=cfg.region_mode, field_kwargs=kw, h1_fill=cfg.h1_fill,
        peak_z=pz, z_star=cfg.z_star,
    )


def verify_patch(image: np.ndarray, cad: np.ndarray, cal: Calibration, win: Window, cfg: PipelineCfg = PipelineCfg()) -> PipelineResult:
    """Singleton path: one patch, no same-layout peers (flagged ``singleton=True``)."""
    t = {}
    t0 = time.perf_counter()
    template = cal.predict(cad)
    aligned, shift = align_to_template(image, template) if cfg.align else (np.asarray(image, dtype=float), (0.0, 0.0))
    t["align"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    emap = energy_map(image, cad, cal, win, cfg.gamma, cfg.cap, aligned=aligned)
    energy, pz = float(emap.max()), peak_z(emap)
    t["sdf_energy"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    verdict, reason = _topology(aligned, cad, cal, win, cfg, template, energy, True, cfg.pre_smooth, pz)
    t["topology"] = (time.perf_counter() - t0) * 1e3
    return PipelineResult(verdict, reason, True, shift, t)


def verify_stack(images: list[np.ndarray], cad: np.ndarray, cal: Calibration, win: Window, cfg: PipelineCfg = PipelineCfg(), seed: int = 0) -> list[PipelineResult]:
    """K same-layout dies: align, low-rank denoise (Module 2), then the same energy + topology per die.

    A stack of one die falls back to the singleton denoiser (flagged ``singleton=True``).
    """
    if len(images) == 1:
        res = denoise_stack(images, cad, cal, seed=seed)
        r = verify_patch(res.field[0], cad, cal, win, PipelineCfg(**{**cfg.to_dict(), "align": cfg.align, "pre_smooth": 0.0}))
        r.reason["singleton"] = True
        return [r]
    t0 = time.perf_counter()
    res = denoise_stack(images, cad, cal, seed=seed, align=cfg.align)
    stack_ms = (time.perf_counter() - t0) * 1e3 / len(images)
    template = cal.predict(cad)
    out = []
    for k in range(len(images)):
        t = {"stack": stack_ms}
        t0 = time.perf_counter()
        emap = energy_map(images[k], cad, cal, win, cfg.gamma, cfg.cap, aligned=res.field[k])
        energy, pz = float(emap.max()), peak_z(emap)
        t["sdf_energy"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        verdict, reason = _topology(res.field[k], cad, cal, win, cfg, template, energy, False, 0.0, pz)
        t["topology"] = (time.perf_counter() - t0) * 1e3
        out.append(PipelineResult(verdict, reason, False, res.shifts[k], t))
    return out
