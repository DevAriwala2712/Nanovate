"""Module 4: Topology verdict and persistent homology arbiter (AGENTS.md §5.2).

Performs cubical persistent homology on a conductor-probability field P in [0, 1].
Sublevel filtration on F = 1.0 - P with the vertex construction:
  - Conductor enters as 1 - P increases from 0 (superlevel set on P);
  - Conductor is 4-connected (diagonal corner touches do not short);
  - Dielectric is 8-connected dual (enclosed holes are 1-cycles with lifetime >= tau).

Verdicts:
  d_beta0 < 0 or d_beta1 > 0 -> KILLER_BRIDGE
  d_beta0 > 0 or d_beta1 < 0 -> KILLER_OPEN/PARTICLE
  d_beta == (0, 0) and bottleneck >= d* and SDF >= e* -> NEAR_MISS
  else -> NUISANCE
"""
from __future__ import annotations

import gudhi
import numpy as np

from ..config import SEMCfg
from ..topo_utils import Window, relevance_region
from .fields import build_field
from .localpers import local_persistence

# Verdict strings
KILLER_BRIDGE = "KILLER_BRIDGE"
KILLER_OPEN_PARTICLE = "KILLER_OPEN/PARTICLE"
NEAR_MISS = "NEAR_MISS"
NUISANCE = "NUISANCE"


def compute_persistence_intervals(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute H0 and H1 persistence intervals for conductor probability field in [0, 1].

    Filtration F = 1.0 - field on vertices. Conductor appears at F = 0.
    Returns (diag0, diag1) as float arrays of shape (N, 2).
    """
    f = np.clip(1.0 - field, 0.0, 1.0).astype(float)
    cc = gudhi.CubicalComplex(vertices=f)
    cc.compute_persistence()
    i0 = cc.persistence_intervals_in_dimension(0)
    i1 = cc.persistence_intervals_in_dimension(1)
    diag0 = np.empty((0, 2), dtype=float) if len(i0) == 0 else np.asarray(i0, dtype=float)
    diag1 = np.empty((0, 2), dtype=float) if len(i1) == 0 else np.asarray(i1, dtype=float)
    return diag0, diag1


def betti_from_intervals(diag0: np.ndarray, diag1: np.ndarray, tau: float) -> tuple[int, int]:
    """Extract Betti numbers (beta_0, beta_1) from intervals with lifetime >= tau.

    For dimension 0, intervals with death=inf have infinite lifetime >= tau.
    """
    b0 = sum(d - b >= tau for b, d in diag0)
    b1 = sum(d - b >= tau for b, d in diag1)
    return int(b0), int(b1)


def compute_bottleneck_distances(
    diag0_img: np.ndarray,
    diag1_img: np.ndarray,
    diag0_ref: np.ndarray,
    diag1_ref: np.ndarray,
) -> tuple[float, float]:
    """Compute bottleneck distances for H0 and H1 between image and reference diagrams."""
    d0 = float(gudhi.bottleneck_distance(diag0_img, diag0_ref))
    d1 = float(gudhi.bottleneck_distance(diag1_img, diag1_ref))
    return d0, d1


def classify_verdict(
    d_beta: tuple[int, int],
    d_bottleneck: float = 0.0,
    sdf_energy: float = 0.0,
    d_star: float = float("inf"),
    e_star: float = float("inf"),
) -> str:
    """Classify defect verdict according to the AGENTS.md §5.2 ladder.

    Precedence:
      1. Killer Bridge (d0 < 0 or d1 > 0)
      2. Killer Open / Particle (d0 > 0 or d1 < 0)
      3. Near-miss (d_beta == 0 and bottleneck >= d* and sdf_energy >= e*)
      4. Nuisance
    """
    d0, d1 = d_beta
    if d0 < 0 or d1 > 0:
        return KILLER_BRIDGE
    if d0 > 0 or d1 < 0:
        return KILLER_OPEN_PARTICLE
    if d0 == 0 and d1 == 0:
        if d_bottleneck >= d_star and sdf_energy >= e_star:
            return NEAR_MISS
        return NUISANCE
    return KILLER_OPEN_PARTICLE


def verify_topology(
    image: np.ndarray,
    cad: np.ndarray,
    win: Window | None = None,
    *,
    tau: float = 0.45,
    field_option: str = "calibrated",
    levels: tuple[float, float] | None = None,
    sem_cfg: SEMCfg | None = None,
    pre_smooth: float = 0.0,
    d_star: float = float("inf"),
    e_star: float = float("inf"),
    sdf_energy: float | None = None,
    singleton: bool = True,
    region_tol: int = 4,
    region_mode: str = "splice",
) -> tuple[str, dict]:
    """Verify topology of SEM patch against CAD reference (AGENTS.md §5.2).

    Args:
        image: 2D array of SEM counts (e-/px).
        cad: 2D bool array of binary CAD.
        win: Optional central window. If provided, both image and CAD are cropped to win.
        tau: Lifetime threshold for persistent features (default 0.45).
        field_option: 'calibrated' or 'percentile'.
        levels: (mu_diel, mu_cond) for calibrated field.
        sem_cfg: optional SEMCfg for oracle levels.
        pre_smooth: optional Gaussian pre-smoothing std (px).
        d_star: Bottleneck distance threshold for NEAR_MISS (default inf).
        e_star: SDF tension energy threshold for NEAR_MISS (default inf).
        sdf_energy: Computed SDF energy (from Module 3; None until M4 lands).
        singleton: True if patch is evaluated without K peers.
        region_tol: px by which the relevance region keeps clear of CAD components outside
            the window; should exceed optical reach (blur + bloom) plus residual shift.
        region_mode: 'splice' (field replaced by CAD outside the region, GUDHI on the tile) or
            'pairs' (no splice: region-local union-find persistence counts only features whose
            component reaches the region; needs ``win``).

    Returns:
        (verdict, reason_dict)
    """
    # 1. Build conductor probability field
    field, meta = build_field(
        image,
        option=field_option,
        levels=levels,
        sem_cfg=sem_cfg,
        pre_smooth=pre_smooth,
    )

    # 2-5. Margin handling (AGENTS.md §7.5) and Betti / bottleneck computation.
    if win is not None and region_mode == "pairs":
        reg = relevance_region(cad, win, tol=region_tol)
        lp_img = local_persistence(field, reg)
        lp_ref = local_persistence(cad.astype(float), reg)
        beta_img = lp_img.betti(tau)
        beta_ref = lp_ref.betti(tau)
        (d0_i, d1_i), (d0_r, d1_r) = lp_img.diagrams(), lp_ref.diagrams()
        bot0, bot1 = compute_bottleneck_distances(d0_i, d1_i, d0_r, d1_r)
    else:
        if region_mode not in ("splice", "pairs"):
            raise ValueError(f"unknown region_mode {region_mode!r}")
        if win is not None:
            # Field replaced by the CAD outside the label's relevance region, so wire
            # fragments created at the window edge cannot add or remove components.
            reg = relevance_region(cad, win, tol=region_tol)
            f_eval = np.where(reg, field, cad.astype(float))
        else:
            f_eval = field
        diag0_img, diag1_img = compute_persistence_intervals(f_eval)
        diag0_ref, diag1_ref = compute_persistence_intervals(cad.astype(float))
        beta_img = betti_from_intervals(diag0_img, diag1_img, tau)
        beta_ref = betti_from_intervals(diag0_ref, diag1_ref, tau)
        bot0, bot1 = compute_bottleneck_distances(diag0_img, diag1_img, diag0_ref, diag1_ref)
    delta_beta = (beta_img[0] - beta_ref[0], beta_img[1] - beta_ref[1])
    bot_max = max(bot0, bot1)

    # 6. Verdict determination
    near_miss_enabled = sdf_energy is not None and not np.isinf(d_star) and not np.isinf(e_star)
    eff_energy = 0.0 if sdf_energy is None else sdf_energy
    verdict = classify_verdict(delta_beta, bot_max, eff_energy, d_star, e_star)

    reason = {
        "verdict": verdict,
        "delta_betti": delta_beta,
        "beta_ref": beta_ref,
        "beta_img": beta_img,
        "bottleneck_h0": bot0,
        "bottleneck_h1": bot1,
        "bottleneck_max": bot_max,
        "max_sdf_energy": sdf_energy,
        "near_miss_enabled": near_miss_enabled,
        "tau": tau,
        "d_star": d_star,
        "e_star": e_star,
        "singleton": singleton,
        "region_mode": region_mode,
        "region_tol": region_tol,
        "field_meta": meta,
    }
    return verdict, reason
