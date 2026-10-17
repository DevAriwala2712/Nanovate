"""Demo: walk one synthetic tile through every stage and show the verdict with its reason.

``notebooks/demo.ipynb`` is a thin caller of this module (jupyter is not an allowed dependency, so the
logic lives here and is tested; the notebook's code cells are executed by ``tests/test_demo.py``).
Stages: testbed sample -> Module 1 calibration (scribe patches) -> Module 3 alignment + SDF energy ->
conductor field -> Module 4 persistence + verdict. Synthetic data.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .config import TestbedCfg
from .modules.align_sdf import align_to_template, cad_sdf, tension_energy
from .modules.calibration import Calibration, calibrate, features
from .modules.fields import build_field
from .modules.topology import compute_persistence_intervals
from .pipeline import PipelineCfg, PipelineResult, verify_patch
from .testbed.dataset import Sample, make_sample, window
from .testbed.scribe import make_scribe_patches


def load_frozen_cfg(path: str | Path = "results/v2/calib/decisions.json") -> PipelineCfg:
    """Verifier config frozen on Manhattan val by experiments/select_config.py (default: v2; default config if absent)."""
    p = Path(path)
    if not p.exists():
        return PipelineCfg()
    d = json.loads(p.read_text())["pipeline_cfg"]
    d = {k: (float("inf") if v is None and k in ("d_star", "e_star") else v) for k, v in d.items()}
    return PipelineCfg(**{k: (float("-inf") if v is None and k == "z_star" else v) for k, v in d.items()})


@dataclass
class Walk:
    """Every intermediate of one tile through the pipeline (arrays are tile-sized, CAD frame)."""

    kind: str
    sample: Sample
    cal: Calibration
    pc: PipelineCfg
    template: np.ndarray        # Module 1 predicted image T(C), e-/px
    aligned: np.ndarray         # Module 3: image shifted onto the template
    shift_est: tuple            # (dy, dx) applied by the aligner, px
    field: np.ndarray           # conductor probability P in [0, 1] passed to Module 4
    energy_map: np.ndarray      # Module 3 SDF tension energy per pixel
    result: PipelineResult      # verdict + reason + timings from the real pipeline
    diag_img_h0: np.ndarray     # persistence diagrams (birth, death) of the field and of the CAD
    diag_img_h1: np.ndarray
    diag_cad_h0: np.ndarray
    diag_cad_h1: np.ndarray


def _finite(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1, 2)
    return np.where(np.isinf(d), 1.0, d)


def walk_through(
    kind: str, family: str = "manhattan", dose: float = 50.0, layout_id: int = 0, die: int = 0, seed: int = 0,
    shift: tuple | None = None, pc: PipelineCfg | None = None, split: str = "test",
) -> Walk:
    """Generate one sample of ``kind`` and run every stage, keeping the intermediates."""
    base = TestbedCfg()
    tcfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
    pc = pc or load_frozen_cfg()
    s = make_sample(family, split, layout_id, die, kind, tcfg, seed, shift=shift)
    cal = calibrate(make_scribe_patches(tcfg, seed=seed))
    template = cal.predict(s.cad)
    aligned, sh = align_to_template(s.image, template)
    kw = {}
    if pc.field_option == "template":
        kw = dict(template=template, cad_smooth=features(s.cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
    field, _ = build_field(aligned, pc.field_option, levels=cal.levels, pre_smooth=pc.pre_smooth, **kw)
    contrast = abs(cal.mu_cond - cal.mu_diel)
    energy_map = tension_energy(ndi.gaussian_filter(aligned - template, 1.0) / contrast, cad_sdf(s.cad), pc.gamma, pc.cap)
    result = verify_patch(s.image, s.cad, cal, window(tcfg), pc)
    d0, d1 = compute_persistence_intervals(field)
    c0, c1 = compute_persistence_intervals(s.cad.astype(float))
    return Walk(kind, s, cal, pc, template, aligned, sh, field, energy_map, result, _finite(d0), _finite(d1), _finite(c0), _finite(c1))


def describe(w: Walk) -> str:
    """Plain-text account of the verdict and the reason it carries."""
    r, s, c = w.result.reason, w.sample, w.cal
    nm = "off" if np.isinf(w.pc.d_star) else f"on (d* {w.pc.d_star:.3g}, e* {w.pc.e_star:.3g})"
    return "\n".join([
        f"Synthetic {s.family} tile, injected kind '{w.kind}' (ground-truth label {s.label}), dose {s.dose:g} e-/px.",
        f"Acquisition shift (applied) ({s.shift[0]:+.1f}, {s.shift[1]:+.1f}) px; Module 3 recovered ({w.shift_est[0]:+.1f}, {w.shift_est[1]:+.1f}) px.",
        f"Module 1 levels: dielectric {c.mu_diel:.1f}, conductor {c.mu_cond:.1f} e-/px; noise gain {c.alpha:.2f}.",
        f"Field '{w.pc.field_option}', persistence lifetime threshold tau = {w.pc.tau}, near-miss rule {nm}.",
        f"Path: {'singleton (no same-layout peers; Gaussian pre-smoothing)' if w.result.singleton else 'stack'} -> singleton={w.result.singleton}.",
        f"beta(image) = {tuple(r['beta_img'])}, beta(CAD) = {tuple(r['beta_ref'])}, delta-beta = {tuple(r['delta_betti'])}.",
        f"Bottleneck distance H0 {r['bottleneck_h0']:.3f}, H1 {r['bottleneck_h1']:.3f}; max SDF tension energy {r['max_sdf_energy']:.2f}.",
        f"VERDICT: {w.result.verdict}",
        "(synthetic data; the verdict comes from topology, not from pixel differences)",
    ])


def plot_walk_through(w: Walk):
    """Figure: CAD, ground-truth wafer, SEM, template, aligned, field, energy, two persistence diagrams, text."""
    import matplotlib.pyplot as plt

    s = w.sample
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.18)
    panels = [("CAD", s.cad, "gray"), ("ground-truth wafer W\n(labels come from this)", s.W, "gray"),
              (f"SEM image (dose {s.dose:g})", s.image, "gray"), ("Module 1: template T(C)", w.template, "gray"),
              ("Module 3: aligned to T(C)", w.aligned, "gray"), (f"field P ({w.pc.field_option})", w.field, "viridis"),
              ("Module 3: SDF tension energy", w.energy_map, "magma")]
    for i, (t, im, cm) in enumerate(panels):
        a = fig.add_subplot(gs[i // 4, i % 4])
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=9); a.axis("off")
    for pos, (t, di, dc) in zip([(1, 3), (2, 0)], [("H0 persistence (components)", w.diag_img_h0, w.diag_cad_h0),
                                                   ("H1 persistence (holes)", w.diag_img_h1, w.diag_cad_h1)]):
        a = fig.add_subplot(gs[pos])
        a.scatter(dc[:, 0], dc[:, 1], s=16, c="C1", label="CAD", alpha=0.7)
        a.scatter(di[:, 0], di[:, 1], s=14, c="C0", marker="x", label="image field")
        a.plot([0, 1], [0, 1], "k:", lw=0.8); a.set_title(t, fontsize=9); a.set_xlabel("birth", fontsize=8); a.set_ylabel("death", fontsize=8)
        a.legend(fontsize=7)
    t = fig.add_subplot(gs[2, 1:3]); t.axis("off")
    t.text(0.0, 1.0, describe(w), va="top", ha="left", family="monospace", fontsize=7)
    fig.suptitle("Nanovate demo: one synthetic tile through every stage", fontsize=11)
    return fig
