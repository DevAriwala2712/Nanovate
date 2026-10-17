"""Sample / stack generation.

Random streams are split so experiments can re-acquire the *same* wafer:
  geometry stream  key = ("geom", family, split, layout_id, die)   -> LER, grain, defect
  acquisition key  = ("acq",  family, split, layout_id, die, tag)  -> shift, ramp, noise
Sweeping dose or shift with the same keys therefore gives paired samples.

Saved per sample: CAD, W, image, mask, label, params (AGENTS.md §5.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SEMCfg, TestbedCfg
from ..rng import child_rng
from ..topo_utils import Window
from .labels import LABEL_OF_KIND, consistent, topology_label
from .layouts import Layout, make_layout
from .perturb import DEFECTS, apply_ler, grain_field
from .sem import acquire, charging_ramp, expected_image, warp_field

KINDS = ("nuisance", "bridge", "open", "particle", "notch", "filament")


@dataclass
class WaferDie:
    """Ground-truth wafer for one die (canvas resolution)."""

    kind: str
    W_canvas: np.ndarray
    grain_canvas: np.ndarray
    mask_canvas: np.ndarray
    topo: dict
    params: dict
    resamples: int
    resample_reasons: dict = field(default_factory=dict)


@dataclass
class Sample:
    """One SEM patch with its ground truth (tile resolution, 1 nm/px)."""

    family: str
    split: str
    layout_id: int
    die: int
    kind: str
    label: str            # NUISANCE / NEAR_MISS / KILLER
    topo: dict            # exact d_beta of W vs CAD on the central window
    cad: np.ndarray       # bool (tile, tile)
    W: np.ndarray         # bool (tile, tile)
    mask: np.ndarray      # bool, pixels changed by the defect
    image: np.ndarray     # float32 counts, e-/px
    expected: np.ndarray  # float32 noise-free mean counts, e-/px
    shift: tuple[float, float]
    dose: float
    params: dict
    resamples: int

    @property
    def killer(self) -> bool:
        return self.label == "KILLER"


def window(cfg: TestbedCfg) -> Window:
    return Window(cfg.layout.tile, cfg.layout.margin)


def get_layout(family: str, split: str, layout_id: int, cfg: TestbedCfg, seed: int) -> Layout:
    return make_layout(family, cfg.layout, child_rng(seed, "layout", family, split, layout_id))


def make_wafer(layout: Layout, kind: str, cfg: TestbedCfg, rng: np.random.Generator) -> WaferDie:
    """Perturb CAD -> W (LER + optional defect); resample until topology matches ``kind``."""
    if kind not in KINDS:
        raise ValueError(kind)
    win = window(cfg)
    p, t = layout.pad, layout.tile
    reasons: dict[str, int] = {}
    for attempt in range(cfg.defect.max_resample + 1):
        W = apply_ler(layout.polygons, layout.N, cfg.nuisance, rng)
        if topology_label(W[p:p + t, p:p + t], layout.cad, win)["topo"] != "none":
            # LER alone changed topology (AGENTS.md §7.6): never let it leak into a label
            reasons["ler_topology"] = reasons.get("ler_topology", 0) + 1
            continue
        mask = np.zeros_like(W)
        dparams: dict = {}
        if kind != "nuisance":
            res = DEFECTS[kind](W, rng, cfg.defect, cfg.layout)
            if res is None:
                reasons["no_site"] = reasons.get("no_site", 0) + 1
                continue
            W, mask, dparams = res
        topo = topology_label(W[p:p + t, p:p + t], layout.cad, win)
        if consistent(kind, topo):
            grain = grain_field(layout.N, cfg.nuisance, rng)
            params = {"defect": dparams, "ler_sigma": cfg.nuisance.ler_sigma,
                      "grain_contrast": cfg.nuisance.grain_contrast}
            return WaferDie(kind, W, grain, mask, topo, params, attempt, reasons)
        key = f"topo_{topo['topo']}"
        reasons[key] = reasons.get(key, 0) + 1
    raise RuntimeError(f"{kind}: no topology-consistent sample after {cfg.defect.max_resample} resamples {reasons}")


def render(
    layout: Layout,
    wafer: WaferDie,
    sem: SEMCfg,
    rng: np.random.Generator,
    shift: tuple[float, float] | None = None,
    warp_amp: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Acquire one SEM image of ``wafer``. Returns (image, expected, shift) in e-/px.

    ``warp_amp`` (px) adds a smooth random geometric warp (Exp 4); 0 leaves every random stream and
    output bit-identical to the unwarped render.
    """
    if shift is None:
        shift = tuple(float(v) for v in rng.uniform(-sem.max_shift, sem.max_shift, 2))
    if max(abs(shift[0]), abs(shift[1])) > layout.pad - 4:
        raise ValueError("shift exceeds rendered context (pad)")
    ramp = charging_ramp((layout.tile, layout.tile), sem.charge_amp, rng)
    warp = None
    if warp_amp > 0:  # separate stream: drawing it must not disturb the ramp / noise draws
        warp = warp_field(wafer.W_canvas.shape, warp_amp, np.random.default_rng(int(rng.integers(2**32))))
    I_bar = expected_image(wafer.W_canvas, wafer.grain_canvas, sem, layout.pad, layout.tile, shift, ramp, warp)
    img = acquire(I_bar, sem, rng)
    return img, (sem.dose * I_bar).astype(np.float32), shift


def make_sample(
    family: str,
    split: str,
    layout_id: int,
    die: int,
    kind: str,
    cfg: TestbedCfg,
    seed: int,
    layout: Layout | None = None,
    shift: tuple[float, float] | None = None,
    acq_tag: str = "",
    warp_amp: float = 0.0,
) -> Sample:
    """Generate one labelled sample (deterministic in all arguments)."""
    layout = layout or get_layout(family, split, layout_id, cfg, seed)
    wafer = make_wafer(layout, kind, cfg, child_rng(seed, "geom", family, split, layout_id, die, kind))
    img, exp, sh = render(layout, wafer, cfg.sem, child_rng(seed, "acq", family, split, layout_id, die, acq_tag), shift, warp_amp)
    p, t = layout.pad, layout.tile
    return Sample(
        family, split, layout_id, die, kind, LABEL_OF_KIND[kind], wafer.topo,
        layout.cad.copy(), wafer.W_canvas[p:p + t, p:p + t].copy(), wafer.mask_canvas[p:p + t, p:p + t].copy(),
        img, exp, sh, cfg.sem.dose, wafer.params | {"resample_reasons": wafer.resample_reasons}, wafer.resamples,
    )


def make_stack(
    family: str, split: str, layout_id: int, kinds: list[str], cfg: TestbedCfg, seed: int
) -> list[Sample]:
    """K same-layout dies (one 'die stack'); ``kinds[k]`` sets die k's content."""
    layout = get_layout(family, split, layout_id, cfg, seed)
    return [make_sample(family, split, layout_id, k, kind, cfg, seed, layout=layout) for k, kind in enumerate(kinds)]
