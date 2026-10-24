"""Phase 4 reference-die mode: a clean neighbouring die replaces CAD as the structural reference (no CAD used).

docs/PHASE4_REFDIE_PREREG.md fixes every choice a priori: reference dies are aligned to the first one (sub-pixel phase
correlation) and averaged; the mean is Gaussian-smoothed (sigma 1.5 px), thresholded at Otsu (conductor = brighter),
and specks / holes under 10 px are removed. That binary pseudo-CAD then plays CAD's role everywhere: Module 1 is fitted
on (pseudo-CAD, each aligned reference image) and the frozen v3 verifier runs against it. Units: px, e-/px.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .modules.calibration import Calibration, calibrate, estimate_shift
from .modules.events import EventRuleCfg
from .pipeline import PipelineCfg, PipelineResult
from .pipeline_v3 import verify_patch_v3
from .topo_utils import Window

SMOOTH_PX = 1.5
MIN_AREA_PX = 10


@dataclass(frozen=True)
class Reference:
    cad: np.ndarray            # bool pseudo-CAD in the frame of the first reference die
    images: tuple              # reference images aligned to the first one
    shifts: tuple              # (dy, dx) applied to each reference image, px
    build_ms: float


def build_reference(ref_images: list[np.ndarray], smooth: float = SMOOTH_PX, min_area: int = MIN_AREA_PX) -> Reference:
    """Pseudo-CAD from one or more clean reference dies (pre-registered rule)."""
    t0 = time.perf_counter()
    base = np.asarray(ref_images[0], float)
    imgs, shifts = [base], [(0.0, 0.0)]
    for im in ref_images[1:]:
        d = estimate_shift(reference=base, moving=np.asarray(im, float))
        imgs.append(ndi.shift(np.asarray(im, float), d, order=1, mode="nearest"))
        shifts.append(d)
    sm = ndi.gaussian_filter(np.mean(imgs, axis=0), smooth)
    cad = sm > threshold_otsu(sm)
    cad = remove_small_objects(cad, min_size=min_area)
    cad = remove_small_holes(cad, area_threshold=min_area)
    return Reference(cad, tuple(imgs), tuple(shifts), (time.perf_counter() - t0) * 1e3)


def calibrate_from_reference(ref: Reference) -> Calibration:
    """Module 1 on (pseudo-CAD, aligned reference image) pairs instead of scribe patches."""
    return calibrate([(ref.cad.astype(float), im) for im in ref.images])


def verify_refdie(target: np.ndarray, ref_images: list[np.ndarray], win: Window, pc: PipelineCfg, rc: EventRuleCfg,
                  oracle_cad: np.ndarray | None = None) -> tuple[PipelineResult, Reference, Calibration]:
    """Frozen v3 verdict against a reference built from ``ref_images`` (or ``oracle_cad``, the diagnostic arm)."""
    if oracle_cad is None:
        ref = build_reference(ref_images)
    else:
        ref = Reference(np.asarray(oracle_cad, bool), tuple(np.asarray(im, float) for im in ref_images[:1]), ((0.0, 0.0),), 0.0)
    if not ref.cad.any() or ref.cad.all():
        raise ValueError("degenerate pseudo-CAD (all one class)")
    cal = calibrate_from_reference(ref)
    return verify_patch_v3(target, ref.cad, cal, win, pc, rc), ref, cal
