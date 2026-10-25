"""Reference-cell mode (Track C Rung 3): a standard-cell instance is verified against the median of other instances of
the same cell type and orientation; no polygon CAD is needed (the abstracted GDS only gives cell boxes and types).

For a W x W window around the instance:
  1. reference instances are aligned to the first one (sub-pixel phase correlation) and their pixelwise MEDIAN taken
     (robust to one modified instance among the references);
  2. pseudo-CAD = Otsu threshold of the smoothed median (``refdie.build_reference`` rule: sigma 1.5 px, 10 px cleanup);
  3. the target window is aligned to the median and SPLICED: target pixels inside the cell box, median pixels outside,
     so neighbouring cells (which differ between instances) cannot create differences;
  4. Module 1 is calibrated on (pseudo-CAD, aligned reference windows) and the frozen verifier runs on the splice.
Units: px.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .modules.calibration import estimate_shift
from .modules.events import EventRuleCfg
from .pipeline import PipelineCfg, PipelineResult
from .pipeline_robust import RobustCalibration, RobustCfg, calibrate_robust, verify_patch_robust
from .refdie import MIN_AREA_PX, SMOOTH_PX
from .topo_utils import Window


@dataclass(frozen=True)
class CellReference:
    median: np.ndarray        # aligned median of the reference windows
    cad: np.ndarray           # bool pseudo-CAD
    rcal: RobustCalibration
    n_refs: int


def build_cell_reference(ref_windows: list[np.ndarray], rcfg: RobustCfg) -> CellReference:
    base = np.asarray(ref_windows[0], float)
    al = [base] + [ndi.shift(np.asarray(w, float), estimate_shift(base, np.asarray(w, float)), order=1, mode="nearest")
                   for w in ref_windows[1:]]
    med = np.median(al, axis=0)
    sm = ndi.gaussian_filter(med, SMOOTH_PX)
    cad = remove_small_holes(remove_small_objects(sm > threshold_otsu(sm), min_size=MIN_AREA_PX), area_threshold=MIN_AREA_PX)
    k = max(2, min(3, len(al)))
    groups = [al[i::k] for i in range(k)]                      # k calibration patches (C1's leave-one-out needs >= 2)
    patches = [(cad.astype(float), np.median(g, axis=0)) for g in groups if g]
    return CellReference(med, cad, calibrate_robust(patches, rcfg), len(al))


def splice(target: np.ndarray, ref: CellReference, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[float, float]]:
    """Target aligned to the reference median, inside ``box`` = (r0, r1, c0, c1); median elsewhere."""
    d = estimate_shift(ref.median, np.asarray(target, float))
    t = ndi.shift(np.asarray(target, float), d, order=1, mode="nearest")
    out = ref.median.copy()
    r0, r1, c0, c1 = box
    out[r0:r1, c0:c1] = t[r0:r1, c0:c1]
    return out, d


def verify_cell(target: np.ndarray, ref: CellReference, box, win: Window, pc: PipelineCfg, rc: EventRuleCfg,
                rcfg: RobustCfg) -> tuple[PipelineResult, tuple[float, float]]:
    img, d = splice(target, ref, box)
    return verify_patch_robust(img, ref.cad, ref.rcal, win, pc, rc, rcfg), d
