"""Input sources for the demonstrator app: one ``ImageSource`` protocol, three implementations.

Every source returns an ``InspectionInput``: the SEM image (e-/px), the binary CAD clip, metadata with an
explicit provenance string, ground truth only when it genuinely exists (simulated inputs), the calibration
reference (scribe patches or a cached calibration id) and 0-2 reference dies for die-to-die.

The verifier was built for 1 nm/px and 256 x 256 px tiles; inputs outside that are refused, never resized,
and the pixel size is never inferred from the image size.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..config import TestbedCfg
from ..testbed.dataset import KINDS, get_layout, make_sample
from ..testbed.layouts import FAMILIES
from ..testbed.scribe import make_scribe_patches

TILE = 256                   # px, tile edge the verifier, its window and the CNN were built for
PIXEL_NM = 1.0               # nm/px the verifier was built for
DOSE_GRID = (5.0, 10.0, 20.0, 35.0, 50.0, 100.0, 200.0)   # e-/px, Exp-2 doses (die-to-die rules exist only here)
REF_DIES = (900, 901)        # reference dies for die-to-die, as in experiments/harness.py::Context.refs
SPLITS = ("live", "val", "test")
MAX_SHIFT = 8.0             # px per axis, the Exp-4 misalignment range

BADGE_SIM = "SIMULATED (Nanovate testbed)"
BADGE_SPLIT = "SIMULATED ({split} split sample)"
BADGE_FILE = "LOADED FROM FILE"


class InputError(ValueError):
    """An input the app refuses; ``code`` / ``field`` become the structured 4xx body."""

    def __init__(self, code: str, message: str, field: str | None = None, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.field, self.status = code, message, field, status


@dataclass(frozen=True)
class InputMeta:
    pixel_nm: float            # nm/px
    dose: float | None         # e-/px, None if unknown
    layer_id: str              # calibration cache key
    source_type: str           # 'simulated' | 'file' | 'tool'
    provenance: str            # exact badge text shown with every image
    params: dict               # source parameters (simulation params; file names + sha256)
    seed: int = 0              # drives simulation; echoed in every response


@dataclass(frozen=True)
class GroundTruth:
    """Only from the testbed's ground-truth wafer W (labels from exact topology of W vs CAD)."""

    kind: str                  # injected kind
    label: str                 # NUISANCE / NEAR_MISS / KILLER
    d_beta: tuple[int, int]    # exact (d_beta0, d_beta1) of W vs CAD on the central window
    W: np.ndarray              # bool wafer
    mask: np.ndarray           # bool, pixels changed by the defect
    expected: np.ndarray       # noise-free mean image, e-/px
    applied_shift: tuple[float, float]
    resamples: int


@dataclass(frozen=True)
class CalibrationRef:
    """Exactly one of: 3 scribe (cad, image) pairs, or the id of a calibration already cached."""

    scribe: tuple | None = None
    calibration_id: str | None = None


@dataclass(frozen=True)
class InspectionInput:
    image: np.ndarray          # float32 (H, W), e-/px
    cad: np.ndarray            # bool (H, W)
    meta: InputMeta
    calibration: CalibrationRef
    truth: GroundTruth | None = None
    references: tuple = ()     # 0-2 reference die images (e-/px, same CAD frame)
    warnings: tuple = ()       # out-of-domain notes, shown prominently
    input_id: str = ""


class ImageSource(Protocol):
    def load(self) -> InspectionInput: ...


def validate_arrays(image: np.ndarray, cad: np.ndarray, image_field: str = "image", cad_field: str = "cad") -> tuple[np.ndarray, np.ndarray]:
    """Check shape, dtype, finiteness and CAD content; return (float32 image, bool CAD) or raise ``InputError``."""
    image, cad = np.asarray(image), np.asarray(cad)
    if image.ndim != 2:
        raise InputError("bad_dtype" if image.ndim == 3 else "shape_mismatch",
                         f"image must be a single-channel 2D array, got shape {image.shape}", image_field)
    if not (np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)) or image.dtype == np.float16:
        raise InputError("bad_dtype", f"image dtype {image.dtype} not supported (use uint8/uint16/float32/float64)", image_field)
    if cad.ndim != 2:
        raise InputError("bad_dtype" if cad.ndim == 3 else "shape_mismatch", f"CAD must be a single-channel 2D mask, got shape {cad.shape}", cad_field)
    if cad.dtype != bool and not np.issubdtype(cad.dtype, np.integer):
        raise InputError("bad_dtype", f"CAD dtype {cad.dtype} not supported (use a bool or integer mask)", cad_field)
    if image.shape != cad.shape:
        raise InputError("shape_mismatch", f"image {image.shape} and CAD {cad.shape} have different sizes", cad_field)
    if image.shape != (TILE, TILE):
        raise InputError("tile_size_out_of_domain",
                         f"tile is {image.shape[0]} x {image.shape[1]} px; the verifier was built for {TILE} x {TILE} px tiles "
                         "(window, margin and CNN input). It is refused, not resized.", image_field)
    img = image.astype(np.float32)
    if not np.isfinite(img).all():
        raise InputError("non_finite", f"image contains {int((~np.isfinite(img)).sum())} NaN/inf pixels", image_field)
    c = cad.astype(bool)
    if not c.any():
        raise InputError("empty_cad", "CAD clip has no conductor pixels; there is no topology to compare against", cad_field)
    if c.all():
        raise InputError("full_cad", "CAD clip is all conductor; there is no topology to compare against", cad_field)
    return img, c


def sim_layer_id(dose: float, ler_sigma: float, grain: float, seed: int) -> str:
    """Calibration id of a simulated layer: the scribe patches depend on exactly these."""
    return f"sim/dose{dose:g}/ler{ler_sigma:g}/grain{grain:g}/seed{seed}"


def sim_tcfg(dose: float, ler_sigma: float, grain: float) -> TestbedCfg:
    """Testbed config for one simulated request (same replacement as experiments/harness.py)."""
    base = TestbedCfg()
    return dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)),
                               nuisance=dataclasses.replace(base.nuisance, ler_sigma=float(ler_sigma), grain_contrast=float(grain)))


@dataclass(frozen=True)
class SimulatedSource:
    """Wraps the testbed. Units: dose e-/px, ler_sigma nm, grain fraction, shift px (None = drawn), warp px."""

    family: str = "manhattan"
    kind: str = "bridge"
    dose: float = 50.0
    ler_sigma: float = 1.0
    grain: float = 0.15
    shift: tuple[float, float] | None = None
    warp: float = 0.0
    seed: int = 0
    split: str = "live"
    layout_id: int = 0
    die: int = 0

    def params(self) -> dict:
        d = dataclasses.asdict(self)
        d["shift"] = None if self.shift is None else [float(v) for v in self.shift]
        d["dose"], d["ler_sigma"], d["grain"], d["warp"] = float(self.dose), float(self.ler_sigma), float(self.grain), float(self.warp)
        return d

    def input_id(self) -> str:
        return "sim-" + hashlib.sha256(json.dumps(self.params(), sort_keys=True).encode()).hexdigest()[:16]

    def check(self) -> None:
        if self.split == "train":
            raise InputError("split_not_allowed", "the train split is refused: the CNN baseline was trained on it", "split")
        if self.split not in SPLITS:
            raise InputError("out_of_range", f"split must be one of {SPLITS}", "split")
        if self.family not in FAMILIES:
            raise InputError("out_of_range", f"family must be one of {FAMILIES}", "family")
        if self.kind not in KINDS:
            raise InputError("out_of_range", f"kind must be one of {KINDS}", "kind")
        if float(self.dose) not in DOSE_GRID:
            raise InputError("out_of_range", f"dose must be one of {DOSE_GRID} e-/px (die-to-die rules exist only there)", "dose")
        # 'not (lo <= v <= hi)' so NaN fails too (every comparison with NaN is False)
        for name, v, lo, hi in (("ler_sigma", self.ler_sigma, 0.5, 2.5), ("grain", self.grain, 0.0, 0.30), ("warp", self.warp, 0.0, 1.0),
                                ("seed", self.seed, 0, 2**31 - 1), ("layout_id", self.layout_id, 0, 999), ("die", self.die, 0, REF_DIES[0] - 1)):
            if not (lo <= v <= hi):
                raise InputError("out_of_range", f"{name} must be in [{lo}, {hi}], got {v}", name)
        if self.shift is not None and (len(self.shift) != 2 or not all(math.isfinite(float(v)) and abs(float(v)) <= MAX_SHIFT for v in self.shift)):
            raise InputError("out_of_range", f"shift must be two values within +-{MAX_SHIFT:g} px (the Exp-4 range)", "shift")

    def load(self) -> InspectionInput:
        self.check()
        tcfg = sim_tcfg(self.dose, self.ler_sigma, self.grain)
        layout = get_layout(self.family, self.split, self.layout_id, tcfg, self.seed)
        shift = None if self.shift is None else (float(self.shift[0]), float(self.shift[1]))
        s = make_sample(self.family, self.split, self.layout_id, self.die, self.kind, tcfg, self.seed,
                        layout=layout, shift=shift, warp_amp=float(self.warp))
        refs = tuple(make_sample(self.family, self.split, self.layout_id, d, "nuisance", tcfg, self.seed, layout=layout).image
                     for d in REF_DIES)
        badge = BADGE_SIM if self.split == "live" else BADGE_SPLIT.format(split=self.split)
        meta = InputMeta(PIXEL_NM, float(self.dose), sim_layer_id(self.dose, self.ler_sigma, self.grain, self.seed),
                         "simulated", badge, self.params(), int(self.seed))
        truth = GroundTruth(self.kind, s.label, (int(s.topo["d_beta0"]), int(s.topo["d_beta1"])), s.W, s.mask, s.expected,
                            (float(s.shift[0]), float(s.shift[1])), int(s.resamples))
        return InspectionInput(s.image, s.cad, meta, CalibrationRef(scribe=tuple(make_scribe_patches(tcfg, seed=self.seed))),
                               truth, refs, (), self.input_id())


class ToolSource:
    """Interface stub for a review-SEM vendor adapter. NOT implemented; the app has no route that uses it.

    A vendor adapter would have to provide, per review site:
      * the exported review image (single channel) with its pixel size in nm/px and the beam dose in e-/px
        (or the detector gain needed to convert grey levels to e-/px);
      * the site location: KLARF ``DefectList`` X/Y plus die index (and the die origin), mapped into the CAD frame;
      * the CAD clip around that location (a GDS/OASIS window at the review field of view, same layer);
      * three scribe-line images of large isolated features on the same layer, for Module-1 calibration;
      * a stable layer / recipe id (used as the calibration cache key);
      * optionally 1-2 reference-die images of the same site, for the die-to-die baseline.
    Nanovate never drives the tool: an adapter would only read exported files. No fake data is provided here.
    """

    def load(self) -> InspectionInput:
        raise NotImplementedError("ToolSource is an interface stub; no vendor adapter is implemented")
