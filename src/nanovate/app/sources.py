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


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
MAX_SIDE = 2048            # px, larger images are refused before decoding (413)


def _decode(data: bytes, field: str) -> tuple[np.ndarray, str]:
    """PNG (gray, 1/8/16-bit) or TIFF (one 2D page) -> (array, 'png' | 'tiff'). Raises InputError otherwise."""
    import io

    if data.startswith(PNG_MAGIC):
        from PIL import Image

        try:
            im = Image.open(io.BytesIO(data))
            if max(im.size) > MAX_SIDE:
                raise InputError("too_large", f"{im.size[0]} x {im.size[1]} px exceeds the {MAX_SIDE} px cap", field, status=413)
            if im.mode not in ("1", "L", "I;16", "I;16B", "I;16L", "I", "F"):
                raise InputError("bad_dtype", f"PNG mode {im.mode} is not single-channel greyscale (convert to 8/16-bit grey)", field)
            return np.array(im), "png"
        except InputError:
            raise
        except Exception as e:  # noqa: BLE001 (any decoder failure is a bad file, not a server error)
            raise InputError("unsupported_format", f"could not decode PNG: {e}", field, status=415) from e
    if data[:4] in TIFF_MAGIC:
        import tifffile

        try:
            with tifffile.TiffFile(io.BytesIO(data)) as tf:
                shape = tf.series[0].shape
                if len(shape) >= 2 and max(shape[-2:]) > MAX_SIDE:
                    raise InputError("too_large", f"TIFF {shape} exceeds the {MAX_SIDE} px cap", field, status=413)
                if len(tf.pages) > 1 or len(shape) != 2:
                    raise InputError("bad_dtype", f"TIFF must hold one 2D single-channel page, got shape {shape} in {len(tf.pages)} page(s)", field)
                return tf.asarray(), "tiff"
        except InputError:
            raise
        except Exception as e:  # noqa: BLE001
            raise InputError("unsupported_format", f"could not decode TIFF: {e}", field, status=415) from e
    raise InputError("unsupported_format", "unsupported format: use PNG or TIFF (CAD may also be GDS)", field, status=415)


def _is_gds(data: bytes) -> bool:
    return len(data) >= 4 and data[2:4] == b"\x00\x02"   # first record is HEADER (record type 0x00, data type 0x02)


def _gds_cad(data: bytes, sc) -> np.ndarray:
    """Rasterise one GDS cell/layer onto the 256 x 256 tile at 1 nm/px (testbed convention: x -> column, y -> row)."""
    import os
    import tempfile

    import gdstk

    from ..testbed.layouts import rasterize

    if sc.gds_cell is None or sc.gds_layer is None:
        raise InputError("bad_sidecar", "a GDS CAD needs gds_cell and gds_layer in the sidecar", "gds_cell")
    fd, path = tempfile.mkstemp(suffix=".gds")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            lib = gdstk.read_gds(path)
        except Exception as e:  # noqa: BLE001
            raise InputError("unsupported_format", f"could not read GDS: {e}", "cad", status=415) from e
    finally:
        os.unlink(path)
    cells = {c.name: c for c in lib.cells}
    if sc.gds_cell not in cells:
        raise InputError("bad_sidecar", f"GDS has no cell {sc.gds_cell!r} (cells: {sorted(cells)[:10]})", "gds_cell")
    scale = lib.unit / 1e-9                                  # user units -> nm
    polys = []
    for p in cells[sc.gds_cell].get_polygons(layer=sc.gds_layer, datatype=sc.gds_datatype):
        xy = np.asarray(p.points, dtype=float) * scale - np.asarray(sc.origin_nm)
        if sc.gds_flip_y:
            xy[:, 1] = -xy[:, 1]
        polys.append(xy / PIXEL_NM)
    if not polys:
        raise InputError("empty_cad", f"GDS cell {sc.gds_cell!r} has no polygons on layer {sc.gds_layer}/{sc.gds_datatype}", "cad")
    return rasterize(polys, TILE)


@dataclass(frozen=True)
class FileSource:
    """Image + CAD + JSON sidecar from files, plus 3 scribe pairs or a cached calibration id, plus 0-2 reference dies.

    Image units: TIFF values are taken as e-/px (times ``intensity_scale`` if given); PNG grey levels need
    ``intensity_scale`` (e-/px per level), else the upload is refused. CAD: PNG/TIFF mask (non-zero = conductor)
    or GDS (sidecar names cell, layer, datatype, origin). Never infers nm/px; refuses anything but 1 nm/px, 256 px.
    """

    image: bytes
    cad: bytes
    sidecar: bytes
    scribe_cad: tuple = ()
    scribe_image: tuple = ()
    calibration_id: str | None = None
    references: tuple = ()
    names: dict = field(default_factory=dict)   # field -> original file name (provenance only)

    def _sidecar(self):
        from pydantic import ValidationError

        from .schemas import Sidecar

        try:
            sc = Sidecar.model_validate_json(self.sidecar)
        except ValidationError as e:
            er = e.errors()[0]
            loc = ".".join(str(x) for x in er.get("loc", ())) or "sidecar"
            raise InputError("bad_sidecar", f"sidecar {loc}: {er.get('msg')}", loc) from e
        if sc.pixel_nm != PIXEL_NM:
            raise InputError("pixel_size_out_of_domain", f"pixel size {sc.pixel_nm:g} nm/px: the verifier was built for {PIXEL_NM:g} nm/px "
                             "(every length in it is in px). It is refused, not rescaled.", "pixel_nm")
        if sc.layer_id.startswith("sim"):
            raise InputError("bad_sidecar", "layer ids starting with 'sim' are reserved for simulated layers", "layer_id")
        return sc

    def _image(self, data: bytes, field: str, sc) -> np.ndarray:
        arr, fmt = _decode(data, field)
        if fmt == "png" and sc.intensity_scale is None:
            raise InputError("bad_sidecar", "a PNG image needs intensity_scale (e-/px per grey level) in the sidecar; "
                             "grey levels are not counts and are not guessed", "intensity_scale")
        if arr.dtype == bool or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)) or arr.dtype == np.float16:
            raise InputError("bad_dtype", f"{field} dtype {arr.dtype} not supported", field)
        return arr.astype(np.float64) * (sc.intensity_scale or 1.0)

    def _mask(self, data: bytes, field: str, sc) -> np.ndarray:
        if field == "cad" and _is_gds(data):
            return _gds_cad(data, sc)
        arr, _ = _decode(data, field)
        if arr.dtype != bool and not np.issubdtype(arr.dtype, np.integer):
            raise InputError("bad_dtype", f"{field} must be an integer or 1-bit mask (non-zero = conductor), got {arr.dtype}", field)
        return arr != 0

    def input_id(self) -> str:
        h = hashlib.sha256()
        for b in (self.image, self.cad, self.sidecar, *self.scribe_cad, *self.scribe_image, *self.references):
            h.update(hashlib.sha256(b).digest())
        h.update((self.calibration_id or "").encode())
        return "file-" + h.hexdigest()[:16]

    def load(self) -> InspectionInput:
        sc = self._sidecar()
        image, cad = validate_arrays(self._image(self.image, "image", sc), self._mask(self.cad, "cad", sc))
        if self.scribe_cad or self.scribe_image:
            if len(self.scribe_cad) != 3 or len(self.scribe_image) != 3:
                raise InputError("calibration_missing", f"calibration needs exactly 3 scribe CAD masks and 3 scribe images, "
                                 f"got {len(self.scribe_cad)} and {len(self.scribe_image)}", "scribe")
            scribe = []
            for i, (c, im) in enumerate(zip(self.scribe_cad, self.scribe_image)):
                si, scd = validate_arrays(self._image(im, f"scribe_image[{i}]", sc), self._mask(c, f"scribe_cad[{i}]", sc),
                                          f"scribe_image[{i}]", f"scribe_cad[{i}]")
                scribe.append((scd, si))
            cal = CalibrationRef(scribe=tuple(scribe))
        elif self.calibration_id:
            cal = CalibrationRef(calibration_id=self.calibration_id)
        else:
            raise InputError("calibration_missing", "Upload 3 scribe patches or name an existing calibration id; "
                             "the verifier will not guess the conductor/dielectric levels.", "calibration")
        if len(self.references) > 2:
            raise InputError("bad_request", f"at most 2 reference dies, got {len(self.references)}", "reference", status=400)
        refs = []
        for i, r in enumerate(self.references):
            ri = self._image(r, f"reference[{i}]", sc)
            if ri.shape != image.shape:
                raise InputError("shape_mismatch", f"reference[{i}] {ri.shape} differs from the image {image.shape}", f"reference[{i}]")
            refs.append(validate_arrays(ri, cad, f"reference[{i}]")[0])
        warnings = ["Loaded from file: no ground truth exists, so no verdict is marked correct or wrong.",
                    "Every method here was built and fitted on simulated data; a real SEM image is outside that domain (sim-to-real gap)."]
        if sc.dose is None:
            warnings.append("Dose unknown: die-to-die has no rule to apply.")
        elif float(sc.dose) not in DOSE_GRID:
            warnings.append(f"Dose {sc.dose:g} e-/px is not one of the die-to-die rule doses {list(DOSE_GRID)}: die-to-die is not available.")
        if not refs:
            warnings.append("No reference die uploaded: die-to-die is not available.")
        if cal.calibration_id:
            warnings.append(f"Calibration reused from cached id {cal.calibration_id!r} (fitted on that layer's own scribe patches).")
        files = {k: {"name": self.names.get(k, k), "sha256": hashlib.sha256(b).hexdigest()[:16]}
                 for k, b in [("image", self.image), ("cad", self.cad), ("sidecar", self.sidecar)]
                 + [(f"scribe_cad[{i}]", b) for i, b in enumerate(self.scribe_cad)] + [(f"scribe_image[{i}]", b) for i, b in enumerate(self.scribe_image)]
                 + [(f"reference[{i}]", b) for i, b in enumerate(self.references)]}
        meta = InputMeta(PIXEL_NM, None if sc.dose is None else float(sc.dose), cal.calibration_id or sc.layer_id, "file", BADGE_FILE,
                         {"files": files, "sidecar": sc.model_dump(exclude_none=True), "calibration_id": self.calibration_id}, 0)
        return InspectionInput(image.astype(np.float32), cad, meta, cal, None, tuple(refs), tuple(warnings), self.input_id())


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
