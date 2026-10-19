"""Write synthetic sample bundles for testing the app's "Load from File" path (python -m nanovate.app).

Each bundle is a folder of files exactly as the upload dialog / POST /api/v1/upload expects them:
  image.tif | image.png      SEM image (TIFF: e-/px float32; PNG: 16-bit grey + intensity_scale in the sidecar)
  cad.png | cad.gds          CAD clip (PNG mask, non-zero = conductor; or GDS with cell/layer/origin in the sidecar)
  sidecar.json               metadata (pixel_nm, layer_id, dose, ...)
  scribe_cad_{0,1,2}.png     3 scribe CAD masks          } Module-1 calibration (pick all 3 of each,
  scribe_image_{0,1,2}.tif   3 scribe images, same order } in the same order)
  reference_{0,1}.tif        2 reference dies (die-to-die)
  truth.json                 ground truth from the simulated wafer W -- NOT uploaded; it is for you to compare with
  README.txt                 what to upload and what to expect

Bundles 01-06 are valid; 07-13 are deliberately broken and must be refused with the error code named in their
README. Everything is synthetic (Nanovate testbed, fixed seeds): re-running the script rewrites identical files.

Usage: python scripts/make_sample_uploads.py [--out samples/uploads]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for the experiments package

from experiments.common import ExpCfg  # noqa: E402
from experiments.select_config import load_pipeline_cfg  # noqa: E402
from nanovate.app.sources import SimulatedSource, sim_tcfg  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.modules.calibration import calibrate  # noqa: E402
from nanovate.pipeline import verify_patch  # noqa: E402
from nanovate.testbed.dataset import get_layout, window  # noqa: E402

OUT = Path("samples/uploads")

VALID = [
    # name, simulation, image format, CAD format, dose in sidecar, references
    ("01_manhattan_bridge", dict(family="manhattan", kind="bridge", seed=3), "tiff", "png", 50.0, True),
    ("02_manhattan_nuisance", dict(family="manhattan", kind="nuisance", seed=4), "tiff", "png", 50.0, True),
    ("03_via_particle_png16", dict(family="via", kind="particle", seed=0), "png16", "png", 50.0, True),
    ("04_curvy_open_gds", dict(family="curvy", kind="open", seed=1), "tiff", "gds", 50.0, True),
    ("05_manhattan_filament_nearmiss", dict(family="manhattan", kind="filament", seed=0), "tiff", "png", 50.0, True),
    ("06_curvy_notch_no_refs_dose_unknown", dict(family="curvy", kind="notch", seed=2), "tiff", "png", None, False),
]
INVALID = [
    ("07_bad_pixel_size_2nm", "pixel_size_out_of_domain", "sidecar says 2 nm/px; the verifier was built for 1 nm/px"),
    ("08_bad_tile_300px", "tile_size_out_of_domain", "300 x 300 px tile; the verifier needs 256 x 256"),
    ("09_bad_rgb_png", "bad_dtype", "colour (RGB) PNG instead of single-channel grey"),
    ("10_bad_nan_pixels", "non_finite", "TIFF with NaN pixels"),
    ("11_bad_no_calibration", "calibration_missing", "no scribe patches and no calibration id"),
    ("12_bad_sidecar_unknown_key", "bad_sidecar", "sidecar has an unknown key (\"colour\")"),
    ("13_bad_png_without_scale", "bad_sidecar", "8-bit PNG image but no intensity_scale in the sidecar"),
]


def tif(a: np.ndarray, path: Path) -> None:
    tifffile.imwrite(path, np.asarray(a, np.float32))


def png_mask(m: np.ndarray, path: Path) -> None:
    Image.fromarray(np.asarray(m, bool).astype(np.uint8) * 255, "L").save(path, "PNG")


def gds(src: SimulatedSource, path: Path) -> list[float]:
    """CAD as GDS (1 nm database unit). Returns origin_nm: the GDS (x, y) of the tile's top-left corner."""
    import gdstk

    L = get_layout(src.family, src.split, src.layout_id, sim_tcfg(src.dose, src.ler_sigma, src.grain), src.seed)
    lib = gdstk.Library(unit=1e-9, precision=1e-12)
    cell = lib.new_cell("TILE")
    for p in L.polygons:
        cell.add(gdstk.Polygon(np.asarray(p).tolist(), layer=1, datatype=0))
    lib.write_gds(str(path))
    pad = TestbedCfg().layout.pad
    return [float(pad), float(pad)]


def write_scribe_and_refs(inp, d: Path, refs: bool) -> None:
    for i, (c, im) in enumerate(inp.calibration.scribe):
        png_mask(c, d / f"scribe_cad_{i}.png")
        tif(im, d / f"scribe_image_{i}.tif")
    if refs:
        for i, r in enumerate(inp.references):
            tif(r, d / f"reference_{i}.tif")


def direct_verdict(inp) -> str:
    """The verifier's verdict on this exact sample, computed by the research core directly (no app involved)."""
    cal = calibrate([(c, i) for c, i in inp.calibration.scribe])
    return verify_patch(inp.image, inp.cad, cal, window(TestbedCfg()), load_pipeline_cfg(ExpCfg.full())).verdict


def valid_bundle(root: Path, name: str, sim: dict, img_fmt: str, cad_fmt: str, dose, refs: bool) -> dict:
    d = root / name
    d.mkdir(parents=True)
    src = SimulatedSource(**sim)
    inp = src.load()
    sc: dict = {"pixel_nm": 1.0, "layer_id": f"sample-{name[:2]}"}
    if dose is not None:
        sc["dose"] = dose
    if img_fmt == "tiff":
        tif(inp.image, d / "image.tif")
        img_file = "image.tif"
    else:   # 16-bit PNG: grey level = counts / scale (rounded), so the app reads counts = grey * scale
        scale = 0.01
        Image.fromarray(np.clip(np.round(inp.image / scale), 0, 65535).astype(np.uint16)).save(d / "image.png", "PNG")
        sc["intensity_scale"] = scale
        img_file = "image.png"
    if cad_fmt == "png":
        png_mask(inp.cad, d / "cad.png")
        cad_file = "cad.png"
    else:
        sc.update(gds_cell="TILE", gds_layer=1, gds_datatype=0, origin_nm=gds(src, d / "cad.gds"))
        cad_file = "cad.gds"
    write_scribe_and_refs(inp, d, refs)
    (d / "sidecar.json").write_text(json.dumps(sc, indent=2) + "\n")
    t = inp.truth
    verdict = direct_verdict(inp)
    truth = {"synthetic": True, "simulation": src.params(), "label": t.label, "injected_kind": t.kind, "delta_beta": list(t.d_beta),
             "applied_shift_px": list(t.applied_shift), "verifier_verdict_direct_core_call": verdict,
             "note": "Ground truth from the simulated wafer W. The app shows NO ground truth for a loaded file; compare by hand."
                     + (" The PNG is rounded to 0.01 e-/px, so its verdict may differ from the unrounded direct call." if img_fmt == "png16" else "")}
    (d / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    expect_d2d = "available (single + double)" if (refs and dose is not None) else \
        ("not available: dose unknown" if dose is None else "not available: no reference die")
    (d / "README.txt").write_text(
        f"{name} -- synthetic (Nanovate testbed), valid upload\n\n"
        f"Upload: image = {img_file}, CAD = {cad_file}, sidecar = sidecar.json,\n"
        f"        calibration = scribe_cad_0..2.png (3 files) + scribe_image_0..2.tif (3 files, same order)"
        + (",\n        reference dies = reference_0.tif, reference_1.tif\n" if refs else "\n        (no reference dies in this bundle)\n")
        + f"\nGround truth (not uploaded): {t.label} ({t.kind}, delta-beta {tuple(t.d_beta)}).\n"
        f"Verifier verdict from a direct core call: {verdict}.\n"
        f"Expected in the app: badge LOADED FROM FILE, no correctness marks, die-to-die {expect_d2d}.\n")
    return {"bundle": name, "valid": True, "image": img_file, "cad": cad_file, "truth": t.label, "kind": t.kind,
            "verifier_direct": verdict, "d2d": expect_d2d}


def invalid_bundle(root: Path, name: str, code: str, why: str, base) -> dict:
    d = root / name
    d.mkdir(parents=True)
    sc = {"pixel_nm": 1.0, "layer_id": f"sample-{name[:2]}", "dose": 50.0}
    image, cad, img_file = base.image, base.cad, "image.tif"
    scribe = True
    if code == "pixel_size_out_of_domain":
        sc["pixel_nm"] = 2.0
    elif code == "tile_size_out_of_domain":
        image, cad = np.pad(image, ((22, 22), (22, 22)), mode="edge"), np.pad(cad, ((22, 22), (22, 22)), mode="edge")
    elif code == "non_finite":
        image = image.copy(); image[100:104, 100:104] = np.nan
    elif code == "calibration_missing":
        scribe = False
    elif code == "bad_sidecar" and "unknown key" in why:
        sc["colour"] = "blue"
    if code == "bad_dtype":
        img_file = "image.png"
        g = np.clip(image / image.max() * 255, 0, 255).astype(np.uint8)
        Image.fromarray(np.stack([g] * 3, -1), "RGB").save(d / img_file, "PNG"); sc["intensity_scale"] = 1.0
    elif "intensity_scale" in why:
        img_file = "image.png"
        Image.fromarray(np.clip(image / image.max() * 255, 0, 255).astype(np.uint8), "L").save(d / img_file, "PNG")
    else:
        tif(image, d / img_file)
    png_mask(cad, d / "cad.png")
    if scribe:
        write_scribe_and_refs(base, d, refs=False)
    (d / "sidecar.json").write_text(json.dumps(sc, indent=2) + "\n")
    (d / "README.txt").write_text(f"{name} -- synthetic, deliberately INVALID upload\n\nProblem: {why}.\n"
                                  f"Upload: image = {img_file}, CAD = cad.png, sidecar = sidecar.json"
                                  + (", scribe files as in the valid bundles" if scribe else ", no calibration") + ".\n"
                                  f"Expected: the app refuses it with error code '{code}', shown next to the offending field.\n")
    return {"bundle": name, "valid": False, "expected_error": code, "why": why}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows = [valid_bundle(root, *v) for v in VALID]
    print("\n".join(f"  {r['bundle']}: truth {r['truth']}, verifier {r['verifier_direct']}" for r in rows), flush=True)
    base = SimulatedSource(family="manhattan", kind="bridge", seed=3).load()
    rows += [invalid_bundle(root, *x, base) for x in INVALID]
    (root / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Sample uploads (synthetic data)", "",
             "Generated by `python scripts/make_sample_uploads.py`. Start the app (`python -m nanovate.app`), open "
             "http://127.0.0.1:8765/, press **Load file…** in the Live tab and pick the files named in each bundle's README.txt.",
             "Ground truth is in each `truth.json` and is never uploaded: the app shows no ground truth for a loaded file.", "",
             "| Bundle | Valid | Image / CAD | Ground truth | Verifier (direct core call) | Die-to-die / expected error |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["valid"]:
            lines.append(f"| {r['bundle']} | yes | {r['image']} / {r['cad']} | {r['truth']} ({r['kind']}) | {r['verifier_direct']} | {r['d2d']} |")
        else:
            lines.append(f"| {r['bundle']} | no | | | | `{r['expected_error']}`: {r['why']} |")
    lines += ["", "Command-line check of one bundle:", "", "```bash",
              "cd samples/uploads/01_manhattan_bridge",
              "curl -s -F image=@image.tif -F cad=@cad.png -F sidecar=@sidecar.json \\",
              "  -F scribe_cad=@scribe_cad_0.png -F scribe_cad=@scribe_cad_1.png -F scribe_cad=@scribe_cad_2.png \\",
              "  -F scribe_image=@scribe_image_0.tif -F scribe_image=@scribe_image_1.tif -F scribe_image=@scribe_image_2.tif \\",
              "  -F reference=@reference_0.tif -F reference=@reference_1.tif http://127.0.0.1:8765/api/v1/upload",
              "```", "", "Then `POST /api/v1/verify` with `{\"input_id\": \"<id from the upload>\"}`."]
    (root / "INDEX.md").write_text("\n".join(lines) + "\n")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"wrote {len(rows)} bundles to {root} ({size / 1e6:.1f} MB); see {root / 'INDEX.md'}")


if __name__ == "__main__":
    main()
