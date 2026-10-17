"""Export testbed samples + current-pipeline verdicts for the review UI (frontend/).

Writes frontend/data/run.js (``window.NANOVATE_RUN = {...}``; a script file so the UI also works from
file://). Each sample goes through the real pipeline (Module 1 calibration on scribe patches, Module 3
alignment + SDF energy, Module 4 with the config frozen on Manhattan val in results/calib/decisions.json);
every number is computed here, nothing is simulated or typed in. Synthetic data.

Usage: python scripts/export_ui.py [--seed 0] [--split val] [--per-cell 2] [--dose 50] [--out PATH]
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from nanovate.config import TestbedCfg
from nanovate.demo import load_frozen_cfg, walk_through
from nanovate.testbed.dataset import KINDS
from nanovate.testbed.layouts import FAMILIES

OUT = Path("frontend/data/run.js")


def _b64(arr: np.ndarray, fmt: str) -> str:
    """8-bit grayscale image as a data URI."""
    buf = io.BytesIO()
    kw = {"quality": 85} if fmt == "JPEG" else {}
    Image.fromarray(arr.astype(np.uint8), "L").save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _gray(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1) * 255


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-cell", type=int, default=2, help="dies per (family, kind)")
    ap.add_argument("--dose", type=float, default=50.0, help="e-/px")
    ap.add_argument("--decisions", default="results/calib/decisions.json", help="frozen verifier config")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)

    cfg = TestbedCfg()
    pc = load_frozen_cfg(a.decisions)
    samples = []
    for fam in FAMILIES:
        for kind in KINDS:
            for die in range(a.per_cell):
                w = walk_through(kind, family=fam, dose=a.dose, layout_id=0, die=die, seed=a.seed, split=a.split, pc=pc)
                s, r, res = w.sample, w.result.reason, w.result
                hi = float(a.dose * (cfg.sem.a0 + cfg.sem.a1) * 1.4)
                samples.append({
                    "id": f"S-{len(samples) + 1:04d}",
                    "family": fam, "split": a.split, "layoutId": 0, "die": die, "kind": kind,
                    "truth": s.label,
                    "truthDBeta": [s.topo["d_beta0"], s.topo["d_beta1"]],
                    "verdict": res.verdict,
                    "dBeta": list(r["delta_betti"]),
                    "betaRef": list(r["beta_ref"]), "betaImg": list(r["beta_img"]),
                    "bottleneckH0": round(r["bottleneck_h0"], 4), "bottleneckH1": round(r["bottleneck_h1"], 4),
                    "bottleneckMax": round(r["bottleneck_max"], 4),
                    "sdfEnergy": round(r["max_sdf_energy"], 3),
                    "singleton": res.singleton,
                    "shift": [round(v, 2) for v in s.shift], "shiftRecovered": [round(-res.shift[0], 2), round(-res.shift[1], 2)],
                    "dose": s.dose, "resamples": s.resamples,
                    "latencyMs": round(sum(res.timings_ms.values()), 2),
                    "fieldMs": round(res.timings_ms["topology"], 2),
                    "defect": {k: v for k, v in s.params.get("defect", {}).items() if isinstance(v, (int, float, str, bool))},
                    "img": {
                        "cad": _b64(s.cad * 255, "PNG"),
                        "wafer": _b64(s.W * 255, "PNG"),
                        "sem": _b64(_gray(s.image, 0, hi), "JPEG"),
                        "field": _b64(w.field * 255, "JPEG"),
                    },
                })
                print(f"{samples[-1]['id']} {fam:9s} {kind:9s} -> {res.verdict}", flush=True)

    run = {
        "synthetic": True,
        "seed": a.seed, "split": a.split, "perCell": a.per_cell,
        "module4": {"tau": pc.tau, "field": pc.field_option, "preSmooth": pc.pre_smooth, "regionMode": pc.region_mode,
                    "levelsSource": "calibrated (Module 1, 3 scribe patches)",
                    "sdfEnergy": f"Module 3 max tension energy (gamma {pc.gamma}, cap {pc.cap} px)",
                    "dStar": None if np.isinf(pc.d_star) else pc.d_star, "eStar": None if np.isinf(pc.e_star) else pc.e_star},
        "testbedCfg": dataclasses.asdict(cfg),
        "tile": cfg.layout.tile, "margin": cfg.layout.margin,
        "samples": samples,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.NANOVATE_RUN = " + json.dumps(run) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(samples)} samples)")


if __name__ == "__main__":
    main()
