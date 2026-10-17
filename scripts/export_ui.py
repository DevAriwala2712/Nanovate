"""Export real testbed samples + Module-4 verdicts for the review UI (frontend/).

Writes frontend/data/run.js (``window.NANOVATE_RUN = {...}``; a script file so the UI also
works from file://). Every number is computed here from the synthetic testbed and the
current Module 4 -- nothing is simulated or typed in. Synthetic data.

Usage: python scripts/export_ui.py [--seed 0] [--split val] [--per-cell 2]
                                   [--tau 0.45] [--field calibrated|percentile] [--pre-smooth 0.0]
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from nanovate.config import TestbedCfg
from nanovate.modules.fields import build_field
from nanovate.modules.topology import verify_topology
from nanovate.testbed.dataset import KINDS, make_sample, window
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-cell", type=int, default=2, help="dies per (family, kind)")
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--field", default="calibrated", choices=["calibrated", "percentile"])
    ap.add_argument("--pre-smooth", type=float, default=0.0, help="px")
    a = ap.parse_args()

    cfg = TestbedCfg()
    win = window(cfg)
    samples = []
    for fam in FAMILIES:
        for kind in KINDS:
            for die in range(a.per_cell):
                s = make_sample(fam, a.split, 0, die, kind, cfg, a.seed)
                t0 = time.perf_counter()
                field, _ = build_field(s.image, option=a.field, sem_cfg=cfg.sem, pre_smooth=a.pre_smooth)
                t1 = time.perf_counter()
                verdict, r = verify_topology(
                    s.image, s.cad, win, tau=a.tau, field_option=a.field,
                    sem_cfg=cfg.sem, pre_smooth=a.pre_smooth, singleton=True)
                t2 = time.perf_counter()
                hi = float(cfg.sem.dose * (cfg.sem.a0 + cfg.sem.a1) * 1.4)
                samples.append({
                    "id": f"S-{len(samples) + 1:04d}",
                    "family": fam, "split": a.split, "layoutId": 0, "die": die, "kind": kind,
                    "truth": s.label,
                    "truthDBeta": [s.topo["d_beta0"], s.topo["d_beta1"]],
                    "verdict": verdict,
                    "dBeta": list(r["delta_betti"]),
                    "betaRef": list(r["beta_ref"]), "betaImg": list(r["beta_img"]),
                    "bottleneckH0": r["bottleneck_h0"], "bottleneckH1": r["bottleneck_h1"],
                    "bottleneckMax": r["bottleneck_max"],
                    "sdfEnergy": r["max_sdf_energy"],           # null until Module 3 lands
                    "singleton": r["singleton"],
                    "shift": list(s.shift), "dose": s.dose, "resamples": s.resamples,
                    "latencyMs": round((t2 - t0) * 1e3, 2),
                    "fieldMs": round((t1 - t0) * 1e3, 2),
                    "defect": {k: v for k, v in s.params.get("defect", {}).items()
                               if isinstance(v, (int, float, str, bool))},
                    "img": {
                        "cad": _b64(s.cad * 255, "PNG"),
                        "wafer": _b64(s.W * 255, "PNG"),
                        "sem": _b64(_gray(s.image, 0, hi), "JPEG"),
                        "field": _b64(field * 255, "JPEG"),
                    },
                })
                print(f"{samples[-1]['id']} {fam:9s} {kind:9s} -> {verdict}", flush=True)

    run = {
        "synthetic": True,
        "seed": a.seed, "split": a.split, "perCell": a.per_cell,
        "module4": {"tau": a.tau, "field": a.field, "preSmooth": a.pre_smooth,
                    "levelsSource": "oracle (Module 1 not built yet)",
                    "sdfEnergy": "not computed (Module 3 not built yet)",
                    "dStar": None, "eStar": None},
        "testbedCfg": dataclasses.asdict(cfg),
        "tile": cfg.layout.tile, "margin": cfg.layout.margin,
        "samples": samples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.NANOVATE_RUN = " + json.dumps(run) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(samples)} samples)")


if __name__ == "__main__":
    main()
