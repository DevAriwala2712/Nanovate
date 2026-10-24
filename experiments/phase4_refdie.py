"""Phase 4: reference-die mode vs CAD mode on synthetic val layouts 8-23 (docs/PHASE4_REFDIE_PREREG.md, committed c2e3601).

Modes on the SAME patches: CAD (frozen v3, scribe calibration), R2 (two reference dies, primary), R1 (one reference die),
R-oracle (true wafer of reference die 900; diagnostic). Kill if pooled F1(CAD) - F1(R2) > 0.10. Synthetic data only.
Writes results/phase4/{rows.csv, summary.csv, per_kind.csv, config.json}. Usage: python -m experiments.phase4_refdie [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from experiments.common import f1_score
from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_v3 import verify_patch_v3
from nanovate.refdie import build_reference, verify_refdie
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.labels import topology_label
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window

OUT = Path("results/phase4")
POPS = [("manhattan", 20.0, False), ("manhattan", 50.0, False), ("manhattan", 200.0, False), ("manhattan", 50.0, True),
        ("via", 50.0, False), ("curvy", 50.0, False)]
MODES = ("CAD", "R2", "R1", "R-oracle")
N_BOOT, SEED = 1000, 0
_CAL: dict = {}


def _cfg():
    dec = json.loads(Path("results/v3/calib/decisions.json").read_text())
    return _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"])


def _tcfg(dose: float) -> TestbedCfg:
    b = TestbedCfg()
    return dataclasses.replace(b, sem=dataclasses.replace(b.sem, dose=float(dose)))


def _pcad_topo_ok(pcad, ref_sample, cad, win) -> bool:
    back = ndi.shift(pcad.astype(float), tuple(-v for v in ref_sample.shift), order=0, mode="nearest") > 0.5
    return topology_label(back, cad, win)["topo"] == "none"


def _layout_task(spec: dict) -> list[dict]:
    pc, rc = _cfg()
    fam, dose, nf, L = spec["family"], spec["dose"], spec["noise_free"], spec["layout"]
    t, win = _tcfg(dose), Window(256, 16)
    if dose not in _CAL:
        _CAL[dose] = calibrate(make_scribe_patches(t, seed=0))
    lay = get_layout(fam, "val", L, t, 0)
    refs = [make_sample(fam, "val", L, 900 + j, "nuisance", t, 0, layout=lay) for j in (0, 1)]
    rimg = [r.expected if nf else r.image for r in refs]
    t0 = time.perf_counter(); r2 = build_reference(rimg); b2 = (time.perf_counter() - t0) * 1e3
    r1 = build_reference(rimg[:1])
    q2, q1 = _pcad_topo_ok(r2.cad, refs[0], lay.cad, win), _pcad_topo_ok(r1.cad, refs[0], lay.cad, win)
    rows = []
    for die in (0, 1):
        for kind in KINDS:
            s = make_sample(fam, "val", L, die, kind, t, 0, layout=lay)
            img = s.expected if nf else s.image
            row = dict(family=fam, dose=dose, noise_free=nf, layout=L, die=die, kind=kind, label=s.label, killer=s.label == "KILLER",
                       pcad_topo_ok_R2=q2, pcad_topo_ok_R1=q1, ref_build_ms_R2=b2)
            v = {"CAD": verify_patch_v3(img, s.cad, _CAL[dose], win, pc, rc).verdict}
            for m, kw in (("R2", dict(ref_images=rimg)), ("R1", dict(ref_images=rimg[:1])),
                          ("R-oracle", dict(ref_images=rimg[:1], oracle_cad=refs[0].W))):
                try:
                    v[m] = verify_refdie(img, kw["ref_images"], win, pc, rc, kw.get("oracle_cad"))[0].verdict
                except ValueError:
                    v[m] = "REF_FAILED"
            for m in MODES:
                row[f"v_{m}"] = v[m]
                row[f"k_{m}"] = str(v[m]).startswith("KILLER")
            rows.append(row)
    return rows


def paired_diff(pa, pb, y):
    pa, pb, y = map(np.asarray, (pa, pb, y))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    d = np.array([f1_score(pa[i], y[i]) - f1_score(pb[i], y[i]) for i in idx])
    return float(f1_score(pa, y) - f1_score(pb, y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    layouts = range(8, 10) if a.quick else range(8, 24)
    specs = [dict(family=f, dose=d, noise_free=nf, layout=L) for f, d, nf in POPS for L in layouts]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(_layout_task, specs) for r in rows])
    df.to_csv(out / "rows.csv", index=False)
    pop = df.family + "|" + df.dose.astype(int).astype(str) + np.where(df.noise_free, "|noise-free", "")
    df["population"] = pop
    summ = []
    for name, g in [("POOLED", df)] + list(df.groupby("population")):
        y = g.killer.values
        for m in MODES:
            d, lo, hi = paired_diff(g["k_CAD"], g[f"k_{m}"], y) if m != "CAD" else (0.0, 0.0, 0.0)
            summ.append(dict(population=name, mode=m, killer_f1=f1_score(g[f"k_{m}"].values, y), f1_cad_minus_mode=d, ci_lo=lo, ci_hi=hi,
                             n=len(g), ref_failed=int((g[f"v_{m}"] == "REF_FAILED").sum())))
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    pk = []
    for m in MODES:
        for (p, k), g in df.groupby(["population", "kind"]):
            pk.append(dict(mode=m, population=p, kind=k, called_killer=float(g[f"k_{m}"].mean()), n=len(g)))
    pd.DataFrame(pk).to_csv(out / "per_kind.csv", index=False)
    prim = s[(s.population == "POOLED") & (s["mode"] == "R2")].iloc[0]
    qual = df.drop_duplicates(["population", "layout"])
    cfg = dict(data="synthetic", split="val layouts 8-23", prereg="docs/PHASE4_REFDIE_PREREG.md", quick=a.quick,
               primary=dict(f1_cad_minus_R2=prim.f1_cad_minus_mode, ci=[prim.ci_lo, prim.ci_hi], n=int(prim.n),
                            killed=bool(prim.f1_cad_minus_mode > 0.10)),
               pseudo_cad_topology_ok=dict(R2=float(qual.pcad_topo_ok_R2.mean()), R1=float(qual.pcad_topo_ok_R1.mean()), n_refs=len(qual)),
               ref_build_ms_R2_median=float(df.ref_build_ms_R2.median()), wall_s=time.time() - t0)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(s.round(4).to_string()); print(json.dumps(cfg["primary"]), json.dumps(cfg["pseudo_cad_topology_ok"]))


if __name__ == "__main__":
    main()
