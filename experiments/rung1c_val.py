"""Rung 1c Part A: re-select the near-miss gate (d*, e*, z*) with edge-halo stress cells on synthetic Manhattan VAL.

Protocol exactly as docs/REAL_DATA_RUNG1C_PREREG.md Part A (committed 2b4ab09 before this ran). Base = the Rung 1b frozen
pipeline (v3 + C1 guard + C2 clamp + C3 bounded alignment); only d*, e*, z* change, so no KILLER verdict can change.
Evidence per patch is computed once with the near-miss gate off; v3_select.near_miss is reused unchanged.
Writes results/real/rung1c/val/{selection_records.csv, heldout_records.csv, near_miss_grid*.csv, heldout_summary.csv,
decisions.json}. Usage: python -m experiments.rung1c_val [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v3_harness import _pc
from experiments.v3_select import near_miss
from nanovate.config import TestbedCfg
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_robust import RobustCfg, calibrate_robust, verify_patch_robust
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window

OUT = Path("results/real/rung1c/val")
BASE = Path("results/real/rung1b/val/decisions.json")
HALO = {"nominal": (1.3, 0.9), "H1": (3.0, 0.9), "H2": (3.0, 1.8), "H3": (5.0, 0.9), "H4": (5.0, 1.8)}
N_BOOT, SEED = 1000, 0


def _base():
    d = json.loads(BASE.read_text())
    rc = RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d["robust_cfg"].items()})
    return _pc(d["base"]["pipeline_cfg"]), EventRuleCfg(**d["base"]["event_rule_cfg"]), rc


def tcfg_for(dose, ler, grain, halo) -> TestbedCfg:
    b = TestbedCfg()
    se, a2 = HALO[halo]
    return dataclasses.replace(b, sem=dataclasses.replace(b.sem, dose=float(dose), sigma_edge=se, a2=a2),
                               nuisance=dataclasses.replace(b.nuisance, ler_sigma=float(ler), grain_contrast=float(grain)))


_CAL: dict = {}


def _record(spec: dict) -> dict:
    pc, rc, rcfg = _base()
    off = dataclasses.replace(pc, d_star=float("inf"), e_star=float("inf"), z_star=float("-inf"))
    t = tcfg_for(spec["dose"], spec["ler"], spec["grain"], spec["halo"])
    key = (spec["dose"], spec["ler"], spec["grain"], spec["halo"])
    if key not in _CAL:
        _CAL[key] = calibrate_robust(make_scribe_patches(t, seed=0), rcfg)
    rcal = _CAL[key]
    lay = get_layout("manhattan", "val", spec["layout"], t, 0)
    s = make_sample("manhattan", "val", spec["layout"], spec["die"], spec["kind"], t, 0, layout=lay)
    r = verify_patch_robust(s.image, s.cad, rcal, Window(256, 16), off, rc, rcfg)
    z = r.reason
    cell = f"{spec['ler']}|{spec['halo']}"
    return dict(spec, sigma=cell, killer=s.label == "KILLER", killer_flag=str(z["verdict_unguarded"]).startswith("KILLER"),
                guard_reliable=rcal.reliable, verdict=r.verdict, bottleneck=z["bottleneck_max"], energy=z["max_sdf_energy"], peak_z=z["peak_z"])


def specs(layouts, doses, cells) -> list[dict]:
    return [dict(dose=float(d), ler=ler, grain=gr, halo=h, layout=L, die=die, kind=k)
            for (ler, gr, h) in cells for d in doses for L in layouts for die in (0, 1) for k in KINDS]


def _alarm(df, d, e, z):
    return (~df.killer_flag.values) & (df.bottleneck.values >= d) & (df.energy.values >= e) & (df.peak_z.values >= z)


def _ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    pc, rc, rcfg = _base()
    sel_layouts = range(1) if a.quick else range(8)
    doses = (50,) if a.quick else (20, 50, 200)
    base_cells = [(s, g, "nominal") for s in ((1.0,) if a.quick else (0.5, 1.0, 1.5)) for g in ((0.15,) if a.quick else (0.0, 0.15, 0.3))]
    halo_cells = [(1.0, 0.15, h) for h in ("H1", "H2", "H3", "H4")]
    with ProcessPoolExecutor(a.workers) as ex:
        sel = pd.DataFrame(list(ex.map(_record, specs(sel_layouts, doses, base_cells + halo_cells), chunksize=4)))
        held = pd.DataFrame(list(ex.map(_record, specs(range(8, 10) if a.quick else range(8, 24), (50,),
                                                         [(1.0, 0.15, "nominal")] + halo_cells), chunksize=4)))
    sel.to_csv(out / "selection_records.csv", index=False)
    held.to_csv(out / "heldout_records.csv", index=False)

    def select(df):
        nominal = ((df.ler == 1.0) & (df.grain == 0.15) & (df.halo == "nominal")).values
        meta = df.rename(columns={})[["kind", "sigma", "grain"]].copy()
        return near_miss(meta, df.killer_flag.values, df[["bottleneck", "energy", "peak_z"]], nominal)

    d1, e1, z1, G1 = select(sel)
    ctrl = sel[sel.halo == "nominal"].reset_index(drop=True)
    d0, e0, z0, G0 = select(ctrl)
    G1.to_csv(out / "near_miss_grid.csv", index=False)
    G0.to_csv(out / "near_miss_grid_control_no_halo.csv", index=False)
    old = (pc.d_star, pc.e_star, pc.z_star)
    rows = []
    for name, (d, e, z) in (("v3/1b thresholds", old), ("control re-selection (no halo cells)", (d0, e0, z0)), ("Rung 1c (with halo cells)", (d1, e1, z1))):
        al = _alarm(held, d, e, z)
        for h in HALO:
            g = held.halo.values == h
            nu, nm = g & (held.kind == "nuisance").values, g & held.kind.isin(["notch", "filament"]).values
            for metric, mask in (("nuisance NEAR_MISS rate", nu), ("near-miss recall (notch+filament)", nm)):
                m, lo, hi = _ci(al[mask])
                rows.append(dict(thresholds=name, halo=h, metric=metric, mean=m, ci_lo=lo, ci_hi=hi, n=int(mask.sum())))
    hs = pd.DataFrame(rows)
    hs.to_csv(out / "heldout_summary.csv", index=False)
    fin = lambda v: None if not np.isfinite(v) else float(v)  # noqa: E731
    dec = dict(data="synthetic", split="val (Manhattan)", prereg="docs/REAL_DATA_RUNG1C_PREREG.md Part A",
               base=json.loads(BASE.read_text()), halo_cells={k: dict(sigma_edge=v[0], a2=v[1]) for k, v in HALO.items()},
               near_miss=dict(d_star=fin(d1), e_star=fin(e1), z_star=fin(z1)), control_no_halo=dict(d_star=fin(d0), e_star=fin(e0), z_star=fin(z0)),
               previous=dict(d_star=fin(old[0]), e_star=fin(old[1]), z_star=fin(old[2])),
               feasible=bool(np.isfinite(d1)), n_selection=len(sel), n_heldout=len(held),
               guard_triggers_selection=int((~sel.guard_reliable).sum()), quick=a.quick)
    (out / "decisions.json").write_text(json.dumps(dec, indent=2))
    print("new", (d1, e1, z1), "control", (d0, e0, z0), "old", old)
    print("guard triggers (selection patches)", dec["guard_triggers_selection"])
    print(hs.round(3).to_string())


if __name__ == "__main__":
    main()
