"""Rung 1d Part A: CD-offset cells in the testbed; design-CAD vs CD-corrected SDF energy; gate re-selected per variant.

Protocol exactly as docs/REAL_DATA_RUNG1D_PREREG.md Part A (committed cf6f258 before this ran). Synthetic Manhattan val.
V0 = Rung 1b/1c robust pipeline (energy vs design CAD); V1 = same + cd_correct (energy vs grow(CAD, b_hat)).
Both variants share the killer path (verified per patch); only energy / peak-z differ.
Writes results/real/rung1d/val/{selection_records.csv, heldout_records.csv, heldout_summary.csv, near_miss_grid_V*.csv,
decisions.json}. Usage: python -m experiments.rung1d_val [--quick] [--workers N]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.rung1c_val import HALO, _base, _ci, tcfg_for
from experiments.v3_select import near_miss
from nanovate.pipeline_robust import calibrate_robust, verify_patch_robust
from nanovate.testbed.cdbias import make_sample_cd, make_scribe_patches_cd
from nanovate.testbed.dataset import KINDS, get_layout
from nanovate.topo_utils import Window

OUT = Path("results/real/rung1d/val")
CD_CELLS = (-3.0, -1.5, 1.5, 3.0)
_CAL: dict = {}


def _record(spec: dict) -> dict:
    pc, rc, rb = _base()
    variants = {"V0": rb, "V1": dataclasses.replace(rb, cd_correct=True)}
    off = dataclasses.replace(pc, d_star=float("inf"), e_star=float("inf"), z_star=float("-inf"))
    t = tcfg_for(spec["dose"], spec["ler"], spec["grain"], spec["halo"])
    b = spec["cd"]
    key = (spec["dose"], spec["ler"], spec["grain"], spec["halo"], b)
    if key not in _CAL:
        P = make_scribe_patches_cd(t, 0, b)
        _CAL[key] = {n: calibrate_robust(P, r) for n, r in variants.items()}
    lay = get_layout("manhattan", "val", spec["layout"], t, 0)
    s, ok = make_sample_cd("manhattan", "val", spec["layout"], spec["die"], spec["kind"], t, 0, b, layout=lay)
    out = dict(spec, sigma=f"{spec['ler']}|{spec['halo']}|cd{b:g}", killer=s.label == "KILLER", topology_ok=ok)
    kf = []
    for n, r in variants.items():
        res = verify_patch_robust(s.image, s.cad, _CAL[key][n], Window(256, 16), off, rc, r)
        z = res.reason
        kf.append(str(z["verdict_unguarded"]).startswith("KILLER"))
        out.update({f"energy_{n}": z["max_sdf_energy"], f"peak_z_{n}": z["peak_z"], f"bottleneck_{n}": z["bottleneck_max"],
                    f"b_hat_{n}": _CAL[key][n].cd_bias, f"reliable_{n}": _CAL[key][n].reliable})
    out["killer_flag"] = kf[0]
    out["killer_paths_agree"] = kf[0] == kf[1]
    return out


def specs(layouts, doses, cells) -> list[dict]:
    return [dict(dose=float(d), ler=ler, grain=gr, halo=h, cd=float(cd), layout=L, die=die, kind=k)
            for (ler, gr, h, cd) in cells for d in doses for L in layouts for die in (0, 1) for k in KINDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    pc, rc, rb = _base()
    q = a.quick
    doses = (50,) if q else (20, 50, 200)
    base_cells = [(s, g, "nominal", 0.0) for s in ((1.0,) if q else (0.5, 1.0, 1.5)) for g in ((0.15,) if q else (0.0, 0.15, 0.3))]
    halo_cells = [(1.0, 0.15, h, 0.0) for h in ("H1", "H2", "H3", "H4")]
    cd_cells = [(1.0, 0.15, "nominal", b) for b in CD_CELLS]
    with ProcessPoolExecutor(a.workers) as ex:
        sel = pd.DataFrame(list(ex.map(_record, specs(range(1) if q else range(8), doses, base_cells + halo_cells + cd_cells), chunksize=4)))
        held = pd.DataFrame(list(ex.map(_record, specs(range(8, 10) if q else range(8, 24), (50,),
                                                         [(1.0, 0.15, "nominal", 0.0)] + halo_cells + cd_cells), chunksize=4)))
    sel.to_csv(out / "selection_records.csv", index=False)
    held.to_csv(out / "heldout_records.csv", index=False)
    excl_sel, excl_held = int((~sel.topology_ok).sum()), int((~held.topology_ok).sum())
    sel, held = sel[sel.topology_ok].reset_index(drop=True), held[held.topology_ok].reset_index(drop=True)
    assert sel.killer_paths_agree.all() and held.killer_paths_agree.all(), "killer path differs between variants"

    nominal = ((sel.ler == 1.0) & (sel.grain == 0.15) & (sel.halo == "nominal") & (sel.cd == 0.0)).values
    thr, rows = {}, []
    for n in ("V0", "V1"):
        sc = sel.rename(columns={f"bottleneck_{n}": "bottleneck", f"energy_{n}": "energy", f"peak_z_{n}": "peak_z"})[["bottleneck", "energy", "peak_z"]]
        d, e, z, G = near_miss(sel[["kind", "sigma", "grain"]], sel.killer_flag.values, sc, nominal)
        G.to_csv(out / f"near_miss_grid_{n}.csv", index=False)
        thr[n] = (d, e, z)
        al = (~held.killer_flag.values) & (held[f"bottleneck_{n}"].values >= d) & (held[f"energy_{n}"].values >= e) & (held[f"peak_z_{n}"].values >= z)
        nmk = held.kind.isin(["notch", "filament"]).values
        for cell, g in held.groupby("sigma"):
            ix = g.index.values
            for metric, mask in (("nuisance NEAR_MISS rate", (held.kind.values == "nuisance")), ("near-miss recall (notch+filament)", nmk)):
                m, lo, hi = _ci(al[ix][mask[ix]])
                rows.append(dict(variant=n, cell=cell, metric=metric, mean=m, ci_lo=lo, ci_hi=hi, n=int(mask[ix].sum())))
        m, lo, hi = _ci(al[nmk])
        rows.append(dict(variant=n, cell="POOLED", metric="near-miss recall (notch+filament)", mean=m, ci_lo=lo, ci_hi=hi, n=int(nmk.sum())))
        m, lo, hi = _ci(al[held.kind.values == "nuisance"])
        rows.append(dict(variant=n, cell="POOLED", metric="nuisance NEAR_MISS rate", mean=m, ci_lo=lo, ci_hi=hi, n=int((held.kind == "nuisance").sum())))
    hs = pd.DataFrame(rows)
    hs.to_csv(out / "heldout_summary.csv", index=False)
    pooled = hs[(hs.cell == "POOLED") & (hs.metric.str.startswith("near-miss"))].set_index("variant")["mean"]
    choice = "V1" if pooled["V1"] > pooled["V0"] + 0.01 else "V0"
    fin = lambda v: None if not np.isfinite(v) else float(v)  # noqa: E731
    bh = held[held.cd != 0].groupby("cd").b_hat_V1.first().to_dict()
    dec = dict(data="synthetic", split="val (Manhattan)", prereg="docs/REAL_DATA_RUNG1D_PREREG.md Part A", choice=choice,
               rule="higher pooled held-out near-miss recall; ties within 1 pt -> V0",
               pooled_heldout_nearmiss_recall={k: float(v) for k, v in pooled.items()},
               thresholds={n: dict(d_star=fin(v[0]), e_star=fin(v[1]), z_star=fin(v[2])) for n, v in thr.items()},
               robust_cfg=dataclasses.replace(rb, cd_correct=(choice == "V1")).to_dict(),
               base=json.loads(Path("results/real/rung1b/val/decisions.json").read_text())["base"],
               b_hat_by_true_cd_V1=bh, excluded_topology_changed=dict(selection=excl_sel, heldout=excl_held),
               n_selection=len(sel), n_heldout=len(held), halo_cells={k: dict(sigma_edge=v[0], a2=v[1]) for k, v in HALO.items()},
               cd_cells_nm=list(CD_CELLS), quick=q)
    (out / "decisions.json").write_text(json.dumps(dec, indent=2, default=float))
    print(json.dumps({k: dec[k] for k in ("choice", "pooled_heldout_nearmiss_recall", "thresholds", "b_hat_by_true_cd_V1", "excluded_topology_changed")}, indent=1, default=float))
    print(hs.round(3).to_string())


if __name__ == "__main__":
    main()
