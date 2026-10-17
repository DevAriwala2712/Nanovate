"""M1 testbed report: resample statistics (AGENTS.md §7.6) and a visual gallery.

Writes results/testbed/{resamples.csv, gallery.png, config.json}. Synthetic data.
Usage: python scripts/testbed_report.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.layouts import FAMILIES

OUT = Path("results/testbed")


def resample_table(cfg: TestbedCfg, seed: int, n_layouts: int, sigmas: list[float]) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        lays = [get_layout(fam, "val", i, cfg, seed) for i in range(n_layouts)]
        for sig in sigmas:
            c = dataclasses.replace(cfg, nuisance=dataclasses.replace(cfg.nuisance, ler_sigma=sig))
            for kind in KINDS:
                n_res, n_ler, n_site, n_topo, fail, t = 0, 0, 0, 0, 0, time.perf_counter()
                for i, lay in enumerate(lays):
                    try:
                        s = make_sample(fam, "val", i, 0, kind, c, seed, layout=lay)
                    except RuntimeError:
                        fail += 1
                        continue
                    r = s.params["resample_reasons"]
                    n_res += s.resamples
                    n_ler += r.get("ler_topology", 0)
                    n_site += r.get("no_site", 0)
                    n_topo += sum(v for k, v in r.items() if k.startswith("topo_"))
                n_ok = n_layouts - fail
                rows.append(dict(family=fam, ler_sigma=sig, kind=kind, n_samples=n_ok, n_failed=fail,
                                 resamples=n_res, resamples_per_sample=n_res / max(n_ok, 1),
                                 ler_topology_rejects=n_ler, no_site_rejects=n_site, defect_topology_rejects=n_topo,
                                 ms_per_sample=1e3 * (time.perf_counter() - t) / n_layouts))
                print(rows[-1])
    return pd.DataFrame(rows)


def gallery(cfg: TestbedCfg, seed: int, path: Path) -> None:
    fig, ax = plt.subplots(2 * len(FAMILIES), len(KINDS), figsize=(2.6 * len(KINDS), 5.4 * len(FAMILIES)))
    m = cfg.layout.margin
    for i, fam in enumerate(FAMILIES):
        for j, kind in enumerate(KINDS):
            s = make_sample(fam, "val", 0, j, kind, cfg, seed)
            a, b = ax[2 * i, j], ax[2 * i + 1, j]
            a.imshow(s.W, cmap="gray")
            a.contour(s.cad, [0.5], colors="c", linewidths=0.4)
            if s.mask.any():
                a.contour(s.mask, [0.5], colors="r", linewidths=0.8)
            a.set_title(f"{fam}/{kind}\nW, dβ=({s.topo['d_beta0']},{s.topo['d_beta1']})", fontsize=7)
            b.imshow(s.image, cmap="gray")
            b.set_title(f"SEM λ={s.dose:g} e⁻/px", fontsize=7)
            for x in (a, b):
                x.axis("off")
                x.add_patch(plt.Rectangle((m - .5, m - .5), 256 - 2 * m, 256 - 2 * m, fill=False, ec="y", lw=.5))
    fig.suptitle("Synthetic testbed (synthetic data). cyan = CAD edge, red = defect mask, yellow = central window", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = TestbedCfg()
    OUT.mkdir(parents=True, exist_ok=True)
    n = 8 if a.quick else 40
    df = resample_table(cfg, a.seed, n, [0.5, 1.0, 1.5, 2.0, 2.5])
    df.to_csv(OUT / "resamples.csv", index=False)
    gallery(cfg, a.seed, OUT / "gallery.png")
    save_config({"testbed": dataclasses.asdict(cfg), "seed": a.seed, "n_layouts_per_cell": n,
                 "note": "synthetic data"}, OUT / "config.json")


if __name__ == "__main__":
    main()
