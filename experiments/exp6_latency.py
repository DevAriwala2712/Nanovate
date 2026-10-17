"""Exp 6: calibration time and per-patch latency (CPU, this machine; synthetic data).

Calibration: median of ``exp6_calib_runs`` runs on 3 scribe patches (stated target < 50 ms). Per-patch: ms per
module of the verifier over ``exp6_patches`` patches (alignment, SDF energy, topology; the Module-2 stack cost
per patch for a K = 32 stack), die-to-die, and the CNN at batch 1 and 64 with 1 thread and all threads.
The verifier modules are single-threaded (scipy / GUDHI / numpy FFT); thread counts are reported for the CNN.
Timings are NOT bit-reproducible and are excluded from the reproducibility check.
Usage: python -m experiments.exp6_latency [--quick]
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import pandas as pd
import torch

from experiments.common import ExpCfg, bootstrap_ci, write_results
from experiments.harness import Context
from experiments.select_config import results_root, with_dose
from nanovate.baselines.cnn import predict_proba
from nanovate.modules.calibration import calibrate
from nanovate.pipeline import verify_stack
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    dose = cfg.nominal_dose
    tcfg = with_dose(ctx.base, dose)
    rows = []

    def add(what: str, vals, unit="ms", **kw):
        vals = np.asarray(vals, dtype=float)
        mean, lo, hi = bootstrap_ci(vals, cfg.n_boot, cfg.seed)
        rows.append(dict(what=what, unit=unit, median=float(np.median(vals)), mean=mean, ci_lo=lo, ci_hi=hi, n=len(vals), **kw))

    patches = make_scribe_patches(tcfg, seed=cfg.seed)
    runs = []
    for _ in range(cfg.exp6_calib_runs):
        t0 = time.perf_counter(); calibrate(patches); runs.append((time.perf_counter() - t0) * 1e3)
    add("calibration (3 scribe patches, with registration)", runs, target_ms=50.0)
    runs = []
    for _ in range(cfg.exp6_calib_runs):
        t0 = time.perf_counter(); calibrate(patches, register=False); runs.append((time.perf_counter() - t0) * 1e3)
    add("calibration (3 scribe patches, no registration)", runs, target_ms=50.0)

    n = cfg.exp6_patches
    specs = [dict(family="manhattan", split="test", layout_id=i % cfg.layouts, die=i // cfg.layouts, kind=KINDS[i % len(KINDS)], tcfg=tcfg,
                  rule_dose=dose) for i in range(n)]
    df = ctx.rows(specs)
    for col, label in (("ms_align", "verifier: Module 3 alignment"), ("ms_energy", "verifier: Module 3 SDF energy"), ("ms_topology", "verifier: Module 4 topology (+ field)")):
        add(label, df[col])
    add("verifier: total per patch (singleton path)", df.ms_align + df.ms_energy + df.ms_topology)
    add("die-to-die single (align + GAT + stats)", df.ms_d2d)
    add("cnn batch 1 (default threads)", df.ms_cnn)

    # Module 2 stack cost per patch, K = 32 same-layout dies
    L = get_layout("manhattan", "test", 0, tcfg, cfg.seed)
    stack = [make_sample("manhattan", "test", 0, k, "nuisance", tcfg, cfg.seed, layout=L).image for k in range(32)]
    cal = ctx.cal(tcfg)
    from dataclasses import replace
    pcs = replace(ctx.pc, use_stack=True)
    t0 = time.perf_counter(); verify_stack(stack, L.cad, cal, ctx.win, pcs); add("verifier: stack path total per patch (K=32)", [(time.perf_counter() - t0) * 1e3 / 32])

    # CNN latency, batch 1 and 64, 1 thread vs all threads
    imgs = np.stack(stack[:32] + stack[:32]); cads = np.stack([L.cad] * 64)
    default_threads = torch.get_num_threads()
    for th in (1, default_threads):
        torch.set_num_threads(th)
        b1 = []
        for k in range(20):
            t0 = time.perf_counter(); predict_proba(ctx.cnn, imgs[k:k + 1], cads[k:k + 1], 1); b1.append((time.perf_counter() - t0) * 1e3)
        add(f"cnn batch 1, {th} thread(s)", b1)
        b64 = []
        for _ in range(5):
            t0 = time.perf_counter(); predict_proba(ctx.cnn, imgs, cads, 64); b64.append((time.perf_counter() - t0) * 1e3 / 64)
        add(f"cnn batch 64, {th} thread(s) (per patch)", b64)
    torch.set_num_threads(default_threads)
    tab = pd.DataFrame(rows)
    cal_ms = tab.loc[tab.what.str.startswith("calibration (3 scribe patches, with"), "median"].iloc[0]
    tab["target_met"] = np.where(tab.what.str.startswith("calibration (3 scribe patches, with"), cal_ms < 50.0, np.nan)
    out = results_root(cfg) / "exp6"
    write_results(out, "latency", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "dose": dose, "data": "synthetic",
                                       "machine": {"platform": platform.platform(), "processor": platform.processor(), "torch_threads": default_threads},
                                       "target": "calibration < 50 ms", "note": "timings are not bit-reproducible"})
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
