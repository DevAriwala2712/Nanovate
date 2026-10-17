"""M6: fit the die-to-die rules and train the CNN on Manhattan only; report val F1 and latency.

Synthetic data. Positive class = killer (bridge / open / particle); notch / filament / nuisance are
negatives. Rules and thresholds are chosen on Manhattan *val* (D2D: sigma/theta/min-area grid; CNN:
probability threshold). The CNN trains on Manhattan *train* with a random dose per sample from
TRAIN_DOSES. Val F1 below is therefore IN-SAMPLE for the selected rules (optimistic); the test split is
not touched here (Experiments 1-6, M7, report test numbers). Usage:
  python scripts/baselines_report.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.baselines.cnn import CnnCfg, load_cnn, pick_threshold, predict_proba, train_cnn
from nanovate.baselines.data import TRAIN_DOSES, make_cnn_train_set, with_dose
from nanovate.baselines.die_to_die import d2d_flag, d2d_score, d2d_stats, fit_d2d_rule
from nanovate.config import TestbedCfg, save_config
from nanovate.modules.calibration import calibrate
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

FIT_DOSES = (20, 50, 200)
SIGMAS, THETAS = (0.0, 1.0, 2.0), (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
AREAS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SEED = 0


def f1(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float((pred & y).sum())
    return 2 * tp / max(2 * tp + float((pred & ~y).sum()) + float((~pred & y).sum()), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/m6_baselines")
    ap.add_argument("--cnn-weights", default=None, help="evaluate weights from scripts/train_cnn.py instead of training here")
    a = ap.parse_args()
    n_train_lay, dies_train, epochs, n_val_lay = (6, 1, 4, 3) if a.quick else (24, 4, 10, 8)
    base = TestbedCfg()
    win = window(base)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- CNN: load weights trained elsewhere, or train here (CPU; slow) ----
    if a.cnn_weights:
        model, cnn_cfg, hist = load_cnn(a.cnn_weights)
        print(f"loaded CNN weights {a.cnn_weights} (epochs {cnn_cfg.epochs}, final loss {hist['loss'][-1]:.3f})", flush=True)
    else:
        t0 = time.time()
        imgs, cads, ys, _ = make_cnn_train_set(n_train_lay, dies_train, seed=SEED)
        print(f"train data: n={len(ys)} (killers {int(ys.sum())}) in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        model, hist = train_cnn(imgs, cads, ys, CnnCfg(epochs=epochs, seed=SEED))
        print(f"cnn trained in {time.time() - t0:.0f}s, loss {hist['loss'][0]:.3f} -> {hist['loss'][-1]:.3f}", flush=True)

    # ---- Manhattan val items at FIT_DOSES ----
    rows, st_single, st_double, v_img, v_cad = [], [], [], [], []
    d2d_ms = []
    for dose in FIT_DOSES:
        cfg = with_dose(base, dose)
        cal = calibrate(make_scribe_patches(cfg, seed=SEED))
        for lay in range(n_val_lay):
            L = get_layout("manhattan", "val", lay, cfg, SEED)
            refs = [make_sample("manhattan", "val", lay, 900 + j, "nuisance", cfg, SEED, layout=L).image for j in range(2)]
            for die in range(2):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, cfg, SEED, layout=L)
                    t1 = time.perf_counter()
                    sc1 = d2d_score(s.image, refs[:1], L.cad, cal, "single")
                    st1 = d2d_stats(sc1, win, SIGMAS, THETAS)
                    d2d_ms.append((time.perf_counter() - t1) * 1e3)
                    sc2 = d2d_score(s.image, refs, L.cad, cal, "double")
                    st2 = d2d_stats(sc2, win, SIGMAS, THETAS)
                    rows.append(dict(dose=dose, layout=lay, die=die, kind=kind, killer=s.killer))
                    st_single.append(st1); st_double.append(st2); v_img.append(s.image); v_cad.append(s.cad)
        print(f"val dose {dose} done", flush=True)
    df = pd.DataFrame(rows)
    y = df.killer.values

    # ---- fit rules on Manhattan val: per dose (recipe tuned per condition = best case for D2D) and pooled ----
    rules, rules_pooled = {}, {}
    for m, st in (("single", st_single), ("double", st_double)):
        def flags(rule, idx):
            i, j = SIGMAS.index(rule.sigma), THETAS.index(rule.theta)
            return np.array([st[k][i, j] >= rule.min_area for k in idx])
        rules_pooled[m] = fit_d2d_rule(st, list(y), "val", "manhattan", m, SIGMAS, THETAS, AREAS)
        df[f"d2d_{m}_pooled"] = flags(rules_pooled[m], range(len(df)))
        df[f"d2d_{m}"] = False
        rules[m] = {}
        for dose in FIT_DOSES:
            idx = np.flatnonzero(df.dose.values == dose)
            rules[m][dose] = fit_d2d_rule([st[k] for k in idx], list(y[idx]), "val", "manhattan", m, SIGMAS, THETAS, AREAS)
            df.loc[idx, f"d2d_{m}"] = flags(rules[m][dose], idx)
    probs = predict_proba(model, np.stack(v_img), np.stack(v_cad))
    thr = pick_threshold(probs, y, "val", "manhattan")
    df["cnn_prob"], df["cnn"] = probs, probs >= thr

    # ---- latency (batch 1 / 64), single process ----
    xi, xc = np.stack(v_img[:64]), np.stack(v_cad[:64])
    t1 = time.perf_counter(); [predict_proba(model, xi[k:k + 1], xc[k:k + 1], 1) for k in range(16)]
    cnn_b1 = (time.perf_counter() - t1) * 1e3 / 16
    t1 = time.perf_counter(); predict_proba(model, xi, xc, 64)
    cnn_b64 = (time.perf_counter() - t1) * 1e3 / 64

    # ---- summaries ----
    summ = []
    for m in ("d2d_single", "d2d_double", "d2d_single_pooled", "d2d_double_pooled", "cnn"):
        for dose in (*FIT_DOSES, "all"):
            g = df if dose == "all" else df[df.dose == dose]
            nu = g[(~g.killer) & (g.kind == "nuisance")]
            summ.append(dict(method=m, dose=dose, n=len(g), n_killer=int(g.killer.sum()), f1=f1(g[m].values, g.killer.values),
                             recall=float(g[g[m] & g.killer].shape[0] / max(g.killer.sum(), 1)),
                             nuisance_false_alarm=float(nu[m].mean()), recall_bridge=float(g[(g.kind == "bridge")][m].mean()),
                             recall_open=float(g[(g.kind == "open")][m].mean()), recall_particle=float(g[(g.kind == "particle")][m].mean())))
    S = pd.DataFrame(summ)
    df.to_csv(out / "val_items.csv", index=False); S.to_csv(out / "summary.csv", index=False)
    (out / "rules.json").write_text(json.dumps({"d2d_per_dose": {m: {str(d): dataclasses.asdict(r) for d, r in rr.items()} for m, rr in rules.items()},
                                                "d2d_pooled": {m: dataclasses.asdict(r) for m, r in rules_pooled.items()}, "cnn_threshold": thr,
                                                "cnn_history": hist, "latency_ms": {"cnn_batch1": cnn_b1, "cnn_batch64": cnn_b64,
                                                "d2d_single_per_patch_incl_alignment": float(np.median(d2d_ms))}}, indent=2))
    save_config({"testbed": dataclasses.asdict(base), "quick": a.quick, "n_train_layouts": n_train_lay, "dies_train": dies_train,
                 "epochs": epochs, "n_val_layouts": n_val_lay, "fit_doses": FIT_DOSES, "train_doses": TRAIN_DOSES,
                 "sigmas": SIGMAS, "thetas": THETAS, "areas": AREAS, "seed": SEED}, out / "config.json")
    print("\nper-dose rules:", {m: {d: (r.sigma, r.theta, r.min_area) for d, r in rr.items()} for m, rr in rules.items()})
    print("pooled rules:", {m: (r.sigma, r.theta, r.min_area) for m, r in rules_pooled.items()}, "| cnn threshold", round(thr, 3))
    print(f"\nMANHATTAN VAL (in-sample for the selected rules; n per dose = {len(df) // len(FIT_DOSES)})")
    print(S.round(2).to_string(index=False))
    print(f"\nlatency ms/patch: CNN batch1 {cnn_b1:.1f}, batch64 {cnn_b64:.1f}; D2D single (align+GAT+stats) {np.median(d2d_ms):.1f}")


if __name__ == "__main__":
    main()
