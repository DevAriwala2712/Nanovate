"""Day-0 review-queue scenario: 1,000 flagged sites on a new via layer, projected from measured rates.

A **projection from synthetic rates**, not a measurement: per-kind flag rates of every method come from the
v2 Exp 3 test samples (via family, dose 50, ``results/v2/exp3/samples.csv``); the site mix is an assumption
stated in config.json. Killers split equally into bridge / open / particle; non-killer sites are 90% nuisance,
5% notch, 5% filament. For each method that exists on day 0 (verifier, die-to-die, Manhattan-trained CNN) and
each prevalence it reports killers caught / flagged as near-miss / missed, harmless sites sent as killers,
near-misses flagged and sites sent to an engineer, with 95% CIs from a bootstrap that resamples samples within
each kind. Writes results/v2/day0_scenario/{scenario.csv, config.json}.

``--version v3`` applies the same projection (same site mix, prevalences, bootstrap seed and n_boot) to the v3
fresh-test Exp 3 samples (``results/v3/exp3/samples.csv``, frozen v3 rule) and writes results/v3/day0/.
Usage: python -m experiments.day0_scenario [--quick] [--version v2|v3]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.common import ExpCfg, write_results
from experiments.select_config import results_root

SAMPLES_CSV = "results/v2/exp3/samples.csv"
SAMPLES_CSV_V3 = "results/v3/exp3/samples.csv"
OUT_DIR = {"v2": "day0_scenario", "v3": "day0"}   # results/<version>/<dir>/scenario.csv
KILLER_KINDS = ("bridge", "open", "particle")
SAFE_MIX = {"nuisance": 0.90, "notch": 0.05, "filament": 0.05}
NEAR_MISS_KINDS = ("notch", "filament")
PREVALENCES = (0.01, 0.05, 0.20)
N_SITES = 1000
METHODS = ("verifier", "d2d_single", "d2d_double", "cnn")
METRICS = ("killers_total", "killers_caught", "killers_flagged_near_miss", "killers_missed", "harmless_sent_as_killer",
           "near_miss_flagged", "nuisance_flagged_near_miss", "sites_to_engineer")


def project(rates: dict[str, dict[str, float]], prevalence: float, n_sites: int = N_SITES) -> dict[str, float]:
    """Expected site counts for one method. ``rates['flag'][kind]`` = P(flagged KILLER | kind),
    ``rates['nm'][kind]`` = P(NEAR_MISS verdict | kind) (0 for methods without that class)."""
    n = {k: prevalence * n_sites / len(KILLER_KINDS) for k in KILLER_KINDS}
    n.update({k: (1 - prevalence) * n_sites * w for k, w in SAFE_MIX.items()})
    f, m = rates["flag"], rates["nm"]
    caught = sum(n[k] * f[k] for k in KILLER_KINDS)
    k_nm = sum(n[k] * m[k] for k in KILLER_KINDS)
    harmless = sum(n[k] * f[k] for k in SAFE_MIX)
    nm_true = sum(n[k] * m[k] for k in NEAR_MISS_KINDS)
    nu_nm = n["nuisance"] * m["nuisance"]
    total = prevalence * n_sites
    return dict(killers_total=total, killers_caught=caught, killers_flagged_near_miss=k_nm, killers_missed=max(total - caught - k_nm, 0.0),
                harmless_sent_as_killer=harmless, near_miss_flagged=nm_true, nuisance_flagged_near_miss=nu_nm,
                sites_to_engineer=caught + k_nm + harmless + nm_true + nu_nm)


def _indicators(df: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "verifier":
        return df.verdict.str.startswith("KILLER").values, (df.verdict == "NEAR_MISS").values
    return df[method].astype(bool).values, np.zeros(len(df), bool)


def _rates(df: pd.DataFrame, flag: np.ndarray, nm: np.ndarray) -> dict[str, dict[str, float]]:
    kinds = df.kind.values
    return {"flag": {k: float(flag[kinds == k].mean()) for k in KILLER_KINDS + tuple(SAFE_MIX)},
            "nm": {k: float(nm[kinds == k].mean()) for k in KILLER_KINDS + tuple(SAFE_MIX)}}


def run(cfg: ExpCfg, samples_csv: str | None = None, family: str = "via") -> pd.DataFrame:
    """Project the review queue for ``cfg.version`` (v2 default inputs/outputs unchanged; v3 -> results/v3/day0/)."""
    samples_csv = samples_csv or (SAMPLES_CSV_V3 if cfg.version == "v3" else SAMPLES_CSV)
    df = pd.read_csv(samples_csv)
    df = df[(df.family == family) & (df.dose == cfg.nominal_dose)].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    by_kind = {k: np.flatnonzero(df.kind.values == k) for k in KILLER_KINDS + tuple(SAFE_MIX)}
    boots = [np.concatenate([rng.choice(ix, size=len(ix)) for ix in by_kind.values()]) for _ in range(cfg.n_boot)]
    rows = []
    for m in METHODS:
        if m != "verifier" and m not in df:
            continue
        flag, nm = _indicators(df, m)
        for p in PREVALENCES:
            point = project(_rates(df, flag, nm), p)
            bs = [project(_rates(df.iloc[b], flag[b], nm[b]), p) for b in boots]
            for metric in METRICS:
                v = np.array([b[metric] for b in bs])
                rows.append(dict(method=m, prevalence=p, metric=metric, mean=point[metric], ci_lo=float(np.percentile(v, 2.5)),
                                 ci_hi=float(np.percentile(v, 97.5)), n_samples=len(df), n_sites=N_SITES))
    tab = pd.DataFrame(rows)
    write_results(results_root(cfg) / OUT_DIR.get(cfg.version, "day0_scenario"), "scenario", tab,
                  {"exp_cfg": cfg.to_dict(), "samples_csv": samples_csv, "family": family, "dose": cfg.nominal_dose,
                   "killer_split": "equal bridge/open/particle", "safe_mix": SAFE_MIX, "prevalences": PREVALENCES, "n_sites": N_SITES,
                   "data": "projection from synthetic rates (not a measurement)"})
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--version", default="v2", choices=("v2", "v3"))
    a = ap.parse_args()
    run(ExpCfg.quick(a.version) if a.quick else ExpCfg.full(a.version))
