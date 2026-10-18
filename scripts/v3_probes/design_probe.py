"""v3 Phase 2 design probe (Manhattan VAL only, noise-free, frozen v2 field). Synthetic data.

Is a merge cue that normalises the defect saddle by the defect's own body level robust to grain?
Prints, per grain level, separability (fewest errors of any threshold, AUC) of bridge vs filament for:
  raw saddle s, excess over the template-expected saddle s - e, and normalised s / body.
Also the template-expected saddle and observed saddle for particles vs filaments read as split.
Usage: python scripts/v3_probes/design_probe.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.select_config import with_dose  # noqa: E402
from experiments.v3_phase1 import frozen_pc  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.diagnostics import defect_saddle, verify_scaled  # noqa: E402
from nanovate.modules.calibration import calibrate, features  # noqa: E402
from nanovate.modules.fields import field_template  # noqa: E402
from nanovate.testbed.dataset import window  # noqa: E402
from nanovate.testbed.resample import make_fine_sample, make_fine_scribe_patches  # noqa: E402


def sep(v, y):
    best = max(int(((v >= t) == y).sum()) for t in np.unique(v))
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in v[y] for b in v[~y]]))
    return len(v) - best, round(auc, 3)


pc = frozen_pc()
rows = []
for grain in (0.15, 0.30):
    t = with_dose(TestbedCfg(), 50.0)
    t = dataclasses.replace(t, nuisance=dataclasses.replace(t.nuisance, grain_contrast=grain))
    cal = calibrate(make_fine_scribe_patches(t, 0, 1), sigmas=(1.5, 4.0))
    for lay in range(24):
        for die in range(2):
            for kind in ("bridge", "filament", "particle"):
                s = make_fine_sample("manhattan", "val", lay, die, kind, t, 0, 1)
                r = verify_scaled(s.expected, s.cad, cal, window(t), pc, 1)
                tmpl = cal.predict(s.cad)
                fexp = field_template(tmpl, tmpl, features(s.cad, *cal.sigmas)[..., 1], cal.mu_cond - cal.mu_diel)
                d = s.params["defect"]
                so = defect_saddle(kind, r.field, s.cad, s.W, s.mask, d, t.layout.pad)
                se = defect_saddle(kind, fexp, s.cad, s.W, s.mask, d, t.layout.pad)
                body = float(np.median(r.field[s.mask])) if s.mask.any() else np.nan
                rows.append(dict(grain=grain, kind=kind, verdict=r.verdict, db0=r.delta_beta[0], s=so["saddle"], e=se["saddle"],
                                 peak=so["peak"], body=body))
df = pd.DataFrame(rows)
out = Path("results/v3/probes"); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "design_probe.csv", index=False)
for g, d in df.groupby("grain"):
    m = d[d.kind.isin(["bridge", "filament"])].dropna(subset=["s"])
    y = (m.kind == "bridge").values
    print(f"grain {g:.2f}: bridge vs filament (n={len(m)}) min-errors/AUC  raw {sep(m.s.values, y)}  excess {sep((m.s - m.e).values, y)}  "
          f"normalised {sep((m.s / m.body).values, y)}")
    print("   bridge saddle q05/50/95", np.round(m[y].s.quantile([.05, .5, .95]).values, 2), " filament", np.round(m[~y].s.quantile([.05, .5, .95]).values, 2))
    sp = d[(d.kind == "filament") & (d.db0 > 0)]
    pa = d[d.kind == "particle"]
    print(f"   filament read as split n={len(sp)}: peak-s {np.round((sp.peak - sp.s).values, 2) if 'peak' in sp else ''}")
    print(f"   particle (n={len(pa)}) peak-s q05/50/95 {np.round((pa.peak - pa.s).quantile([.05, .5, .95]).values, 2)}  s-e q05/50/95 {np.round((pa.s - pa.e).quantile([.05, .5, .95]).values, 2)}")
