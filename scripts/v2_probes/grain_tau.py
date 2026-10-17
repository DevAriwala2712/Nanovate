"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
import sys, dataclasses, numpy as np, pandas as pd
from experiments.harness import nuisance_cfg
from experiments.select_config import with_dose, load_pipeline_cfg
from experiments.common import ExpCfg
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.pipeline import verify_patch
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
pc0 = load_pipeline_cfg(ExpCfg.full()); base = TestbedCfg(); win = window(base); rows = []
for gr in (0.0, 0.15, 0.30):
  for dose in (20, 200):
    tc = with_dose(nuisance_cfg(base, 1.0, gr), dose); cal = calibrate(make_scribe_patches(tc, seed=0))
    for lay in range(8):
        L = get_layout("manhattan", "val", lay, tc, 0)
        for die in range(2):
            for kind in ("nuisance", "bridge", "open", "particle", "filament"):
                s = make_sample("manhattan", "val", lay, die, kind, tc, 0, layout=L)
                for tau in (0.35, 0.45, 0.55):
                    v = verify_patch(s.image, s.cad, cal, win, dataclasses.replace(pc0, tau=tau)).verdict
                    rows.append(dict(grain=gr, dose=dose, kind=kind, tau=tau, verdict=v))
  print(gr, flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{sys.argv[1]}/grain_tau.csv", index=False)
d["killer"] = d.verdict.str.startswith("KILLER"); d["bridge_v"] = d.verdict == "KILLER_BRIDGE"
print(d.pivot_table(index=["grain", "tau"], columns="kind", values="killer").round(2))
