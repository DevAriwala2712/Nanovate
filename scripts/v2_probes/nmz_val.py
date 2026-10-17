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
pc = dataclasses.replace(load_pipeline_cfg(ExpCfg.full()), h1_fill="dielectric")
base = TestbedCfg(); win = window(base); rows=[]
for dose in (20, 50, 200):
  for sig in (0.5, 1.0, 1.5, 2.0):
    for gr in (0.15, 0.30):
      if gr == 0.30 and dose != 50: continue
      tc = with_dose(nuisance_cfg(base, sig, gr), dose); cal = calibrate(make_scribe_patches(tc, seed=0))
      for lay in range(8):
        L = get_layout("manhattan","val",lay,tc,0)
        for die in range(2):
          for kind in ("nuisance","notch","filament"):
            s = make_sample("manhattan","val",lay,die,kind,tc,0,layout=L)
            r = verify_patch(s.image, s.cad, cal, win, pc).reason
            rows.append(dict(dose=dose, sigma=sig, grain=gr, layout=lay, die=die, kind=kind, d_b=tuple(r["delta_betti"]),
                             bott=r["bottleneck_max"], energy=r["max_sdf_energy"], z=r["peak_z"]))
      print(dose, sig, gr, flush=True)
pd.DataFrame(rows).to_csv(f"{sys.argv[1]}/nmz_val.csv", index=False)
