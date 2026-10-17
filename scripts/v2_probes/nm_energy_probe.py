"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
import sys, numpy as np, pandas as pd
from scipy import ndimage as ndi
from experiments.harness import nuisance_cfg
from experiments.select_config import with_dose
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.align_sdf import cad_sdf, align_to_template
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
base = TestbedCfg(); win = window(base)
MS = (0, 1, 2, 3, 4, 5, 6)
rows = []
for dose in (20, 50, 200):
 for sig in (0.5, 1.0, 1.5, 2.0, 2.5):
  for gr in (0.15, 0.30):
    if dose != 50 and gr != 0.15: continue
    tc = with_dose(nuisance_cfg(base, sig, gr), dose)
    cal = calibrate(make_scribe_patches(tc, seed=0))
    for lay in range(8):
      L = get_layout("manhattan", "val", lay, tc, 0); phi = cad_sdf(L.cad); T = cal.predict(L.cad)
      for die in range(2):
        for kind in ("nuisance", "notch", "filament"):
          smp = make_sample("manhattan", "val", lay, die, kind, tc, 0, layout=L)
          al, _ = align_to_template(smp.image, T)
          s = win.crop(np.abs(ndi.gaussian_filter(al - T, 1.0)) / abs(cal.mu_cond - cal.mu_diel))
          p = np.abs(win.crop(phi))
          r = dict(dose=dose, sigma=sig, grain=gr, layout=lay, die=die, kind=kind)
          for m in MS:
              w = np.exp(0.2 * np.minimum(p, 10.0)) * (p >= m)
              r[f"e{m}"] = float((s * w).max())
              r[f"s{m}"] = float(s[p >= m].max())
          rows.append(r)
    print(dose, sig, gr, flush=True)
pd.DataFrame(rows).to_csv(f"{sys.argv[1]}/nm_probe2.csv", index=False)
