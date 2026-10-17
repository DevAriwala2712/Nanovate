# Nanovate
A revolutionary semiconductor defect system

## Review UI
`frontend/index.html` views real testbed samples with their Module 4 verdicts, ground-truth labels, and patch layers (CAD, W, SEM, conductor field).
```
python scripts/export_ui.py            # writes frontend/data/run.js (synthetic data)
python3 -m http.server --directory frontend   # or just open frontend/index.html
```
The export uses Module 4 as implemented, with its defaults unless flags are given (`--tau`, `--field`, `--pre-smooth`); it is not tuned. Module 1–3 outputs (SDF energy, calibration, low-rank) show as unavailable until those modules exist.
