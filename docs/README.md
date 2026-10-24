# docs/ index

Measured results live in the top-level [README](../README.md) (generated from the CSVs under `results/` by
`scripts/reproduce.py`). Every number in this folder outside those generated blocks is either a design note, a
pre-registration, or an aspirational claim. All measured numbers are **synthetic data** unless a document says
"real" (Track C).

## Measured work (design notes, pre-registrations, results write-ups)

| Document | What it is |
|---|---|
| [USE_CASE.md](USE_CASE.md) | The day-0 argument, with the generated v3 and v2 day-0 tables (synthetic). |
| [V2_FIXES.md](V2_FIXES.md) | What v2 changed after v1, why, and the val evidence (synthetic). |
| [V3_PHASE1_DIAGNOSIS.md](V3_PHASE1_DIAGNOSIS.md) | Why the v2 verifier tops out on noise-free images (val only). |
| [V3_PHASE2_DESIGN.md](V3_PHASE2_DESIGN.md) | Design note for the v3 per-gap merge rule. |
| [V3_PHASE2_VAL.md](V3_PHASE2_VAL.md) | v3 validation check before the fresh test seed. |
| [REAL_DATA_PREREG.md](REAL_DATA_PREREG.md) | Track C Rung 1 pre-registration (written before any real run). |
| [REAL_DATA_RUNG1.md](REAL_DATA_RUNG1.md) | Track C Rung 1 result on real SEM images (target missed; see the README status box). |
| [REAL_DATA_RUNG1B_PREREG.md](REAL_DATA_RUNG1B_PREREG.md) · [REAL_DATA_RUNG1B.md](REAL_DATA_RUNG1B.md) | Rung 1b: three robustness changes chosen on synthetic val, one real repeat. |
| [REAL_DATA_RUNG1C_PREREG.md](REAL_DATA_RUNG1C_PREREG.md) · [REAL_DATA_RUNG1C.md](REAL_DATA_RUNG1C.md) | Rung 1c: near-miss gate re-selected under a synthetic edge halo, one real repeat. |
| [REAL_DATA_RUNG1D_PREREG.md](REAL_DATA_RUNG1D_PREREG.md) · [REAL_DATA_RUNG1D.md](REAL_DATA_RUNG1D.md) | Rung 1d: testbed CD offset, CD-corrected SDF energy (rejected on val), one real repeat. |
| [B2_FAST_BETTI.md](B2_FAST_BETTI.md) | Fast Betti path acceptance test (failed; GUDHI stays in the hot path). |
| [LAB_REQUEST.md](LAB_REQUEST.md) | Track C Rung 2: draft request to fabricate and image a defect test chip (not sent). |
| [NEXT_PATHS.md](NEXT_PATHS.md) | Roadmap: done / in progress / next. |
| [superpowers/](superpowers/) | Design specs and implementation plans for the app and the day-0 case. |
| [AGENTS.md](AGENTS.md) | Working rules for coding agents on this repo. |

## Early aspirational design notes (NOT measured results)

These documents predate the prototype. **Early aspirational design notes; claims and numbers here are NOT measured
results. See README for measured results.** Their scorecards (F1, suppression rates, latencies, speed-ups) were never
measured and no number in the README comes from them.

| Document | Note |
|---|---|
| [Defect Detection System Plan v4.0.md](Defect%20Detection%20System%20Plan%20v4.0.md) | Scope document (Part A = what was built). Parts B/C carry the v3.0 plan's content as future R&D / parked claims. Header added. |
| `(v3.0) Defect Detection System Plan.docx` | Superseded plan. A .docx cannot carry the header line, so it is flagged here instead: **early aspirational design notes; claims and numbers in it are NOT measured results. See README for measured results.** |
| [surgical_experimental_protocol_zero_hour_universal_inspection.md](surgical_experimental_protocol_zero_hour_universal_inspection.md) | Math and reference-code sketches the testbed and modules started from. Header added. |
| [existing_semiconductor_inspection_systems_bottlenecks.md](existing_semiconductor_inspection_systems_bottlenecks.md) | Background on existing inspection systems. Header added. |
