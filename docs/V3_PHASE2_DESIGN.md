# v3 Phase 2 design note: per-feature merge rule (draft for approval; synthetic data)

**Scope (approved at Stop Point 1).** Fix the *logic-limited* errors from `docs/V3_PHASE1_DIAGNOSIS.md`:
(1) merges decided by one global tau, (2) a seen merge cancelled by an unrelated event in the Delta-beta sum,
(3) filaments read as a split. Cuts and necks are *not* re-engineered. They get a reported detection limit in nm.

**Rule (new `modules/events.py`; v2 path untouched).**
1. *Seeds.* Each CAD component inside the relevance region is a seed. Its core is the component eroded by 1 nm.
2. *Merge events, per seed pair.* For each pair of seeds closer than `pair_dist` (a priori 30 nm, about 2 pitches; fixed, not
   searched), the observed merge level `s_ij` is the highest field level at which the two cores are 4-connected in the spliced
   field, inside a local corridor (pixels with d_i + d_j <= d_ij + 2 nm). The expected level `e_ij` is the same quantity on the
   field the verifier would see for a defect-free image (the template-anchored field of T(C) itself, i.e. zero residual), so a
   designed narrow gap gets a higher expectation than a wide one. Call a **bridge between i and j** iff
   `s_ij - e_ij >= m_merge`, optionally on the grain-normalised level `s_ij / body_ij` (selected on val, see below).
   Seeds already joined in CAD outside the corridor (loops) are handled by the H1 rule.
3. *Event set, no sum.* The verdict comes from the set of located events, not from Delta-beta counts: any bridge event or
   new-hole event -> KILLER_BRIDGE; else any split, new-component or lost-hole event -> KILLER_OPEN/PARTICLE. Every event
   carries its location, seeds, `s`, `e` and margin in the reason, plus the v2 Delta-beta for comparison.
4. *Splits, particles, holes.* Unchanged from v2 (global lifetime tau on H0 splits / new components and on H1), so open recall
   and the cut/neck trade-off stay where Phase 1 put them. A per-neck variant (`s_k` vs `e_k` with `m_cut`) is implemented
   and offered to val selection, but Phase 1 predicts it cannot beat the saddle floor.
5. *Filament read as split.* Hypothesis: a new component whose merge point sits on a CAD wire edge (expected field there
   >= `e_edge`) is a dimmed protrusion, not an isolated particle, so it is not counted as a split. Not yet measured. If val
   shows no cue separating it from particles (100% recall today), it is dropped and reported as a remaining error.

**Selection (Manhattan val only).** Grid over `m_merge`, normalisation on/off, `m_cut` on/off, `e_edge` on/off. Same
pre-registered criterion as v2: max killer F1 on val doses 20/50/200 pooled, subject to nuisance suppression >= the v2 floor.
Ties go to the simpler rule. Exp 1's condition is covered by adding val grain 0/15/30% cells to the selection set, as v2's
robust near-miss selection already does. d*, e*, z* are re-selected with v2's robust procedure, because the near-miss gate
now sits behind the new killer rule. Frozen to `results/v3/calib/decisions.json`, then STOP POINT 3.

**Evaluation.** Experiments 1-6 run once on a **fresh test seed** (test layouts and dies keyed by a new `test_seed`; val,
scribe, D2D rules and CNN threshold unchanged from seed 0). The results go to `results/v3/`, and a generated
"Results (v3)" README block sits above v2. Plus a **detection-limit table**: open recall per 1 nm cut-length bin and
notch "not called killer" per remaining-width bin, with the limit defined in advance as the shortest length from which every
bin has recall >= 90%. Baselines are re-scored on the same fresh test patches; the CNN is not retrained.

**Back-compat.** New `EventRuleCfg` dataclass passed alongside `PipelineCfg` (`rule="v2_global_tau" | "v3_events"`);
`PipelineCfg`, `ExpCfg` and their `to_dict()` are left byte-identical, so `reproduce.py --version v1/v2` still regenerates
identical CSVs and configs. v3 gets its own `ExpCfg` subclass carrying `test_seed`.

**Tests.** A synthetic dim bridge next to a spurious split is called KILLER_BRIDGE (no cancellation). A clean gap equals its
expectation (margin 0). A designed narrow gap does not fire. The v2 rule selected by config reproduces `verify_patch`
exactly. A shifted sample gives the same events. An LER-only sample gives no events.

**Expected outcome (hypothesis, not a target change).** The pre-registered targets stand. From the Phase 1 floors, noise-free
F1 >= 0.92 is **likely missed** (rough projection about 0.90, because cuts are untouched). Exp 1 bridge recall >= 90% is
plausible but uncertain (noise-free grain-30% separation is 7/96 errors; noisy is worse). Manhattan F1 >= 0.88 is plausible.
Cost: about 20 local corridor saddles per patch, estimated +30-60 ms on top of the 99 ms patch time; Exp 6 will report it.
