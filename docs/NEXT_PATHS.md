# Nanovate: Next Paths (post-v2 roadmap)

*Status as of v2 (synthetic data only). Every target below is a hypothesis to be measured and reported as met or missed, the same discipline the README already uses.*

---

## Where we stand

| Area | v2 measured | Realistic ceiling | v3 doc claim |
|---|---|---|---|
| Zero-shot killer F1 (via / curvy) | 0.88 / 0.84 | 0.90–0.93 | 0.96 |
| Manhattan killer F1 | 0.81 (CNN: 0.94) | ~0.90 | 0.97 |
| Noise-free ceiling | 0.84 | >0.92 needed for the above | not considered |
| Near-miss recall | 90% | ~95% | 97.4% |
| Nuisance suppression (LER 1.5 nm, grain 30%) | 88%, bridge recall 79% | ~95% / ~95% | 98.6% |
| Low-dose F1 (5 e⁻/px) | 0.67 | ~0.75 | 0.92 |
| Calibration | 43 ms | ~15 ms | 10 ms |
| Time per patch | 99 ms (CPU) | 5–10 ms (GPU, batched) | ~0.03 ms |
| Real SEM validation | none | one public benchmark | claimed |

**Key facts that drive the plan**

1. The verifier tops out at **0.84 F1 even on noise-free images**, so denoising will not raise the ceiling. The bottleneck is resolution and/or the single global τ.
2. The CNN beats us **in-domain**, at **low dose**, and on **speed**. We win on **unseen via layouts with zero labels** and **miss fewer killers** (2 vs 12 of 50).
3. We send **more** sites to review (179 vs 111). The pitch is "fewer escapes", not "less review".
4. The method needs **CAD aligned to every image**, and public SEM datasets almost never include CAD.

---

## Phase 0: BIOS hack (now, until submission)

**Goal:** an honest, polished demo. No new research.

- [ ] Freeze v2 (tag a release, e.g. `v2.0-hack`).
- [ ] Change the repo tagline from "revolutionary" to match the README.
- [ ] Rewrite Exp 3 wording: "via: +7.3 pts (improved)" instead of "-7.3 pts".
- [ ] Build the end-to-end demo app: live frontend, upload or select tile, verdict plus reason (Δβ, bottleneck, SDF energy).
- [ ] Demo script: one killer the CNN misses on via, one near-miss only we catch, **one case where we fail**.
- [ ] PPT Slide 5: impact = via result + killer-miss reduction. Risks = bigger review queue and the CNN's in-domain lead.
- [ ] Keep v3 numbers out of the deck. v3 stays a roadmap of directions only.

---

## Phase 1: Diagnose the ceiling (about 1–3 days)

**Question:** is the 0.84 noise-free ceiling caused by resolution or by our decision logic?

- [ ] Re-run the noise-free diagnostic at **1×, 2× and 3× finer sampling** (same physical defects, same blur in nm).
- [ ] Sweep defect size (cut length, gap, particle distance) and record recall versus physical size.
- [ ] Log every noise-free miss with its Δβ, field value at the gap and τ margin.

**Decision point**

- F1 rises clearly with sampling → misses are **resolution-limited**. Report a physical detection limit in nm and stop chasing them in software.
- F1 stays flat → misses are **logic-limited**. Go to Phase 2.

---

## Phase 2: Fix the core decision rule (about 1–3 weeks)

**Goal:** break the single-τ trade-off (dim bridges vs bloom fakes, bridges vs opens).

- [ ] **Per-gap / per-wire local thresholds.** Use the calibrated template T(C) to predict the expected field value at each CAD gap and wire neck. Call merged or cut relative to *that* expectation, not one global τ.
- [ ] **Full-filtration evidence.** Use the birth/death of the specific feature that merges or splits, instead of a single lifetime cutoff.
- [ ] Revisit grain handling: a dark grain on a narrow bridge should not cancel it (Exp 1, 79% bridge recall).
- [ ] Re-select all thresholds on **Manhattan val only**; evaluate on a **fresh test seed** (v2 already used the test set twice).

**Targets (hypotheses):** noise-free F1 ≥ 0.92; Exp 1 suppression ≥ 90% *and* bridge recall ≥ 90%; Manhattan F1 ≥ 0.88.

---

## Phase 3: Pseudo-label hybrid (about 2–3 weeks)

**Thesis:** the verifier is how you get an in-domain CNN on a new layer without hand labels.

1. Day 0 on a new layer: the verifier runs alone.
2. Its high-confidence verdicts become pseudo-labels.
3. Fine-tune the Manhattan CNN on those pseudo-labels only.
4. Topology stays on as an explainable veto when the two disagree.

- [ ] Run on synthetic **via** and **curvy** families.
- [ ] Measure the label break-even again: how many *hand* labels the hybrid needs to match a CNN trained on 192 true labels.
- [ ] Report failure mode: does the CNN inherit the verifier's systematic misses?

**Target:** hybrid matches or beats the CNN-with-192-labels F1 using ≤ 20 hand labels.

---

## Phase 4: Make real data possible (parallel with Phases 2–3)

**Blocker:** no CAD in public SEM data.

- [ ] Add a **reference-die mode**: build the structural reference from a clean die (or the median of many dies) instead of CAD. Binarise it into a pseudo-layout, compute its Betti numbers and SDF.
- [ ] Validate reference-die mode against CAD mode on synthetic data first (how much F1 do we lose?).
- [ ] Find a public SEM defect dataset with die-to-die references or repeated patterns. Check licence, pixel size and label quality.
- [ ] Contact the faculty / university SEM facility about imaging a few test structures (even non-semiconductor patterned samples help).

**Target:** one real-image benchmark against die-to-die and a CNN, with frozen thresholds, CIs and honest failure cases.

---

## Phase 5: Speed (after the method is settled)

- [ ] Profile Module 4 (76 ms of the 99 ms).
- [ ] Batch the field + persistence step on GPU, or run topology only on candidates that pass a cheap pre-filter.
- [ ] Use the stack path (K = 32) where same-layout peers exist.

**Target:** 5–10 ms per patch on a single GPU. Report honestly against the CNN's ~3 ms.

---

## Out of scope (keep as long-term R&D only)

- 3D Gaussian-splatting X-ray laminography (needs synchrotron-class hardware).
- Fab-wide ASPH-GNN / Causal-KAN root-cause engine (needs fab telemetry).
- 25-bin few-shot ADC (needs real labelled defects).
- F1 0.92 at 5 e⁻/px (the information is not in the image).
- Optical-discovery nuisance filtering (defects are unresolved 3×3 blurs; topology needs resolved images).

---

## Rules carried over

- Thresholds chosen on validation only; test touched once per version.
- Every number with n and a 95% CI.
- Targets reported as met or missed, never tuned toward.
- Every new version keeps the previous one in the README for comparison.
