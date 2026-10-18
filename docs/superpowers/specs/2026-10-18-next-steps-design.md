# Next steps after v2: prove the value, then move to real data (design)

Date: 2026-10-18. Status: **draft for approval**; nothing below is built yet.
Repo state: v2 at commit `58cf3b6` (pushed). All existing results are synthetic data.

## Intent and the why

An outside review of the repo concluded that the next milestone is proof, not more modules: say clearly why anyone
would choose this over existing tools, verify it independently, and test it on real SEM images. The user wants:

1. **The point made clear:** why this is *more desirable than the existing solution*, said in a way a fab engineer or
   a hackathon judge accepts. Existing review-SEM classification (KLA aiSIGHT, AMAT SEMVision ADC; see
   `docs/existing_semiconductor_inspection_systems_bottlenecks.md`) is a trained model. It needs labelled images
   per defect class and per layer, so on day 0 of a new layer it does not exist yet, and the fallback is die-to-die
   differencing plus manual review.
2. **Then real-world data**, starting with what can be done without a fab partner.

Done in this order: Part 1, then Part 2, then Part 3. Each part gets its own implementation plan
(superpowers:writing-plans) after this design is approved.

## The claim we will defend (and the one we will not)

**Defended:** "On day 0 of a new layer or layout, before any labelled images exist, the verifier gives usable killer
calls with a reason for each, from 3 scribe images and 43 ms of calibration. A trained CNN needs *N* labelled images of
that layout to catch up." *N* is measured in Part 1.

**Not claimed:** that the verifier beats a CNN trained on the same layout (it does not: Manhattan F1 0.81 vs 0.94), or
that it is fast (about 99 ms vs 2.5 ms per patch on CPU). Both stay stated in the README as they are now.

**Anti-goals:** a pitch that hides the in-distribution loss or the latency; any number tuned on test; any number in
the README typed by hand; claiming real-world validity from simulated or unpatterned-wafer data.

---

## Part 1 - The day-0 value case (synthetic data)

### 1a. Label break-even experiment ("how many labels is the verifier worth?")

**Question:** how many labelled images of a new layout family must be added to the Manhattan-trained CNN before its
killer F1 on that family matches the verifier's?

**Design:**
- Families: via (C) and curvy (B). Verifier reference = v2 Exp 3 test F1 at dose 50: via 0.88 [0.82, 0.93], curvy
  0.84 [0.77, 0.90] (n = 144 each). The verifier is not retrained or retuned.
- CNN: start from the existing Manhattan weights (`m6_cnn/cnn.pt`) and fine-tune on *k* labelled patches of the new
  family's **train** split, all six kinds, random dose 5-200 (same recipe as `make_cnn_train_set`).
  k in {0, 12, 24, 48, 96, 192, 384, 768}. This is the realistic vendor workflow (adapt an existing model).
- Threshold: chosen on 25% of the k labelled patches (held out from fine-tuning), so the labels the threshold uses
  are counted in k; for k = 0 the Manhattan-val threshold (0.39) is used, as in Exp 3. At k = 12 that is 3 patches;
  the noise this causes is real (a day-0 user has no more) and shows up in the seed spread. A second curve with the
  threshold fixed at 0.39 for every k is reported alongside, to separate threshold noise from learning.
- Rule check: AGENTS.md forbids tuning **the verifier** on families B/C. Here only the CNN baseline is adapted, and
  only on the B/C **train** split; the verifier and the test split are untouched. This is the experiment's premise
  (what a vendor would do with k labels), stated in config.json and the README.
- Seeds: 3 fine-tuning seeds per (family, k), because GPU training is not bit-reproducible and small k is noisy.
- Evaluation: the same 144 test patches per family as Exp 3 (dose 50), F1 with 95% bootstrap CI per seed, plus the
  mean over seeds.
- **Break-even k (pre-registered definition):** the smallest k whose seed-mean CNN F1 is >= the verifier's F1 point
  estimate. Also reported: the smallest k whose CNN CI lower bound exceeds the verifier's point estimate (CNN clearly
  better). If curvy breaks even at k = 0 (the CNN already ties there: 0.85 vs 0.84), that is reported as such.
- Compute: **Colab GPU.** About 48 fine-tunes, each small (at most 768 patches); the existing 1,152-patch, 60-epoch
  training took 212 s on Colab. A notebook `notebooks/breakeven_colab.ipynb` generates the data, fine-tunes, and saves
  weights + configs + loss histories; evaluation runs locally from the returned weights (same split as M6).
- Output: `results/v2/breakeven/` (CSV, plot of F1 vs k with the verifier as a band, config.json).

### 1b. Day-0 review-queue scenario (computed from measured rates)

A concrete picture of the use case, computed only from existing per-sample test results (`results/v2/exp3/samples.csv`,
via family, dose 50) and labelled as a **projection from synthetic rates**:
- A review SEM visits 1,000 flagged sites on a new via layer; killer prevalence p (default 5%, also 1% and 20%).
  Killers split equally into bridge / open / particle; non-killer sites are 90% nuisance and 10% near-miss
  (notch / filament equally). These mixes are assumptions, shown in the table header.
- For each method available on day 0 (die-to-die single/double, Manhattan-trained CNN, verifier), project: killers
  found, harmless sites sent to an engineer, near-misses flagged, and sites needing manual review. Use the measured
  per-kind rates; propagate the bootstrap CI.
- Also report what the CNN at the break-even k would do, to show the cost of waiting for labels.
- Output: `results/v2/day0_scenario/` + a README section "Why use this: the day-0 case" generated from the CSVs.

### 1c. The point, written down

A short README section and a one-page `docs/USE_CASE.md`: who uses it (process/yield engineer on a new layer ramp),
when (before labelled data exists; and later as a CAD-grounded cross-check that explains each call), what it replaces
(die-to-die + manual review on day 0), measured advantages and measured disadvantages side by side, and the break-even
number. Every number is pulled from CSVs by the report generator.

**Done for Part 1:** break-even k per family with CIs; scenario table; README + USE_CASE.md generated; tests for the
fine-tune/threshold helpers and the scenario arithmetic; PROGRESS.md entry; commit.

---

## Part 2 - Independent verification with a simulator we did not write (synthetic data)

**Why:** every result so far uses a forward model written by the same authors. Rendering the same ground-truth wafers
through an independent SEM simulator tests whether the verifier's behaviour depends on our simulator's quirks, while
keeping exact topology labels (they come from the wafer W, not the image).

**Design:**
- 2a. **Feasibility spike (throwaway, Colab):** Nebula (TU Delft, open-source Monte Carlo electron-matter simulator,
  GPU/CPU, SoftwareX 2020) as first choice; NIST ARTIMAGEN (public domain, phenomenological) as fallback. Build it,
  extrude one 256 x 256 CAD/W tile into a 3D line-on-substrate geometry, render one SE image, and time it. Report
  runtime per tile and what is needed (materials files, geometry format). **Adding either tool is a new dependency:
  the spike result comes back to the user for approval before anything enters the repo** (AGENTS.md section 8).
- 2b. After approval: an adapter `testbed/sim_external.py` that renders (CAD, W) to an image in e-/px at a chosen dose
  (Nebula gives detected electrons; dose = primary electrons per pixel), a harness flag selecting the forward model,
  and Exp 1 (LER sigma 1.0/1.5, grain 0/30%) and Exp 2 (dose 20/50/200) rerun on the test split **with the frozen v2
  config and no retuning**. Scribe patches are rendered by the same external simulator so Module 1 calibrates on it.
- Sample counts are bounded by simulation cost; fixed before the run once 2a gives the per-tile time.
- Output: `results/v2_sim2sim/`, labelled "synthetic data (independent simulator)".

**Done for Part 2:** spike report; on approval, the adapter, tests (adapter output units, label unchanged), the reruns
with CIs and n, and a PROGRESS entry comparing against the native testbed.

---

## Part 3 - Real SEM images (what public data allows)

**What exists:** no public dataset pairs real SEM images with CAD layouts and topology-checked labels. The usable open
set is **Carinthia-S** (Zenodo 16895427, CC BY 4.0, 139 MB): 4,591 real SEM defect images from one production layer of
*unstructured* wafers (no circuit pattern), 6 classes, expert binary masks.

**What it can test (and only this):**
- 3a. **Noise model on real images:** fit Module 1's Var = alpha * mu + sigma2 on flat background regions and check
  that the generalised Anscombe transform stabilises variance (residual variance vs mean before and after).
- 3b. **Particle branch on a known-empty layout:** CAD = all dielectric, so any conductor component is delta-beta0 > 0.
  Measure detection rate per defect class and the overlap of the detected component with the expert mask (IoU),
  and the false-detection rate on defect-free background crops. Calibration levels come from the images themselves
  (no scribe patches exist), which is a stated deviation.
- Pixel size, intensity units and the six class names are read from the dataset's description file before any run;
  if intensities are not in counts, 3a reports only relative variance behaviour.

**What it cannot test:** anything CAD-relative (bridges, opens, near-misses vs a layout). That needs a line/space set
with GDS clips (imec-type data) through a research partner, with labels from electrical test or voltage contrast.
Part 3 ends with a one-page data request describing exactly the inputs a pilot needs (GDS clip per site, review SEM
image with pixel size, 3 scribe images per layer at the same beam settings, electrical truth for a validation subset).

**Consent needed:** downloading Carinthia-S (139 MB, CC BY 4.0) is a download; it is asked for before it happens.
Real-data results go to `results/real_carinthia/`, labelled "real SEM, unpatterned wafers, no CAD".

**Done for Part 3:** noise-model check, particle-branch metrics with CIs and n, the data-request page, PROGRESS entry.

---

## Constraints

- AGENTS.md rules hold: no tuning on test or on families B/C for the verifier; one seed drives each run; CSV + PNG +
  config.json per result; README numbers generated from CSVs; missed targets kept with a one-line reason.
- Allowed dependencies only (numpy, scipy, scikit-image, gdstk, gudhi, torch, matplotlib, pandas, pytest). Nebula /
  ARTIMAGEN and any download are asked for first.
- Colab is used for GPU training (Part 1a) and the simulator spike (Part 2a). The user runs the notebooks and returns
  weights or images; notebooks are idempotent and clone the repo at a pinned commit.
- `pytest -q` green before every commit.

## Decided by the user

- Order: value case first, then independent simulator, then real SEM images.
- The point to make: why this is more desirable than the existing solution.
- Colab is available on request.

## Calls made by Claude (check these)

- The value claim is framed as **day-0 / new-layer**, not "better overall", because the measured results only support
  that.
- Break-even uses **fine-tuning the existing Manhattan CNN** (vendor-realistic) rather than training from scratch.
- Threshold labels are counted inside k (no free labelled validation set on day 0).
- 3 seeds per point; k grid {0 ... 768}.
- Default prevalence 5% in the scenario, with 1% and 20% shown; non-killers 90% nuisance / 10% near-miss.
- A second break-even curve with the threshold fixed at 0.39, to separate threshold noise from learning.
- Nebula before ARTIMAGEN.

## Left to the builder

File and function names inside the new modules; plot styling; exact fine-tuning schedule (epochs, learning rate),
fixed before any test number is seen and recorded in config.json.
