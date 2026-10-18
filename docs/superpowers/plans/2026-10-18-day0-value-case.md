# Day-0 Value Case (Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how many labelled images of a new layout the CNN needs to match the verifier (break-even k), project a day-0 review queue from measured rates, and generate the use-case write-up from CSVs.

**Architecture:** Training runs on Colab (`scripts/breakeven_train.py` via `notebooks/breakeven_colab.ipynb`) and returns weights + a manifest. Everything else runs locally and deterministically: `experiments/breakeven.py` evaluates the weights on the Exp 3 test patches, `experiments/day0_scenario.py` projects the queue from `results/v2/exp3/samples.csv`, and `experiments/report.py` renders the README / `docs/USE_CASE.md` blocks.

**Tech Stack:** Python 3.11, numpy, pandas, torch, matplotlib, pytest (no new dependencies).

**Spec:** `docs/superpowers/specs/2026-10-18-next-steps-design.md` (Part 1)

## Global Constraints

- Families: via and curvy. k grid: {0, 12, 24, 48, 96, 192, 384, 768}. Seeds: {0, 1, 2}.
- Fine-tune starts from `m6_cnn/cnn.pt`; pool = the family's **train** split, all six kinds, random dose from `TRAIN_DOSES`.
- Threshold labels are inside k: 25% of the k patches (rounded up) choose the threshold, 75% fine-tune; k = 0 uses 0.39.
- A second curve uses threshold 0.39 for every k.
- Evaluation: the same 144 test patches per family as Exp 3 (dose 50, 12 layouts x 2 dies x 6 kinds), F1 + 95% bootstrap CI (1,000 resamples).
- Verifier reference = v2 Exp 3 F1 from `results/v2/exp3/geometry_f1.csv`; the verifier is not rerun or retuned.
- Break-even k = smallest k with seed-mean CNN F1 >= verifier F1 point estimate; "clearly better" k = smallest k with every seed's CI lower bound > verifier point estimate. Report "none" if not reached.
- Fine-tune schedule fixed now: Adam lr 5e-4, batch 32, 300 optimiser steps total (epochs = ceil(300 / ceil(n_ft / 32))), augment on.
- Scenario: 1,000 sites; prevalence {0.01, 0.05, 0.20}; killers split equally bridge/open/particle; non-killers 90% nuisance, 5% notch, 5% filament.
- Allowed deps only; no number typed into README or USE_CASE.md by hand; every result labelled "synthetic data"; `pytest -q` green before each commit.

## Review Focus

- k smaller than 4 after the 25% split (k = 12 gives 9 / 3): the threshold subset may hold no killer; `pick_threshold_labelled` must then fall back to 0.39, not crash or return 1.0.
- A manifest whose weights file is missing (partial Colab run): local evaluation must skip that row and report it as missing, not crash.
- Fine-tuning must not mutate the loaded Manhattan model shared across runs (each fine-tune deep-copies it).
- Scenario with a method column absent from samples.csv (e.g. CNN not run): that method is skipped, the others still render.
- Re-running the report must be idempotent (USE_CASE.md markers replaced, not duplicated).

---

### Task 1: Family-aware train set and warm-start fine-tuning

**Files:**
- Modify: `src/nanovate/baselines/data.py` (`make_cnn_train_set`)
- Modify: `src/nanovate/baselines/cnn.py` (`train_cnn`)
- Test: `tests/test_cnn.py`

**Interfaces:**
- Produces: `make_cnn_train_set(n_layouts, dies, seed=0, doses=TRAIN_DOSES, cfg=None, family="manhattan", split="train")` (same return); `train_cnn(images, cads, labels, cfg=CnnCfg(), init: ResNetSmall | None = None) -> (ResNetSmall, dict)`. With `init`, training starts from a deep copy of `init`'s weights; `init` is unchanged afterwards.

- [ ] **Step 1: Write failing tests**

```python
def test_train_set_family_argument():
    imgs, cads, ys, ds = make_cnn_train_set(1, 1, seed=0, family="via")
    assert imgs.shape == (6, 256, 256) and ys.sum() == 3
    ref = make_cnn_train_set(1, 1, seed=0)   # default stays Manhattan, bit-identical to before
    assert not np.array_equal(ref[1], cads)

def test_warm_start_does_not_mutate_init_and_starts_from_it():
    imgs, cads, ys, _ = make_cnn_train_set(1, 1, seed=0)
    base, _ = train_cnn(imgs, cads, ys, CnnCfg(epochs=1))
    before = {k: v.clone() for k, v in base.state_dict().items()}
    ft, hist = train_cnn(imgs, cads, ys, CnnCfg(epochs=1, lr=1e-9), init=base)
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    # lr ~ 0: fine-tuned weights stay at the init (BatchNorm running stats may move; compare conv weights)
    assert torch.allclose(ft.stem[0].weight, base.stem[0].weight, atol=1e-6)
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q -o addopts="" tests/test_cnn.py` - expect the two new tests to FAIL (unexpected keyword).
- [ ] **Step 3:** Implement: `family`/`split` passed to `make_sample`; in `train_cnn`, `model = copy.deepcopy(init).to(dev) if init is not None else ResNetSmall(cfg.width).to(dev)`.
- [ ] **Step 4:** Run the full `tests/test_cnn.py` - expect PASS (existing CNN tests unchanged).
- [ ] **Step 5:** Commit `feat(baselines): family-aware CNN train set and warm-start fine-tuning`.

### Task 2: Break-even helpers (subsets, threshold, schedule)

**Files:**
- Create: `src/nanovate/baselines/breakeven.py`
- Test: `tests/test_breakeven.py`

**Interfaces:**
- Produces:
  - `K_GRID = (0, 12, 24, 48, 96, 192, 384, 768)`, `SEEDS = (0, 1, 2)`, `DEFAULT_THRESHOLD = 0.39`, `FT_STEPS = 300`, `FT_LR = 5e-4`
  - `breakeven_subsets(n_pool: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]`: (fine-tune indices, threshold indices) into the pool; a seeded permutation, first k taken, last `ceil(k/4)` of those are the threshold set; k = 0 returns two empty arrays; raises `ValueError` if k > n_pool.
  - `pick_threshold_labelled(probs: np.ndarray, labels: np.ndarray, default: float = DEFAULT_THRESHOLD) -> float`: max-F1 threshold (ties: lowest); returns `default` when `labels` is empty or holds no positive.
  - `finetune_epochs(n_ft: int, steps: int = FT_STEPS, batch: int = 32) -> int`: `ceil(steps / ceil(n_ft / batch))`, 0 when n_ft = 0.

- [ ] **Step 1: Write failing tests**

```python
def test_subsets_sizes_disjoint_nested_and_seeded():
    ft, th = breakeven_subsets(768, 12, seed=0)
    assert len(ft) == 9 and len(th) == 3 and not set(ft) & set(th)
    ft2, th2 = breakeven_subsets(768, 12, seed=0)
    assert np.array_equal(ft, ft2) and np.array_equal(th, th2)
    assert not np.array_equal(breakeven_subsets(768, 12, seed=1)[0], ft)
    bft, bth = breakeven_subsets(768, 24, seed=0)
    assert set(np.r_[ft, th]) <= set(np.r_[bft, bth])                 # nested in k for a seed
    assert all(len(a) == 0 for a in breakeven_subsets(768, 0, seed=0))
    with pytest.raises(ValueError):
        breakeven_subsets(10, 12, seed=0)

def test_threshold_fallbacks():
    assert pick_threshold_labelled(np.array([]), np.array([], bool)) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.9]), np.array([False, False])) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.6, 0.9]), np.array([False, True, True])) == 0.6

def test_finetune_epochs():
    assert finetune_epochs(0) == 0 and finetune_epochs(9) == 300 and finetune_epochs(576) == 17
```

- [ ] **Step 2:** Run `tests/test_breakeven.py` - expect FAIL (module missing).
- [ ] **Step 3:** Implement `breakeven.py` with the signatures above. Nesting rule: for one seed, the permutation is the same for every k (`np.random.default_rng((seed, 7231)).permutation(n_pool)`), so the subset for k is the first k entries.
- [ ] **Step 4:** Run - expect PASS.
- [ ] **Step 5:** Commit `feat(breakeven): subset, threshold and schedule helpers`.

### Task 3: Colab training script and notebook

**Files:**
- Create: `scripts/breakeven_train.py`
- Create: `notebooks/breakeven_colab.ipynb`
- Test: `tests/test_breakeven.py` (smoke)

**Interfaces:**
- Consumes: Task 1 `make_cnn_train_set(..., family=)`, `train_cnn(..., init=)`; Task 2 helpers; `load_cnn`, `save_cnn`.
- Produces: `<out>/manifest.json` = `{"base_weights": "m6_cnn/cnn.pt", "pool": {"layouts": int, "dies": int, "seed": 0}, "runs": [{"family", "k", "seed", "weights": "<family>_k<k>_s<seed>.pt" | null, "ft_idx": [...], "thr_idx": [...], "epochs", "lr", "steps", "final_loss"}], "torch", "device"}` and one weights file per run with k > 0. CLI: `python scripts/breakeven_train.py --families via,curvy --k 0,12,...,768 --seeds 0,1,2 --pool-layouts 32 --pool-dies 4 --device auto --out <dir>`; pool = 32 x 4 x 6 = 768 patches per family.

- [ ] **Step 1: Write the failing smoke test** (CPU, tiny):

```python
def test_breakeven_train_smoke(tmp_path):
    out = tmp_path / "be"
    subprocess.run([sys.executable, "scripts/breakeven_train.py", "--families", "via", "--k", "0,12",
                    "--seeds", "0", "--pool-layouts", "2", "--pool-dies", "1", "--steps", "2",
                    "--device", "cpu", "--out", str(out)], check=True)
    m = json.loads((out / "manifest.json").read_text())
    runs = {(r["family"], r["k"]): r for r in m["runs"]}
    assert runs[("via", 0)]["weights"] is None
    assert (out / runs[("via", 12)]["weights"]).exists() and len(runs[("via", 12)]["thr_idx"]) == 3
```

- [ ] **Step 2:** Run it - expect FAIL (script missing).
- [ ] **Step 3:** Implement the script: generate each family's pool once, load the base model once, and per (k, seed) fine-tune a deep copy on `pool[ft_idx]` with `CnnCfg(epochs=finetune_epochs(len(ft_idx), steps), lr=FT_LR, seed=seed, device=device)`; write the manifest after every run (a partial run leaves a valid manifest). `--steps` defaults to `FT_STEPS`.
- [ ] **Step 4:** Run the smoke test - expect PASS.
- [ ] **Step 5:** Write `notebooks/breakeven_colab.ipynb` in the style of `notebooks/train_cnn_colab.ipynb`: GPU check; idempotent clone + `git checkout <pinned commit>` + `pip install -e .`; smoke cell (`--k 0,12 --seeds 0 --pool-layouts 2 --pool-dies 1 --steps 5`); full cell (defaults, `--out /content/breakeven`); zip + download cell. Markdown states the expected time (about 20-30 min on a T4), that results are synthetic, and to return `breakeven.zip`.
- [ ] **Step 6:** Commit `feat(breakeven): Colab fine-tuning script and notebook`.

### Task 4: Local break-even evaluation

**Files:**
- Create: `experiments/breakeven.py`
- Test: `tests/test_breakeven.py`

**Interfaces:**
- Consumes: the Task 3 manifest; Task 2 `pick_threshold_labelled`, `DEFAULT_THRESHOLD`; `results/v2/exp3/geometry_f1.csv` (verifier F1 per family); `make_cnn_train_set` (to regenerate the pool for the threshold subset) and `make_sample` (test patches).
- Produces: `run(cfg: ExpCfg, weights_dir: Path) -> pd.DataFrame` writing `results/v2/breakeven/{breakeven.csv, summary.csv, breakeven.png, config.json}`. `breakeven.csv` columns: `family, k, seed, threshold_rule ('labelled'|'fixed_0.39'), threshold, f1, ci_lo, ci_hi, n, missing (bool)`. `summary.csv`: `family, threshold_rule, verifier_f1, breakeven_k, clearly_better_k` (k or "none").

- [ ] **Step 1: Write failing tests** with the Task 3 smoke output as fixture: the k = 0 row equals the M6 CNN's F1 on the same test patches at threshold 0.39; a manifest row with a deleted weights file yields `missing=True` and no crash; `summary.csv` break-even for a synthetic table `{k: f1}` = `{0: .70, 12: .80, 24: .90}` vs verifier 0.88 gives 24 (test the pure `breakeven_k(table, verifier_f1)` helper, defined in this file).
- [ ] **Step 2:** Run - expect FAIL.
- [ ] **Step 3:** Implement. Test patches regenerated exactly as Exp 3 (dose 50, test split, `cfg.layouts` x `cfg.dies` x KINDS); CPU predictions; plot F1 vs k (log-x, 0 drawn at the left edge) with seed mean +- range per rule and the verifier as a horizontal CI band; title says synthetic data.
- [ ] **Step 4:** Run - expect PASS.
- [ ] **Step 5:** Commit `feat(breakeven): local evaluation of fine-tuned CNNs against the verifier`.

### Task 5: Day-0 review-queue scenario

**Files:**
- Create: `experiments/day0_scenario.py`
- Test: `tests/test_day0_scenario.py`

**Interfaces:**
- Consumes: `results/v2/exp3/samples.csv` (via, dose 50; columns `kind, verdict, d2d_single, d2d_double, cnn`).
- Produces: `project(rates: dict[str, dict[str, float]], prevalence: float, n_sites: int = 1000) -> dict[str, float]` (pure arithmetic) and `run(cfg: ExpCfg) -> pd.DataFrame` writing `results/v2/day0_scenario/{scenario.csv, config.json}`. Per (method, prevalence): `killers_total, killers_caught, killers_missed, harmless_sent_as_killer, near_miss_flagged, sites_to_engineer` with 95% CIs from a bootstrap that resamples samples within each kind.

- [ ] **Step 1: Write failing tests** for `project`: with killer rate 1.0 for every killer kind and 0 for every non-killer kind at prevalence 0.05, `killers_caught == 50`, `harmless_sent_as_killer == 0`; with nuisance flag rate 0.1 only, `harmless_sent_as_killer == 0.1 * 950 * 0.9 == 85.5`; a method column missing from the frame is skipped by `run` (call it on a frame without `cnn`).
- [ ] **Step 2:** Run - expect FAIL.
- [ ] **Step 3:** Implement. Flag = KILLER verdict (verifier) or the boolean column (baselines); `near_miss_flagged` = verifier NEAR_MISS on notch/filament sites (0 for baselines, which have no such class); `sites_to_engineer = killers_caught + harmless_sent_as_killer + near_miss_flagged`. Mix from Global Constraints.
- [ ] **Step 4:** Run - expect PASS.
- [ ] **Step 5:** Commit `feat(day0): review-queue projection from measured via rates`.

### Task 6: Generated write-up (README section + docs/USE_CASE.md)

**Files:**
- Modify: `experiments/report.py` (add `render_day0(root) -> str`, markers `DAY0_START/END`)
- Create: `docs/USE_CASE.md` (prose + a generated block between the markers)
- Modify: `README.md` (section "Why use this: the day-0 case" with the same markers)
- Modify: `scripts/reproduce.py` (run `day0_scenario`; run `breakeven` when `--breakeven-weights` is given; then update both files)
- Test: `tests/test_m7_support.py` (or a new `tests/test_report_day0.py`)

**Interfaces:**
- Consumes: Task 4 `summary.csv`/`breakeven.csv`, Task 5 `scenario.csv`, `results/v2/exp3/geometry_f1.csv`, `results/v2/exp6/latency.csv`.
- Produces: `render_day0(root)` returning markdown with: the advantages/disadvantages table (verifier vs Manhattan-trained CNN vs die-to-die: day-0 F1 on via and curvy, in-distribution F1, labels needed, calibration ms, ms/patch, explains each call), the scenario table at 5% prevalence (1% and 20% in a second table), and the break-even line ("the CNN needs k = ... labelled via patches to match", or "not yet run" when `summary.csv` is absent).

- [ ] **Step 1: Write failing tests:** `render_day0` on a tmp root with only the scenario CSV says "not yet run" for break-even and still renders the scenario; calling `update_readme(..., (DAY0_START, DAY0_END))` twice leaves exactly one block.
- [ ] **Step 2:** Run - expect FAIL.
- [ ] **Step 3:** Implement; write the `docs/USE_CASE.md` prose (who: process / yield engineer on a new-layer ramp; when: before labelled data exists, and later as a CAD-grounded cross-check; what it replaces: die-to-die + manual review; honest losses stated), with numbers only inside the generated block.
- [ ] **Step 4:** Run the tests, then `python -m experiments.day0_scenario` and the report update; check that both files render the block.
- [ ] **Step 5:** `pytest -q` green, PROGRESS.md entry, commit `feat(day0): generated use-case section and USE_CASE.md`.

### Task 7: Hand-off for the Colab run, then close the loop

- [ ] **Step 1:** Push; give the user the notebook link (`notebooks/breakeven_colab.ipynb`, pinned to the pushed commit) and what to return (`breakeven.zip`).
- [ ] **Step 2 (after the zip arrives):** unzip to `results/v2/breakeven_weights/` (gitignored; weights are large), run `python -m experiments.breakeven --weights results/v2/breakeven_weights`, regenerate README + USE_CASE.md, `pytest -q`, PROGRESS.md entry with the break-even numbers and any missed expectation, commit, push.
