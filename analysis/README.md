# `analysis/` — the scripts behind every number in the paper

These are the measurement scripts for **"Voxels or Millimetres? A Controlled
Comparison of Lesion-Size Filtering for Native-Space Stroke Segmentation"**
(MICCAI 2026 SWITCH+ workshop) and for the ISLES'26 challenge submission.

They all operate on **predictions that already exist** — nnU-Net writes a
`validation/` folder per fold when trained with `--npz`, containing both the
binary mask and the softmax probability volume for every case the model of that
fold never saw. Pooling folds 0–4 gives out-of-fold predictions for all 1,453
subjects, which is the honest surface on which to fit and report a decision
rule. Nothing here retrains anything.

Everything is scored with the **organisers' own** `eval_utils.py`, which is not
redistributed — see [`official/README.md`](official/README.md) for the one file
you have to fetch before any of this runs.

---

## Setup

```bash
export ISLES_ROOT=/path/to/this/checkout
export ISLES_NNUNET_RESULTS=/path/to/nnUNet_results/Dataset001_ISLES26
export ISLES_GT=/path/to/nnUNet_preprocessed/Dataset001_ISLES26/gt_segmentations

pip install -r ../requirements.txt
# then drop the organisers' eval_utils.py into analysis/official/
```

Install from the pinned file, not by hand: `panoptica`, `scikit-learn`,
`connected-components-3d` and the numpy major are marked LOAD-BEARING there
because they decide what Detection F1 and PR-AUC mean. A different resolution
changes these numbers silently instead of failing.

Every script here takes a **run** — the directory name nnU-Net writes under
`$ISLES_NNUNET_RESULTS`, `<trainer>__<plans>__<configuration>`. The two shipped
members are in `paths.py` as `MEMBER_500EP` and `MEMBER_TOPK10`:

```
nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres
nnUNetTrainerDiceTopK10_250epochs__nnUNetResEncUNetMPlans__3d_fullres
```

`official_score.py --runs "A+B"` averages two members' probability maps and scores
the result as one model (equally weighted — the 0.65 / 0.35 weighting is
`weighted_ensemble.py`). Scripts that default to a single run
(`sweep_mm3_all5.py --run`, `lesion_pr.py` via `ISLES_RUN`) default to the
**250-epoch baseline** those two analyses were measured on, which
`training/train_shipped.sh` does not train — pass a run you actually have.

`paths.py` is the single source of truth for those roots; every scorer here
imports it rather than hardcoding a machine. The one script that does not is
`site_transfer.py`, which reads nothing but the per-case cache and therefore
needs no roots at all (`ISLES_MM3_SWEEP` points it at that file). Defaults
assume the layout in `training/README.md`, so the environment variables are only
needed if yours differs.

The shipped configuration referred to throughout: **500-epoch nnU-Net ResEnc-M,
5 folds, ensembled with a Dice+TopK10 member at weights 0.65 / 0.35, τ = 0.5, and
a 30 mm³ instance decision layer.** Pooled out-of-fold over n = 1,453 subjects it
scores Dice 0.6552, Detection F1 0.6063, PR-AUC 0.7617.

---

## The scripts

| Script | What it measures | Paper claim it backs | How to run |
|---|---|---|---|
| **`official_score.py`** | The authoritative scorer. Scores any run (or `runA+runB` probability-averaged ensemble) over any folds, at a grid of binarisation thresholds × minimum component sizes × component max-probability cuts, on all five challenge metrics, and simulates the challenge's rank-then-aggregate ranking. | **Every reported number.** Each of the other scripts either calls it or reproduces its metric definitions. | `python3 analysis/official_score.py --runs <run> --folds 0,1,2,3,4 --thresholds 0.5 --min-sizes 0,20,30,40 --mm3 --tag oof` |
| **`sweep_mm3_all5.py`** | Scores every out-of-fold subject under *both* parameterisations of the size rule — voxel counts `{0, 20}` and mm³ `{10,20,30,40,60,80,120}` — and writes the per-case matrix `sweep_mm3_all5.json`. | The mm³-vs-voxels comparison at **matched magnitude: +0.0036 Detection F1** (20 mm³ vs 20 voxels), 95% CI [0.0016, 0.0058] by case-level bootstrap over the `f1_mm3_20` and `f1_vox_20` columns. Also the "filtering at all" row. | `python3 analysis/sweep_mm3_all5.py --run <run>` |
| **`site_transfer.py`** | The decisive generalisation test. Splits the 55 sites into disjoint halves, picks the operating point on the fit half **by the challenge's own rank objective**, scores that choice on the held-out half, over 400 random partitions. | **mm³ beats voxels on held-out sites in 99.5% of the 400 partitions**, paired difference +0.0075 ± 0.0033 Detection F1. The falsification condition was fixed in advance; the claim survived it. | `python3 analysis/sweep_mm3_all5.py --run <run>` once, then `python3 analysis/site_transfer.py`. The second step reads only the cache — no volumes, no weights, not even `eval_utils.py` — so it is cheap to re-run once the first has been paid for |
| **`nested_cv.py`** | Nested cross-validation of the decision layer: for each fold, fit (τ, v_min) on the other four by the rank objective, apply to the held-out fold, pool. No fold helps choose the threshold it is then scored under. Also stratifies Detection F1 by reference lesion size. | **Selection optimism = +0.0021 Detection F1.** The paper fits the operating point on the same pool it reports on; this quantifies exactly how much that is worth, in response to a reviewer. | `python3 analysis/nested_cv.py` (first run builds and caches `nested_cv_percase.json`) |
| **`miss_anatomy.py`** | For every ground-truth lesion: was it detected (IoU ≥ 0.25, the challenge rule), how big is it, and what is the model's **peak probability inside it**. | The **small-infarct analysis** — whether the 30 mm³ rule destroys genuine small lesions, and whether misses are blindness (peak probability ≈ 0) or lost proposals (appreciable probability, lost to component matching). | `python3 analysis/miss_anatomy.py` |
| **`weighted_ensemble.py`** | Sweeps the weight on the 500-epoch member against the Dice+TopK10 member (0.35 / 0.50 / 0.65 / 0.80) at the shipped operating point. | The **0.65 / 0.35 choice**. The 50/50 average that shipped first was never tuned; the two members are not equally good (500ep leads Dice and Detection F1, TopK10 carries PR-AUC). | `python3 analysis/weighted_ensemble.py` |
| **`best_ensemble.py`** | Two-stage search over the 3-member weight simplex (adding a Dice+CE-×2 member): coarse ranking on a 350-case random subset, then the winner *and* the incumbent re-scored on all 1,453. | Confirms **0.65 / 0.35 survives** against tuned three-member alternatives. Subsetting only ever decides which candidates to confirm; every reported figure comes from the full cohort. | `python3 analysis/best_ensemble.py` |
| **`confidence_filter.py`** | Tests a second keep-criterion alongside size: `volume ≥ v_min AND stat(prob inside component) ≥ p`, for `stat ∈ {mean, max}`. `p = 0` reproduces the shipped rule exactly and is the control column. | A **measured negative result**: confidence filtering does not beat size alone by more than the noise floor. Reported as a negative rather than dropped. | `python3 analysis/confidence_filter.py` |
| **`lesion_pr.py`** | Pooled lesion-wise TP / FP / precision / recall before and after the decision layer, plus how many *correctly matched* lesions the filter destroys and how many false-positive components it removes. | The mechanism behind the filtering gain — it separates "helps by removing false positives" from "hurts by deleting true small lesions", which Detection F1 alone cannot do. | `python3 analysis/lesion_pr.py` |
| **`empty_map_risk.py`** | How often we predict no lesion at all, how often that is correct, and the net PR-AUC of emitting a *constant* soft map on those cases. | Why **`EMIT_CONSTANT_MAP_WHEN_EMPTY` stays `False`** in the container. The organisers' scorer returns PR-AUC 1.0 for an empty ground truth given a perfectly constant map and ~0 otherwise, so the trick is worth up to a full point per empty-GT case — and destroys real ranking information whenever we are wrong. | `python3 analysis/empty_map_risk.py` |
| `paths.py` | Not a scorer — the single source of truth for filesystem roots. | — | imported, not run |
| `sweep_mm3_all5.json` | Not a script and **not in this repository** — the per-case cache, 1,453 rows, that `sweep_mm3_all5.py` writes and `site_transfer.py` reads. It is per-subject derived data, so it is not redistributed; rebuild it. | — | built by `sweep_mm3_all5.py`, then checked against the column means below |

---

## `official_score.py` is the one to read first

It is the authoritative scorer, and two things in it are load-bearing.

**It calls the organisers' code, not a reimplementation of the metric spec.**
Reading `eval_utils.py` corrected three things this project had wrong: Detection
F1 is Panoptica Recognition Quality at IoU ≥ 0.25 (not "any voxel overlap",
which inflates it); PR-AUC is trapezoidal (not `average_precision_score`); and
for an empty ground truth the released code returns 1.0 for a perfectly constant
soft map and 0.0 for anything that varies, which the design document does not
say. That last one is the entire subject of `empty_map_risk.py`.

**`ProbReadError` refuses to silently fall back to the binary mask.** This is
deliberate and it should stay. An earlier version of `_load_prob` caught every
exception and returned `None`, and the caller then scored the stored `.nii.gz`
mask instead of `prob ≥ τ`. Under heavy concurrent I/O a handful of cases per run
took that path, and **two scorings of identical predictions with identical code
disagreed by up to 0.004 Dice and 0.007 Detection F1 — larger than the seed-noise
floor those numbers were being compared against.** A read that fails is now
retried and then raised, and `main()` additionally aborts any run where only
*some* cases have a soft map, because a mixed run silently blends two decision
paths. A run is either fully soft-map-scored or it fails. If you change one
thing in this directory, do not change that.

---

## Reading the numbers

Nothing here is interesting unless it clears the **two-seed reference floors**,
measured by retraining the same configuration with a different seed:

| Metric | Floor |
|---|---|
| Dice | 0.0043 |
| Detection F1 | 0.0057 |
| PR-AUC | 0.0034 |

A change smaller than its floor is not a result. `confidence_filter.py` prints
these floors next to its own deltas for exactly that reason, and
`weighted_ensemble.py` reports its changes against them.

The paper's headline for the filtering contrast — no filter vs the shipped
30 mm³ rule — is Detection F1 **0.5333 → 0.5895** with mean lesion-count error
**2.08 → 1.85**.

### Checksum for `sweep_mm3_all5.json`

The `sweep_mm3_all5.json` behind the paper was built from the **250-epoch baseline**
run over all 1,453 pooled out-of-fold subjects. Its column means are recorded
here so a rebuild with `sweep_mm3_all5.py` can be checked rather than trusted:

| setting | Dice | Detection F1 | \|Δcount\| |
|---|---|---|---|
| no filter (`vox_0`) | 0.6395 | 0.5283 | 2.091 |
| 20 voxels | 0.6385 | 0.5772 | 1.849 |
| 10 mm³ | 0.6394 | 0.5678 | 1.898 |
| 20 mm³ | 0.6392 | 0.5808 | 1.863 |
| **30 mm³** | 0.6391 | 0.5849 | 1.862 |
| 40 mm³ | 0.6383 | 0.5873 | 1.873 |
| 60 mm³ | 0.6361 | 0.5864 | 1.898 |
| 80 mm³ | 0.6317 | 0.5821 | 1.931 |
| 120 mm³ | 0.6258 | 0.5766 | 2.010 |

`20 mm³ − 20 voxels = +0.0036` Detection F1 is the matched-magnitude units
effect. Comparing 40 mm³ against 20 voxels instead gives +0.0101, but roughly
two thirds of that is magnitude rather than units — the paper reports +0.0036.

---

## What is deliberately not here

This directory is a curated subset of a much larger sweep. Left out: the
container timing benchmarks (`bench_cpu.py`, `bench_gpu.py`) and the checkpoint
rename verification (`check_rename.py`), which are packaging work rather than
results; the five earlier scorers (`evaluate.py`, `challenge_metrics.py`,
`ensemble_eval.py`, `oof_analysis.py`, `postproc_sweep.py`) that reimplemented
the metric spec from the challenge design document and were superseded once
`official_score.py` started calling the organisers' code; and the learned
false-positive suppression stage (`fp_suppression.py`, `fp_apply.py`), which
reached lesion-wise macro-F1 0.6691 against the blunt 30 mm³ rule's 0.6665 — a
+0.0026 gain, less than half the 0.0057 Detection F1 noise floor, for a second
model, a metadata dependency and a feature pipeline. It did not ship, and it is
not a result.

A few docstrings carry section markers — `SS8e`, `SS8f`, `SS8g`, `SS8k`. They
are the authors' own cross-references into the project's internal experiment
log, which is not part of this repository (`SS` is the plain-ASCII spelling of
`§`). They are left verbatim rather than scrubbed, because they record what the
script was written in response to. Nothing in them is lost, and they resolve
here as follows:

* `SS8f` (in `miss_anatomy.py`) and `SS8g` (in `confidence_filter.py`) are used
  in sentences that already state the finding they stand for — recall 0.408 with
  no operating point recovering it, and residual error dominated by detection
  rather than delineation. The script carrying each one is the script that
  measures it.
* `SS8e` and `SS8k` (in `weighted_ensemble.py` and `confidence_filter.py`) are
  not findings but *comparability anchors*: they name the pooled out-of-fold
  scoring of the shipped ensemble, at equal member weight (`SS8e`) and at the
  container's 0.65 / 0.35 (`SS8k`). Both rows are in this repository — equal
  weight is the Results table in the top-level README (Dice 0.6552, Detection F1
  0.6063, PR-AUC 0.7617), and 0.65 / 0.35 is in
  [`../submission/README.md`](../submission/README.md) (Dice 0.6570, Detection
  F1 0.6097), regenerated by `weighted_ensemble.py` itself. A docstring saying
  its numbers are "comparable to SS8e" is saying they were scored at the shipped
  operating point over all 1,453 subjects, which is the only property the
  comparison needs.
