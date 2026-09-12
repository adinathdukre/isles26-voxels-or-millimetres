# Voxels or Millimetres? — ISLES'26 stroke lesion segmentation

Code for our **ISLES'26** challenge entry and for the MICCAI 2026 SWITCH+ workshop paper
**"Voxels or Millimetres? A Controlled Comparison of Lesion-Size Filtering for
Native-Space Stroke Segmentation"**.

**This repository is code only.** It holds the submitted algorithm container, the
training code and the measurement scripts — nothing else. No manuscript, no imaging
data, no checkpoints, no per-subject result caches: none of those are ours to
redistribute, and a per-case table of Dice values is still a table about identifiable
subjects. What the experiments found is written down below rather than shipped as a
report, so this README is the record; the scripts that produced every number in it are
in [`analysis/`](analysis/).

The short version: **the thing that helped was not the network.** It was writing a
familiar post-processing threshold in physical units instead of voxels, and fitting it
to the challenge's own ranking objective on pooled out-of-fold predictions rather than
to Dice on one fold. Everything we tried at the network and loss level — a larger
planner, a transformer backbone, four instance-aware or boundary-aware losses, six
augmentation regimes, metadata conditioning — either failed or landed inside the noise
floor of retraining the same recipe twice.

---

## The finding

ISLES'26 scores **five quantities per case** — Dice, absolute volume difference, PR-AUC
on the soft probability map, lesion-wise Detection F1, and absolute lesion-count
difference — ranks every case on every metric, and averages the ranks. Only one of the
five is Dice.

That changes where the headroom is. Our nnU-Net ResEnc-M baseline, out of fold over all
1,453 training subjects:

```
Dice           0.6452      challenge-reported inter-rater agreement  0.76 ± 0.14
Detection F1   0.5333
|ΔN|           2.08 lesions per case
```

Dice sits close to the annotation ceiling. Detection F1 is nowhere near it. **The room
that remains is in finding lesions, not in outlining them** — which points at the
instance decision layer rather than at the network.

The standard remedy is to delete small predicted components, and its threshold can be
written in voxels or in cubic millimetres. On a registered, isotropic cohort those are
the same rule. ISLES'26 is **native space**: across the 1,453 volumes, voxel volume
spans **0.031–5.27 mm³**, a 168× spread, with only 71.6% of cases inside [0.9, 1.1] mm³
and 231 distinct volume shapes. A 20-voxel rule — the form every prior ATLAS
post-processing rule takes, because ATLAS v2.0 was MNI-registered at 1 mm³ — means
0.63 mm³ on one scanner and 105 mm³ on another.

So the decision layer is: binarise at τ, label 26-connected components, and keep those
of volume ≥ `v_min` mm³, which on a given scan means ≥ `n_min = ⌈v_min/ν⌉` voxels with
the voxel volume ν read from that scan's own header. Fitted on pooled out-of-fold
predictions by the challenge's rank-then-aggregate objective: **τ = 0.5, v_min = 30 mm³**.
The soft map ships unfiltered, because PR-AUC is computed from it alone and filtering
could only discard ranking information.

Both effects are measured, and they are not the same size:

* **Filtering at all** is the large effect: Detection F1 **0.5333 → 0.5895**,
  lesion-count error **2.08 → 1.85**, for 0.001 of Dice and no training. Lesion
  precision rises 0.448 → 0.614 for a recall cost of 0.014.
* **The choice of units** is a second, far smaller effect at matched magnitude:
  **+0.0036 Detection F1**, bootstrap 95% CI [0.0016, 0.0058] — roughly a sixteenth of
  the first. It holds on held-out sites in **99.5% of 400 site splits** (a falsification
  condition fixed before the test was run) and follows a dose-response: it is +0.0209
  where voxels are under 0.5 mm³, +0.0121 above 2 mm³, and **−0.00004** in the 1,041
  cases near 1 mm³ where the two rules almost coincide by construction.

Two numbers keep the rest of the repository honest. The **two-seed reference floor** —
retraining the identical recipe with a different seed — is Dice 0.0043, Detection F1
0.0057, PR-AUC 0.0034; nothing smaller than that is a result. And refitting the
operating point under nested cross-validation, so no fold helps choose the threshold it
is then scored under, puts the **selection optimism at +0.0021** Detection F1.

---

## What was submitted

ISLES'26 asks for an open repository for the submitted algorithm under a permissive
licence. This is it, and this is the algorithm, exactly:

| | value | where it lives in the code |
|---|---|---|
| architecture | nnU-Net v2, **ResEnc-M**, `3d_fullres`, T1w-only | `plans.json` inside each member — that file travels with the weights, not with this repository; the planner that writes it and the configuration it produced (128³ patches, batch 2, 1 mm isotropic, `batch_dice=False`, z-score with `use_mask_for_norm`) are in [`training/README.md`](training/README.md#2-dataset-conversion) |
| member A | `nnUNetTrainer_500epochs`, **folds 0–4** (stock nnU-Net trainer) | `NNUNET_MODEL_SUBDIRS[0]` in [`submission/inference.py`](submission/inference.py) |
| member B | `nnUNetTrainerDiceTopK10_250epochs`, **folds 0–4** | [`training/trainers/isles_wave4.py`](training/trainers/isles_wave4.py) |
| ensemble | weighted average of the two members' **softmax** maps, **0.65 / 0.35** (A / B) | `ENSEMBLE_WEIGHTS` |
| checkpoint | **`checkpoint_final.pth`** — deliberately not `_best` | `CHECKPOINT_NAME` |
| binarisation | **τ = 0.5** | `BINARY_THRESHOLD` |
| instance decision layer | drop 26-connected components **< 30 mm³**, physical units, voxel volume read per case from the scan header | `MIN_LESION_MM3` |
| soft map output | shipped **unfiltered** | — |
| test-time augmentation | mirroring, GPU path only | `init_model()` |

Metadata (`CENTER`, `CHRONICITY`, `DAYS_POST_STROKE`) is read and logged for
traceability and never reaches the network. The container is a FastAPI server that
Grand Challenge boots once per job and calls once per case; [`submission/`](submission/)
holds it, and [`submission/inference.py`](submission/inference.py) is the whole method
in one readable file, with the reason for each constant in the comment beside it.

---

## Results

Pooled out-of-fold, n = 1,453 subjects, scored with the organisers' own evaluation code.
Every prediction comes from the one fold model that never saw that subject.

| configuration | Dice | Detection F1 | \|ΔN\| | \|ΔV\| (mL) | PR-AUC |
|---|---|---|---|---|---|
| 250-epoch baseline, no filter | 0.6452 | 0.5333 | 2.08 | — | 0.7339 |
| 250-epoch baseline + 30 mm³ decision layer | 0.6441 | **0.5895** | **1.85** | — | 0.7339 |
| 500-epoch member alone + decision layer | **0.6565** | **0.6097** | 1.81 | 5.36 | 0.7528 |
| **shipped: 500 ep + Dice+TopK10 ensemble** | 0.6552 | 0.6063 | 1.82 | **5.25** | **0.7617** |

Three notes on reading that table, because the repository would otherwise look
inconsistent with itself:

1. The ensemble buys a **real PR-AUC gain** (+0.0088, 2.6× its own floor) for Dice and
   Detection F1 losses that are *inside* the floor (0.3× and 0.6×). It is not trading
   real accuracy for real ranking.
2. The shipped row is quoted at **equal member weights**, which is how the paper quotes
   it and the only thing `analysis/official_score.py` can do — it averages members
   equally. The container ships **0.65 / 0.35**, which
   [`analysis/weighted_ensemble.py`](analysis/weighted_ensemble.py) measured as better
   on four of five metrics — Dice 0.6570, Detection F1 0.6097, |ΔN| 1.80, |ΔV| 5.30 mL —
   at no inference cost. The PR-AUC that script prints is a *naive* value computed
   without the organisers' empty-ground-truth convention, so it is not comparable with
   0.7617 and the two tables were never merged into a single row. Every individual delta
   from the weight change is below its own floor; what makes it credible is that four
   move together.
3. The per-case cache behind the units comparison scores the same 250-epoch baseline
   about 0.005 lower in absolute terms (0.5283 → 0.5849 rather than 0.5333 → 0.5895)
   because it comes from a different scoring job. The units contrast is a **paired
   within-cache difference** and does not depend on that offset;
   [`analysis/README.md`](analysis/README.md) records the cache's column means as a
   checksum.

---

## What did not work

This project's negative results cost GPU hours that another team now does not have to
spend. All deltas below are against the two-seed reference floor (Detection F1 0.0057).
Every arm was trained on the identical split; unless stated otherwise, fold 0 at 250
epochs on ResEnc-M.

* **ResEnc-L** — nnU-Net's own recommendation when given a bigger GPU, 4.4× the voxels
  per optimiser step. A tie at 250 epochs, and at a matched 500-epoch budget it is worse
  on *both* folds tested and on every metric that matters (ΔDetection F1 −0.0149 and
  −0.0229). At ~41 h/fold against ~15, the worst compute trade in the sweep.
* **Primus (transformer backbone)** — Detection F1 0.340 against the baseline's 0.603.
  A collapse, not a near-miss; ranks 22 and 23 of 23 arms. On the CNNs' 250-epoch
  budget, which likely understates a transformer.
  [`training/trainers/isles_primus.py`](training/trainers/isles_primus.py) is the arm.
* **Blob loss** — 0.014 *below* baseline. On synthetic patches it genuinely supplies
  ~3 orders of magnitude more gradient than Dice+CE on a missed small lesion. The
  mechanism is real; it did not translate on top of a well-configured nnU-Net.
* **Tversky (0.3/0.7) and focal Tversky** — both cost volume error (7.2 and 7.1 mL
  against 5.9) for nothing on the aggregate.
* **Batch Dice** — making the Dice term a batch-global statistic, which nnU-Net's own
  guidance recommends for small and sparse structures, since a patch holding a handful
  of lesion voxels otherwise yields a near-degenerate per-sample Dice. It did not clear
  the seed-noise floor. Blob loss, both Tverskys and batch Dice all attack *size*
  imbalance, and none of the four beat the floor — which is what sent us looking for a
  loss that attacks fragmentation instead.
* **Augmentation: the default is the optimum, and every deviation costs.** DA5
  (aggressive) −0.0059, DA5 order-0 −0.0196, no augmentation at all −0.0216, no
  mirroring −0.0357, mirroring restricted to the in-plane axes −0.0050 (inside the
  floor). More, less and restricted augmentation are all worse. One caveat we found by
  reading nnU-Net's source rather than our results: the NoDA and NoMirroring trainers
  also disable *test-time* mirroring, so their deficits mix two changes; the clean
  comparison is the in-plane-only arm, where restricting mirroring costs nothing
  detectable. That is the opposite of the intuition that lateralised pathology should
  not be mirrored.
* **clDice** — the one loss chosen from a measured fact rather than from fashion, and it
  delivered on its own prediction (Detection F1 +0.0144, 2.5× the floor; better lesion
  count) — and still finished *behind* baseline on the challenge's aggregate rank,
  because it paid on the other three metrics. A caution against reporting the metric
  your method targets.
* **FiLM metadata conditioning** — conditioning a ResEnc-M on the scan's own metadata,
  as a fully powered null: +0.0010 ± 0.0067 across five folds, 2 of 5 favouring FiLM,
  p = 0.76. On fold 0 alone it had the best Detection F1 *and* the best lesion-count
  error of any arm we had, and would have been our headline — which is the whole reason
  it was run out to five folds. The trap worth repeating even though the arm is dead:
  nnU-Net's predictor calls the network with the image tensor alone, so a metadata model
  dropped in naively **trains with conditioning and silently validates without it**, and
  measures a null for a reason that has nothing to do with metadata.
* **Multi-size labeling** — a negative replication of an ATLAS-specific method; merged
  MSL lost to the plain single-class baseline on the ranked objective.
  [`training/build_msl_dataset.py`](training/build_msl_dataset.py) builds it.
* **The naive ensemble** — averaging the 500-epoch and 250-epoch models loses out of
  fold on four of five metrics, because it averages a strong model with a weak one.
  "Ensembling helps" is false as stated; ensembling *comparably strong* members is what
  works, which is why the shipped pair is 500 ep + TopK10.
* **A third ensemble member** — adding Dice+CE-×2 gives 0.6538 / 0.6048, behind the
  two-member pair, and a two-stage search over the three-member weight simplex confirms
  0.65 / 0.35 survives against tuned alternatives, at two thirds of the inference cost.
  [`analysis/best_ensemble.py`](analysis/best_ensemble.py) is that search.
* **Confidence filtering** — gating components on mean or max probability inside them, in
  addition to size, does not beat size alone by more than the noise floor.
  [`analysis/confidence_filter.py`](analysis/confidence_filter.py) ships so the negative
  is reproducible.
* **A learned decision layer** — a gradient-boosted classifier over component shape,
  contrast, position and metadata, with site-grouped CV, reaches AUC 0.819 and lesion-wise
  macro-F1 0.6691 against the one-line 30 mm³ rule's 0.6665. A +0.0026 gain, under half
  the noise floor, in exchange for a second model, a metadata dependency and a feature
  pipeline. Its dominant feature is component size. The blunt rule is not leaving much
  on the table.

### Three conclusions that reversed, and two bugs in our own tooling

Kept here because they are the reason the rest of the numbers are trusted at all.

Pooling all five folds reversed **three** conclusions that single-fold scoring had
supported: the binarisation threshold (τ = 0.8 looked better on one fold and is worse
pooled, which is why the shipped τ is 0.5), one loss comparison, and FiLM — the arm that
would have been the headline on fold 0 and is a null across five.

Two bugs were ours, not the data's. **clDice returned exactly 0.0 for every input**
because `-a.min(b)` parses as `-(a.min(b))`, not as `(-a).min(b)` — a silent no-op loss
term that looked like a trained model. And an early scorer **silently fell back from the
soft map to the stored binary mask** when a read failed under heavy concurrent I/O, so
two scorings of *identical* predictions disagreed by up to 0.004 Dice and 0.007
Detection F1 — larger than the seed-noise floor those numbers were being judged against.
`official_score.py` now raises `ProbReadError` instead of falling back, and aborts any
run where only *some* cases have a soft map. If you change one thing in `analysis/`, do
not change that.

---

## Caveats

**The preliminary leaderboard is not a target.** Our container passed the ISLES'26
Preliminary Docker Evaluation at Dice 0.887 ± 0.071, 34th place. That phase is **two
cases** — the reported spreads are reproduced exactly by n = 2 and no other small n — and
the sanity case shipped with the starter kit is **one of the 1,453 public training
subjects**, held out in our fold 2 and trained on by the other four:

```
fold-2 model (never saw the case)     Dice 0.9210
our 5-fold ensemble (4/5 saw it)      Dice 0.9475
honest out-of-fold Dice, n = 1,453    Dice 0.6565
```

A model trained on all the data with no held-out fold scores near-perfectly on such a
case, which is the most economical explanation for the leaders at 0.966 ± 0.007 and the
tight clustering just below. That leaderboard largely ranks **memorisation of the public
training set**. Expect **~0.65 Dice on genuinely held-out data**, for us and for everyone
else, and do not tune against it.

Other limits, stated rather than glossed:

* **The ablation is single-fold.** Only the decision layer and the FiLM null are
  validated across all five folds. The 23-arm re-score that the ranked claims rest on is
  fold 0, one job, one code path — and it widens the camera-ready's claim: seven arms
  beat the baseline's aggregate rank by more than the 0.080-rank-unit seed margin, not
  the two the manuscript states.
* **Cross-site behaviour is probed by refitting the operating point across site splits,
  not by leave-site-out training.** Conclusions about augmentation speak only to
  in-distribution performance.
* **The filter does delete real lesions.** 28.1% of ground-truth lesions are below the
  30 mm³ threshold. [`analysis/miss_anatomy.py`](analysis/miss_anatomy.py) separates
  misses that are model blindness (peak probability ≈ 0 inside the lesion) from proposals
  lost to component matching, and finds most small misses are the former. The filter's
  gain is concentrated in *large* lesions — spurious fragments beside a correctly
  segmented infarct, charged twice under one-to-one matching — which mean Dice cannot
  see.
* **The container's two-member GPU path has never been exercised inside a container.**
  The build machine has no NVIDIA container runtime, so every local test fell back to
  CPU, where `init_model()` deliberately drops to one member and disables mirroring TTA
  to stay inside the runtime budget. An earlier image did pass the organisers'
  preliminary evaluation.
* **`analysis/sweep_mm3_all5.py` is a reconstruction.** Its original was lost; it was
  rebuilt as a thin wrapper over `official_score.py`'s scoring path so the metrics come
  from the authoritative scorer, and it has not been run end to end. The per-case cache
  it writes is not in this repository either — it is per-subject derived data — so the
  units comparison and `site_transfer.py` both depend on a rebuild that has never been
  validated end to end. Check that rebuild against the column-means checksum table in
  [`analysis/README.md`](analysis/README.md) before trusting it; that table is exactly
  why it is there.

---

## Repository layout

Four things, and nothing else.

| directory | what is in it |
|---|---|
| [`submission/`](submission/) | The Grand Challenge algorithm container that was submitted: a FastAPI server whose [`inference.py`](submission/inference.py) holds the whole shipped method and the failure guard, plus build / local-test / export scripts and the test fixtures. |
| [`training/`](training/) | Everything between the ISLES'26 download and the checkpoints: dataset conversion, [`env.sh`](training/env.sh), the custom trainer classes for every ablation arm, [`train_shipped.sh`](training/train_shipped.sh) for the two shipped members, and the queue-directory GPU scheduler the sweep ran under. |
| [`analysis/`](analysis/) | The measurement scripts behind every number above. They operate on predictions that already exist and score them with the organisers' `eval_utils.py`; nothing here retrains anything. [`official_score.py`](analysis/official_score.py) is the authoritative scorer and the one to read first. |
| top level | [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), [`CITATION.cff`](CITATION.cff), [`requirements.txt`](requirements.txt), [`.gitignore`](.gitignore) and this README. |

Model weights (~7.6 GB), imaging data, `.npz` softmax maps, scorer output, logs and
build artefacts are **not** in this repository, and `.gitignore` is written to keep it
that way. Each directory's README says exactly what to supply and where.

---

## Reproducing the numbers

### 0. What a fresh clone alone can do

Nothing, honestly. Every number above is computed from model predictions over the
ISLES'26 training set, and neither the data, the weights, nor the per-case result caches
derived from them are redistributable here. So there is no "run this and see 0.6552"
shortcut, and pretending otherwise would be the one dishonest line in the repo.

What you can do immediately is *read*: this README is the experimental record,
[`analysis/README.md`](analysis/README.md) carries the column-means checksum table for
the per-case cache so a rebuild can be checked rather than trusted, and
[`submission/inference.py`](submission/inference.py) is the whole shipped method in one
file.

The cheapest thing to actually reproduce is the decision layer, not the network. With
checkpoints in hand — yours, or retrained from §3 — `analysis/official_score.py` emits
the Results table, one scoring job per configuration, and `analysis/sweep_mm3_all5.py` followed by
`analysis/site_transfer.py` reproduces the mm³-versus-voxels result and its
held-out-site falsification test without any further training. §3 says which command
produces which published number.

### 1. What you have to supply

| you need | where to get it | where it goes |
|---|---|---|
| ISLES'26 training data (ATLAS v3.0 Raw, 1,453 subjects) | the ISLES'26 challenge / NITRC ATLAS download portal | anywhere; point `ISLES_ATLAS3_RAW` at it |
| `eval_utils.py`, the organisers' scorer | the ISLES'26 challenge materials — **not redistributed here**, see [`analysis/official/README.md`](analysis/official/README.md) | `analysis/official/eval_utils.py` |
| trained checkpoints | train them (below), or your own | `nnUNet_results/…`, and `submission/model/` for the container — see [`submission/model/README.md`](submission/model/README.md) |
| the per-case cache the units comparison is computed from | rebuild it with `python3 analysis/sweep_mm3_all5.py --run <run>` once you have out-of-fold predictions; checksum table in [`analysis/README.md`](analysis/README.md) | `analysis/sweep_mm3_all5.json`, which stays gitignored |

Almost nothing under `analysis/` runs until `eval_utils.py` is in place. The three
exceptions are `miss_anatomy.py`, which does its own IoU matching; `lesion_pr.py`, which
calls Panoptica directly because it needs the raw TP/FP/FN counts `eval_utils` does not
expose; and `site_transfer.py`, which only reads the per-case cache. That dependency is
deliberate: we call the organisers' implementation rather than a reimplementation of the
metric spec, and reading it corrected three things we had wrong — Detection F1 is
Panoptica Recognition Quality at IoU ≥ 0.25 (not "any voxel overlap", which inflates it),
PR-AUC is trapezoidal, and an empty ground truth is *not* excluded from PR-AUC.

### 2. Environment

```bash
pip install -r requirements.txt
```

Install from the pinned file rather than by hand. The pins marked LOAD-BEARING in
[`requirements.txt`](requirements.txt) carry the reason next to each one: `panoptica`,
`scikit-learn`, `connected-components-3d` and the numpy major decide what Detection F1
and PR-AUC *mean*, so a different resolution changes the published numbers silently
rather than failing. `torch` is deliberately left unpinned — install the build matching
your CUDA. [`submission/requirements.txt`](submission/requirements.txt) is a different
environment on purpose; installing it will not reproduce these numbers, and installing
this one will not reproduce the container.

Three environment variables locate everything. [`analysis/paths.py`](analysis/paths.py)
is the single source of truth: every scorer under `analysis/` reads its roots from it
rather than hardcoding a machine. (`site_transfer.py` is the exception — it touches no
volume, weight or ground truth, only the per-case cache, so it needs none of these
roots.) Shell scripts use the same names with `${VAR:-default}` defaults.

```bash
export ISLES_ROOT=/path/to/this/checkout
export ISLES_NNUNET_RESULTS=$ISLES_ROOT/nnUNet_results/Dataset001_ISLES26        # default
export ISLES_GT=$ISLES_ROOT/nnUNet_preprocessed/Dataset001_ISLES26/gt_segmentations  # default
```

`source training/env.sh` sets nnU-Net's own three directories, the custom-trainer search
path, and exports the two variables above from them, so the training and analysis halves
cannot drift apart.

Every other variable is local to one module, has a `:-` style default, and is documented
where it is read. Two of them you will actually want:

| variable | read by | default | set it when |
|---|---|---|---|
| `ISLES_ATLAS3_RAW` | `training/convert_to_nnunet.py` | `$ISLES_ROOT/data/ATLAS3_Training_Raw` | your extracted download is anywhere else — this is the one variable conversion cannot guess |
| `ISLES_RUN` | `analysis/sweep_mm3_all5.py`, `analysis/lesion_pr.py` | `nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres` | always, unless you trained the 250-epoch baseline. Both scripts default to the *baseline* run those two analyses were measured on, and `train_shipped.sh` does not train it — see §3 |

The rest, for completeness: `ISLES_MM3_SWEEP` (where `site_transfer.py` looks for the
per-case cache), `ISLES_MODEL_DIR` (where the container's build and test scripts look
for checkpoints), `ISLES_SWEEP_DIR`, `ISLES_WORKERS`, `ISLES_DATASET_ID`,
`ISLES_TEST_GPUS` and `ISLES_TEST_NET`.

### 3. The pipeline

```bash
source training/env.sh

# convert ATLAS v3.0 Raw -> nnUNet_raw/Dataset001_ISLES26, then plan and preprocess
export ISLES_ATLAS3_RAW=/path/to/extracted/ATLAS3_Training_Raw
python3 training/convert_to_nnunet.py
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres --verify_dataset_integrity

# train the two shipped members, five folds each  (--npz is NOT optional)
training/train_shipped.sh

# score out of fold and fit the decision layer on the pooled 1,453 subjects
A=nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres
B=nnUNetTrainerDiceTopK10_250epochs__nnUNetResEncUNetMPlans__3d_fullres
python3 analysis/official_score.py --runs "$A+$B" --folds 0,1,2,3,4 \
        --thresholds 0.5 --min-sizes 0,20,30,40 --mm3 --tag oof

# build, test and export the container
cd submission && ./do_build.sh && ./do_test_run.sh && ./do_save.sh
```

A "run" is the **directory name nnU-Net writes under
`$ISLES_NNUNET_RESULTS`** — `<trainer>__<plans>__<configuration>`, which is why the two
above are that long. `ls "$ISLES_NNUNET_RESULTS"` lists the ones you have;
`analysis/paths.py` holds the same two strings as `MEMBER_500EP` and `MEMBER_TOPK10`.
`runA+runB` averages the two members' probability maps and scores the result as one
model; a comma-separated list scores several runs side by side instead.

**Which command produces which headline number.** The scorer prints one row per
`threshold|min-size|pmax` variant, so the command above emits every decision-layer row
of the shipped ensemble in one job; the other configurations are the same command with
a different `--runs`:

| row of the Results table | where it comes from |
|---|---|
| shipped ensemble, Dice **0.6552** / Det F1 **0.6063** / PR-AUC **0.7617** | the command above, row `0.5\|30\|0.0` of run `$A+$B` |
| 500-epoch member alone + decision layer (0.6565 / 0.6097) | same command with `--runs "$A"` |
| baseline with and without the filter (0.5333 → 0.5895) | same command with the 250-epoch baseline run, rows `0.5\|0\|0.0` and `0.5\|30\|0.0` |
| the container's 0.65 / 0.35 weighting (0.6570 / 0.6097) | `python3 analysis/weighted_ensemble.py` — `official_score.py` only ever averages members **equally**, which is how the paper quotes the shipped row |
| mm³ versus voxels, +0.0036 | `analysis/sweep_mm3_all5.py` then `analysis/site_transfer.py` (see below) |
| selection optimism, +0.0021 | `python3 analysis/nested_cv.py` |

`sweep_mm3_all5.py` and `lesion_pr.py` default to the **250-epoch baseline** run, which
`train_shipped.sh` does not train. Point them at a run you have — `--run "$A"` for the
first, `ISLES_RUN="$A"` for the second — or they will stop on a missing `validation/`
directory. The published numbers for both came from the baseline; a different run moves
the absolute values, and for the units comparison the contrast is paired within one
cache and survives the offset (see note 3 under Results).

`--npz` makes nnU-Net keep the validation softmax maps. Without them you cannot ensemble
the members, cannot fit the (τ, v_min) decision layer, and cannot compute PR-AUC.
**Keep `splits_final.json`** — every number here is out of fold against it, and any
second dataset you build must be handed the identical copy or no comparison is valid.

Budget roughly **6 h/fold** for the 500-epoch member on an uncontended card and up to
**15 h/fold** on a shared one ([`training/README.md`](training/README.md) publishes the
real per-fold table rather than one averaged figure), plus ~4.5 h/fold for the 250-epoch
TopK10 member. Several analysis scripts hardcode 12–32 worker processes; only
`official_score.py` and `sweep_mm3_all5.py` expose `--workers`. They will oversubscribe
a small machine.

---

## Citation

The paper is the thing to cite; it is not distributed here.

```bibtex
@inproceedings{kolli2026voxels,
  author    = {Kolli, Govinda and Dukre, Adinath Madhavrao and Razzak, Imran},
  title     = {Voxels or Millimetres? A Controlled Comparison of Lesion-Size
               Filtering for Native-Space Stroke Segmentation},
  booktitle = {MICCAI 2026 Workshop on SWITCH+},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear; volume and pages not yet assigned}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff) (Citation File
Format 1.2.0, validated against the 1.2.0 schema); its `preferred-citation` is the
SWITCH+ paper, i.e. the same record as the BibTeX entry above.

## Licence

The code in this repository is released under the **Apache License 2.0**. Parts of
`submission/` — `app.py`, the `Dockerfile`, the three `do_*.sh` scripts and the
`test/input/` fixtures — derive from the official ISLES'26 algorithm submission template,
which is itself Apache-2.0; [`submission/README.md`](submission/README.md) records that
provenance file by file. The full licence text is in [`LICENSE`](LICENSE), carrying the
authors' own copyright line. The template's own copy was not reused verbatim, because
its copyright holder line was still the unfilled `[yyyy] [name of copyright owner]`
placeholder and shipping that would have been misleading. [`NOTICE`](NOTICE) records the
third-party material item by item: the submission template, nnU-Net,
`dynamic-network-architectures`, Panoptica, every library the code actually imports with
its licence, and the material that is deliberately *not* redistributed here. One of those
libraries, **`connected-components-3d`**, is **LGPL-3.0-or-later** rather than permissive;
it is used unmodified, imported at runtime and never vendored here, so this repository
stays Apache-2.0 — but anyone redistributing a built artefact that contains it (the
submission image, for instance) inherits the LGPL's obligations for that copy.

Not covered by that licence, and not redistributed here:

* **`eval_utils.py`**, the ISLES'26 evaluation module, authored by **Ezequiel de la Rosa**
  (challenge organiser) and shipped without a license header. Obtain it from the
  challenge and place it at `analysis/official/eval_utils.py`; it is gitignored so that
  the copy you drop in cannot travel back out.
* The **ISLES'26 / ATLAS v3.0 data**, which is the organisers' to distribute under their
  own terms, along with every checkpoint, prediction volume and per-case score table
  derived from it.

## Acknowledgements

Every metric reported here is computed by the **ISLES'26 organisers' own evaluation
code**, used verbatim rather than reimplemented — that decision is the reason several of
these numbers are what they are, and it is what makes them comparable with the challenge's.
Thanks to the ISLES'26 organisers for the challenge, the native-space dataset and the
submission template, to the ATLAS v3.0 contributors for the data, and to the nnU-Net
authors, whose self-configuring defaults were the baseline that almost nothing here beat.
