# `training/` — dataset conversion and custom nnU-Net trainers

Everything needed to go from the ISLES'26 (ATLAS v3.0 Raw) download to the
trained checkpoints the submission ensembles. Scoring, the decision layer and
the paper's statistics live in [`../analysis/`](../analysis/); the container that
was actually submitted is [`../submission/`](../submission/).

---

## The shipped model, in one table

| | Member A | Member B |
|---|---|---|
| Trainer | `nnUNetTrainer_500epochs` | `nnUNetTrainerDiceTopK10_250epochs` |
| Where | **stock nnU-Net** (`nnunetv2/training/nnUNetTrainer/variants/training_length/`) | [`trainers/isles_wave4.py`](trainers/isles_wave4.py) |
| Plans | `nnUNetResEncUNetMPlans` | `nnUNetResEncUNetMPlans` |
| Configuration | `3d_fullres` | `3d_fullres` |
| Folds | 0–4 | 0–4 |
| Ensemble weight | 0.65 | 0.35 |

Decision layer applied on top of the averaged softmax: threshold **τ = 0.5**,
then drop predicted connected components smaller than **30 mm³**. Pooled
out-of-fold over all n = 1,453 subjects that configuration scores **Dice
0.6552, Detection F1 0.6063, PR-AUC 0.7617**.

Member A needs no code from this repository — it is a stock nnU-Net trainer.
Member B is one 5-line subclass in `trainers/isles_wave4.py`. That is the whole
"architecture" of the shipped system, and it is deliberate: the paper's claim is
about the *decision layer*, not about the network.

Reproduce both with:

```bash
source training/env.sh
training/train_shipped.sh
```

---

## Layout

```
training/
  env.sh                  every path and runtime knob, all with :- defaults
  convert_to_nnunet.py    ATLAS v3.0 Raw  ->  nnUNet_raw/Dataset001_ISLES26
  build_msl_dataset.py    Dataset504: multi-size labeling (an ablation arm)
  train_shipped.sh        the two shipped members, five folds each
  runner.py               queue-directory GPU scheduler used for the sweep
  trainers/               all custom trainer classes  (see trainers/README.md)
    isles_losses.py       Tversky / focal-Tversky, written to nnU-Net's dice_class contract
    isles_blob.py         blob loss (instance-imbalance-aware)
    isles_wave1.py        single-change ablations
    isles_wave2.py        combinations of those levers
    isles_wave4.py        clDice, CE×2, **Dice+TopK10**, less-augmentation arms
    isles_primus.py       Primus transformer backbone
```

---

## 0. Prerequisites

* `nnunetv2 == 2.8.0` (the `nnUNet_extTrainer` mechanism in §3 needs ≥ 2.8)
* PyTorch with CUDA, a card with ≥ 20 GB free for `3d_fullres` at batch 2
* `SimpleITK`, `numpy`
* `connected-components-3d` (`cc3d`) — blob loss falls back to
  `scipy.ndimage.label` without it, at roughly 5× the CPU cost per step;
  `build_msl_dataset.py` requires it outright
* `dynamic_network_architectures` — ships with nnU-Net; only the Primus arms use it

## 1. Environment

```bash
source training/env.sh
```

Sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, the
`nnUNet_extTrainer` search path, `PYTHONPATH`, and the two runtime knobs the
sweep deliberately pinned (`nnUNet_compile=f`, `nnUNet_n_proc_DA=12` — the file
explains why). It also exports `ISLES_NNUNET_RESULTS` and `ISLES_GT` so that
[`analysis/paths.py`](../analysis/paths.py) resolves to the same files.

Everything derives from `ISLES_ROOT`, which defaults to the parent of this
directory. Point it elsewhere if your nnU-Net directories are not inside the
checkout:

```bash
export ISLES_ROOT=/data/isles26
source training/env.sh
```

## 2. Dataset conversion

Download and extract the ISLES'26 training release (ATLAS v3.0 Raw) from the
challenge, then:

```bash
export ISLES_ATLAS3_RAW=/path/to/extracted/ATLAS3_Training_Raw
python3 training/convert_to_nnunet.py
```

This walks `<SITE>/sub-*/ses-*/anat/`, **symlinks** each `*_T1w.nii.gz` to
`imagesTr/<case>_0000.nii.gz` and each `*_mask.nii.gz` to
`labelsTr/<case>.nii.gz`, and writes `dataset.json` (one channel `T1w`, labels
`background`/`lesion`) plus `case_metadata.csv` (site, chronicity,
days-post-stroke — the per-case metadata table; the shipped model is T1w-only
and never reads it). Symlinks, not copies: nnU-Net reads the raw volumes exactly
once, during preprocessing.

Two subjects have empty metadata rows but valid images and masks. They are
included as training cases; only the metadata table skips them. Expect
**1,453** cases. (Which two is a property of the released data, not of this
algorithm; this repository reproduces no subject identifiers.)

Then plan and preprocess with the ResEnc-M planner:

```bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncM -c 3d_fullres \
    --verify_dataset_integrity
```

The resulting `nnUNetResEncUNetMPlans` for this dataset is 128³ patches, batch
2, 1 mm isotropic spacing, `batch_dice=False`, z-score normalisation with
`use_mask_for_norm`. nnU-Net generates `splits_final.json` on first training:
five folds of 1,162/291, 1,162/291, 1,162/291, 1,163/290, 1,163/290. **Keep that
file.** Every number in the paper is out-of-fold against it, and any other
dataset you build (§6) must be given the identical copy or no comparison is
valid.

## 3. Registering the custom trainers

nnU-Net ≥ 2.8 looks for trainer classes outside its own package when
`nnUNet_extTrainer` is set, splitting it on the path separator and scanning each
directory (`nnunetv2/utilities/find_objects.py`). `env.sh` sets it to:

```
$ISLES_ROOT/training/trainers
```

After that, `-tr nnUNetTrainerDiceTopK10_250epochs` just works. Three things
worth knowing:

* **The value is a path-separator-joined list**, so a second trainer directory
  of your own can be appended without touching `trainers/`.
* **nnU-Net only puts the directory it is currently scanning on `sys.path`.** A
  trainer that imports a class out of a *different* trainer directory therefore
  needs that directory on `PYTHONPATH` as well, which is why `env.sh` prepends
  the list there too. `isles_wave2.py` does exactly this for its two
  metadata-conditioning combination arms, and wraps the import in
  `try/except ImportError`: getting the path wrong does not crash — it silently
  drops those two arms, which is much worse. The FiLM trainer they subclass is
  not part of this repository, so in a fresh checkout those two classes are
  simply never defined and the other six in the file load normally.
* **On older nnU-Net there is no `nnUNet_extTrainer`.** Copy (or symlink) the
  files in `trainers/` into
  `nnunetv2/training/nnUNetTrainer/variants/` instead; nothing else changes.

### The signature rule for new trainers

Quoting the note at the top of [`isles_wave1.py`](trainers/isles_wave1.py):

> `nnUNetTrainer.__init__` rebuilds its own init kwargs with
> `inspect.signature(self.__init__)` + `locals()`, so every subclass must repeat
> the full explicit signature — a `*args, **kwargs` passthrough raises KeyError.

So every trainer here, even a one-line one, carries the full
`(self, plans, configuration, fold, dataset_json, device)` signature. Keep it
that way when you add an arm.

## 4. Train the shipped configuration

```bash
source training/env.sh
training/train_shipped.sh          # all five folds of both members
training/train_shipped.sh 0 2      # or just some folds
```

which is exactly ten invocations of:

```bash
nnUNetv2_train 1 3d_fullres $FOLD -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainer_500epochs --npz
nnUNetv2_train 1 3d_fullres $FOLD -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainerDiceTopK10_250epochs --npz
```

**`--npz` is not optional here.** It makes nnU-Net keep the validation softmax
maps, and without them you cannot ensemble the two members, cannot fit the
(τ, minimum-size) decision layer, and cannot compute PR-AUC. Regenerating them
afterwards means re-running validation on all five folds.

### Expected runtime

Measured wall clock for the 500-epoch ResEnc-M member, per fold, from the
training logs of the actual shipped run:

| Fold | mean s/epoch | hours |
|---|---|---|
| 0 | 44.8 | 6.2 |
| 1 | 44.6 | 6.2 |
| 2 | 104.6 | **14.5** |
| 3 | 41.0 | 5.7 |
| 4 | 45.6 | 6.3 |

Fold 2 is not a different model — it is the same job on a card that was being
shared with other users' work. **Budget ~15 h/fold** (~75 h for the five folds)
if the GPU is contended, and expect closer to 6 h/fold if it is not. The
250-epoch TopK10 member came in at ~4.5 h/fold uncontended.

For scale, the arms that were *not* shipped: ResEnc-L cost ~41 h per fold
against ResEnc-M's ~15, which made it the worst compute trade in the sweep.

Validation-only re-runs, to regenerate `.npz` maps without retraining, are
`nnUNetv2_train … --val --npz`.

## 5. Running the whole sweep

`runner.py` is how the sweep's ablation arms were actually executed: a queue
directory of JSON job files, one worker thread per GPU slot, each worker
claiming the highest-priority pending job and refusing to start it until the
(shared) card reports enough free VRAM.

```bash
source training/env.sh
cat > "$ISLES_SWEEP_DIR/queue/w4_dicetopk10.json" <<'J'
{"id":"w4_dicetopk10","trainer":"nnUNetTrainerDiceTopK10_250epochs",
 "plans":"nnUNetResEncUNetMPlans","config":"3d_fullres","dataset":1,"fold":0,
 "extra_args":["--npz"],"min_free_gb":20,"priority":1}
J
python3 training/runner.py --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 1
```

Jobs can be dropped in while it is already running. Finished jobs move to
`queue_done/`, failures to `queue_failed/`, and `logs/status.json` carries the
live state plus the hours each job took. `$ISLES_SWEEP_DIR` (default
`$ISLES_ROOT/sweep`) is pure scratch — queue plus logs, nothing to commit.

If you only want the shipped model, skip this and use `train_shipped.sh`.

## 6. Variants that are plans edits, not trainers

The patch-size and batch-size arms are not trainers at all; they are copies of
the plans JSON. All of them share `data_identifier = nnUNetPlans_3d_fullres`,
so **none of them needs re-preprocessing**:

```python
import json, os
d = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset001_ISLES26")
p = json.load(open(f"{d}/nnUNetResEncUNetMPlans.json"))
p["plans_name"] = "nnUNetResEncUNetMPlans_p128b4"          # <- see the warning below
p["configurations"]["3d_fullres"]["batch_size"] = 4
json.dump(p, open(f"{d}/nnUNetResEncUNetMPlans_p128b4.json", "w"), indent=1)
```

> **Set `plans_name` to match the filename.** It is the field that drives the
> output folder name inside `nnUNet_results`. A mismatch does not error — it
> silently writes the variant's checkpoints into the *baseline's* results
> directory, on top of the run you were comparing against.

Arms run this way: `_p128b4` (batch 4), `_p128b8` (batch 8) and `_p160` (160³
patches). Batch 8 never fit in VRAM and has no final checkpoint; the other two
ran to completion.

Oversampling is the other half of this axis and *is* a trainer, because it is a
plain attribute: `nnUNetTrainer_250epochs_oversample066` /
`_oversample090` in [`isles_wave1.py`](trainers/isles_wave1.py). 0.90 failed.

### Multi-size labeling (a second raw dataset)

```bash
python3 training/build_msl_dataset.py
nnUNetv2_plan_and_preprocess -d 504 -pl nnUNetPlannerResEncM -c 3d_fullres
cp "$nnUNet_preprocessed/Dataset001_ISLES26/splits_final.json" \
   "$nnUNet_preprocessed/Dataset504_ISLES26_MSL/splits_final.json"   # MANDATORY
nnUNetv2_train 504 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_250epochs --npz
```

Splits each ground-truth component into one of four classes by voxel count
(<100, 100–1k, 1k–10k, ≥10k), after Shang et al. 2024; merged back to a single
lesion class at inference. Copying `splits_final.json` across is not optional —
without it the two datasets get different folds and nothing is comparable.

It did not reproduce here: merged MSL lost to the plain single-class baseline on
the challenge's aggregate rank objective on fold 0. Kept because it is the
cleanest published size-aware labeling arm we ran, and a negative replication of
an ATLAS-specific method is worth fifty lines.

## 7. Gotchas the sweep paid for

These are recorded in the trainer docstrings too; collected here because each
one cost real GPU hours.

* **clDice returned exactly 0.0 for every input**, for a while. `-a.min(b)`
  parses as `-(a.min(b))`, so the compact one-liner soft-erosion eroded the
  wrong sign and produced an empty skeleton. The fixed version writes the three
  directional erosions out longhand
  ([`isles_wave4.py`](trainers/isles_wave4.py), `_soft_erode`).
* **Primus crashes before epoch 1** on
  `set_deep_supervision_enabled`, which does `mod.decoder.deep_supervision = x`
  and assumes a U-Net. Primus has no `.decoder`. Overridden as a no-op in
  [`isles_primus.py`](trainers/isles_primus.py). The `ConnectionResetError` that
  appears alongside it is the dataloader tearing down — a red herring.
* **Primus also needs AdamW**, not nnU-Net's SGD(0.01, nesterov), which diverges.
  Following the paper's optimiser rather than the framework default is the only
  way the run tests the *architecture* instead of the recipe.
* **Batch 8 does not fit**; oversampling 0.90 did not complete either.
* **blob loss labels components on the CPU**, ~30–60 ms per ~250 ms step with
  `cc3d`. It applies the instance term at full resolution only: the
  deep-supervision heads keep plain Dice+CE, because relabelling at every scale
  multiplies that cost and the low-resolution heads cannot resolve small lesions
  anyway — which is the entire point of the loss.

## 8. What happens next

Training produces checkpoints and (with `--npz`) out-of-fold softmax maps. From
there:

1. **Fit the decision layer** — the joint (τ, minimum component size in mm³)
   sweep, on *pooled out-of-fold predictions for all 1,453 subjects*, not on one
   291-case fold. Fitting on fold 0 alone overstated the size filter's gain by
   more than 2×. See [`../analysis/`](../analysis/).
2. **Ensemble** the two members at 0.65 / 0.35 and re-score.
3. **Build the submission container** — see [`../submission/`](../submission/).

Scoring uses the official ISLES'26 evaluation code, which is not redistributed
here; `../analysis/README.md` says where to obtain it and where to put it.

## 9. What was dropped from this directory, and why

Faithfully, so nobody goes looking for something that was left behind
deliberately:

* `baseline/train_folds_2to4.sh`, `baseline/train_remaining_folds.sh` — one-off
  job scripts that activated a specific user's conda environment by absolute
  path and blocked on a hard-coded PID (`while kill -0 71280`). Their only
  reproducible content is the `nnUNetv2_train` loop, which is now
  `train_shipped.sh`.
* `sweep/status.sh` — a wall-display monitor that scraped epoch numbers out of
  training logs and paired them with `nvidia-smi`. Useful on that machine, no
  value in a checkout. The runtime numbers it was used to collect are in §4.
* `sweep/wait_and_score.sh` — a `sleep 120` poll loop waiting for N validations
  to appear before calling the scorer. Pure operational glue.
* `sweep/score.sh`, `sweep/leaderboard.sh`, `sweep/fit_operating_point.sh` —
  wrappers around the scorer (`nohup`, PID files, a wait loop). They belong to
  the analysis half of the project, not to training; the substance — the
  τ ∈ {0.5 … 0.9} × v_min ∈ {0, 20, 30, 40, 60} mm³ grid fitted on pooled
  out-of-fold predictions — is in [`../analysis/`](../analysis/).
* `sweep/queue*/`, `sweep/logs/` — several hundred finished job JSONs and
  training logs. Runtime state. One representative job file is inlined in §5.
* `baseline/cpu_timing_test/` and the `bench_*.py` scripts — inference
  throughput probes for choosing submission hardware, superseded by the
  submission container's own timing.
