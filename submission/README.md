# `submission/` — the Grand Challenge container

This is the algorithm container that was submitted to **ISLES'26**. It is a
FastAPI server that Grand Challenge boots once per job, then calls once per
case; `inference.py` holds the whole method — the weighted two-member ensemble,
the instance decision layer the paper is about, and the failure guard.

`inference.py` is the file to read. Its comments record *why* each constant has
the value it has and what was measured to get there; they are the primary
documentation of the shipped operating point, and they are reproduced here
unedited.

---

## What it does, per case

```
/input/images/t1-brain-mri/*.mha          (one skull-stripped T1w volume, native space)
/input/stroke-metadata.json               (CENTER / CHRONICITY / DAYS_POST_STROKE — logged, not used)
          │
          ├─ read reference geometry (header only) ─────────────────┐
          │                                                         │
          ├─ member A: nnU-Net ResEnc-M, 500 epochs, 5 folds        │ needed by
          ├─ member B: nnU-Net ResEnc-M, Dice+TopK10, 5 folds       │ every output,
          │    └─ weighted average of softmax maps  (0.65 / 0.35)   │ including the
          │                                                         │ fallback
          ├─ threshold at τ = 0.5  ──────────────► binary mask      │
          │                                                         │
          ├─ instance decision layer: drop 26-connected components  │
          │    smaller than 30 mm³ (physical units, not voxels)     │
          │                                                         │
          └─ NaN/range clamp on the soft map ──────────────────────┘
          │
/output/images/stroke-lesion-segmentation/output.mha   (uint8 mask, filtered)
/output/images/lesion-probability-map/output.mha       (float32 soft map, UNfiltered)
```

The soft map is scored separately, for PR-AUC only, so it ships **unfiltered** —
filtering it could only destroy ranking information the binary mask's threshold
has already discarded.

The model is **T1w-only**. The stroke metadata is read and logged for
traceability but never reaches the network; reading it is advisory and cannot
fail a case.

---

## The shipped configuration, exactly

| | value | where |
|---|---|---|
| architecture | nnU-Net v2, ResEnc-M, `3d_fullres` | `plans.json` in each member — it ships with the weights, not here; the planner and the resulting configuration are in [`../training/README.md`](../training/README.md#2-dataset-conversion) |
| member A | `nnUNetTrainer_500epochs`, folds 0–4 | `NNUNET_MODEL_SUBDIRS[0]` |
| member B | `nnUNetTrainerDiceTopK10_250epochs`, folds 0–4 | `NNUNET_MODEL_SUBDIRS[1]` |
| ensemble weights | **0.65 / 0.35** (A / B), applied to softmax maps | `ENSEMBLE_WEIGHTS` |
| checkpoint | `checkpoint_final.pth` (**not** `_best`) | `CHECKPOINT_NAME` |
| binarisation | **τ = 0.5** | `BINARY_THRESHOLD` |
| instance decision layer | drop components **< 30 mm³**, 26-connectivity | `MIN_LESION_MM3` |
| test-time augmentation | mirroring, GPU only | `init_model()` |
| tiling | `tile_step_size=0.5`, Gaussian weighting | `init_model()` |
| constant-map trick | **off** | `EMIT_CONSTANT_MAP_WHEN_EMPTY` |

Pooled out-of-fold performance, all 1,453 training subjects, scored with the
organisers' evaluation code:

| Dice | Detection F1 | PR-AUC |
|---|---|---|
| 0.6552 | 0.6063 | 0.7617 |

Those three are the **equal-weight** member average, which is how the paper
quotes the ensemble. The container ships 0.65 / 0.35, which the weight sweep in
[`../analysis/weighted_ensemble.py`](../analysis/weighted_ensemble.py) measured
as better on four of five metrics — Dice 0.6570, Detection F1 0.6097, |ΔN| 1.80,
|ΔV| 5.30 mL — at no inference cost. That sweep's PR-AUC column is a *naive*
value computed without the organisers' empty-ground-truth convention, so it is
not comparable with the 0.7617 above, and the two were deliberately never merged
into one row. Every individual delta from the weight change is below its own
reference floor; what makes it credible is that four move together.

For scale, the two-seed reference floors (what two runs of the *same*
configuration differ by) are Dice 0.0043, Detection F1 0.0057, PR-AUC 0.0034.

### Why 30 mm³ and not "20 voxels"

ISLES'26 is a **native-space** challenge: across the 1,453 training volumes,
voxel volume spans 0.031–5.27 mm³, a 168× spread, with only 71.6% of cases
inside [0.9, 1.1] mm³. A 20-voxel rule — the form every prior ATLAS
post-processing rule takes, because ATLAS v2.0 was MNI-registered at 1 mm³ —
means 0.63 mm³ on one scanner and 105 mm³ on another.

The measured effects, all on pooled out-of-fold predictions:

* **filtering at all:** Detection F1 0.5333 → 0.5895, lesion-count error 2.08 → 1.85
* **mm³ vs voxels at matched magnitude:** +0.0036 Detection F1, 95% CI [0.0016, 0.0058]
* **nested-CV selection optimism** on the fitted operating point: +0.0021

The figures quoted in `inference.py`'s own comments (0.5283 → 0.5849, Dice
0.6391) come from the earlier single-member sweep at which 30 mm³ was first
fitted; the numbers above are from the later joint (τ, mm³) fit and the shipped
ensemble. They are the same experiment measured at two points in the project,
and both are left as written rather than retrofitted. The full size-rule trace —
every setting from no filter through 120 mm³, with the column means to check a
rebuild against — is the table in
[`../analysis/README.md`](../analysis/README.md), and
[`../analysis/sweep_mm3_all5.py`](../analysis/sweep_mm3_all5.py) is what
regenerates it. The controlled units comparison is the subject of the paper this
repository accompanies; `CITATION.cff` at the repository root has the full
citation.

### Why τ = 0.5 is not a tuning knob

A single-fold sweep said τ = 0.8 was worth +0.008 Detection F1. It did not
replicate on 1,453 cases — out-of-fold, τ = 0.5 is rank-optimal and everything
above ~0.6 is progressively worse. Do not re-tune τ on one fold; and note that
averaging ten softmaxes flattens the distribution relative to any single member,
so a cutoff fitted on single-fold predictions would not transfer cleanly anyway.

### The failure guard

Inference is wrapped in a try/except that, on any exception, prints the
traceback and writes **geometrically valid empty outputs** — so a pathological
case scores 0 and every other case is unaffected. Without it, one unexpected
NIfTI header or one CUDA OOM propagates out, fails the container, and scores
zero on *every* case. This is why the reference geometry is read first, before
anything that can fail: the fallback needs it to write anything at all. Only
`find_input_image` and that header read sit outside the guard, and without
geometry there is no output to write, so that exposure is irreducible.

Two smaller pieces belong to the same guard. Reading `stroke-metadata.json` is
advisory and carries its own try/except, because a missing or malformed metadata
file must never be able to fail a case for a model that does not consume
metadata. And the NaN/inf clamp on the soft map sits *after* the try/except
rather than inside it, so it covers the fallback outputs as well as the normal
ones — a NaN or inf reaching the writer would produce an unreadable map, and
clamping is cheaper than trusting the network's output range.

Adding the guard was verified behaviour-neutral: re-running the container test
gave byte-identical outputs (196,262 mask voxels, 4 components).

**Known untested path.** The machine this was built on has no NVIDIA container
runtime, so the container always fell back to CPU — and on CPU `init_model()`
deliberately drops to **one** ensemble member and disables mirroring TTA to stay
inside the per-case runtime budget. The two-member GPU path that actually runs
on Grand Challenge was never exercised inside a container locally. An earlier
image did pass the organisers' preliminary evaluation, and the weighting change
since alters coefficients rather than the number of forward passes, but this is
stated rather than glossed.

---

## Model weights are not in this repository

The ten checkpoints (2 members × 5 folds) are ~**7.6 GB**. They are uploaded to
Grand Challenge separately from the image, as a `model.tar.gz` that is extracted
to `/opt/ml/model/` at runtime.

Supply them yourself under `submission/model/` (or point `ISLES_MODEL_DIR`
anywhere else):

```
model/
├── nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres/
│   ├── dataset.json
│   ├── plans.json
│   └── fold_{0,1,2,3,4}/checkpoint_final.pth
└── topk10__nnUNetResEncUNetMPlans__3d_fullres/
    ├── dataset.json
    ├── plans.json
    └── fold_{0,1,2,3,4}/checkpoint_final.pth
```

Each directory is what `nnUNetv2_train` writes under
`nnUNet_results/Dataset001_ISLES26/`, minus training logs and `validation/`.
`model/README.md` has the details — including the one mandatory edit to the
TopK10 checkpoints (`trainer_name` must be rewritten to the stock
`nnUNetTrainer`, because the custom trainer class does not exist in the image;
verified bit-for-bit identical predictions before and after). `training/`
explains how the two members were trained.

No imaging data is in this repository either. Before testing, drop one
skull-stripped T1w volume into
`test/input/interf0/images/t1-brain-mri/`.

---

## Build, test, export

```bash
cd submission

# 1. build the image
./do_build.sh

# 2. run it exactly as Grand Challenge does: boot the server, poll /health,
#    POST /invoke, collect /output
./do_test_run.sh
#    results appear under test/output/interf0/images/

# 3. pack the image and the model for upload
./do_save.sh
#    -> example_algorithm_..._<timestamp>.tar.gz   (upload as the container image)
#    -> model.tar.gz                               (upload separately, as a Model)
```

`do_test_run.sh` builds the image, starts it detached, waits on `/health` with
Grand Challenge's own timeouts, issues `POST /invoke` from a `curl` sidecar
container, and copies `/output` back to the host. Environment overrides:

| variable | default | purpose |
|---|---|---|
| `ISLES_MODEL_DIR` | `./model` | where the checkpoints live |
| `ISLES_TEST_GPUS` | `all` | value passed to `docker --gpus`, e.g. `device=2` on a shared box |
| `ISLES_TEST_NET` | `published` | `internal` uses an `--internal` network and container-name DNS (closest to Grand Challenge); `published` publishes port 4743 to the host, which is what this submission was tested with because name resolution on the internal network did not work on that machine — it does **not** reproduce the no-internet restriction |

With no NVIDIA runtime the test still runs, on CPU, with one ensemble member —
useful as a plumbing check, but not the shipped configuration.

### Resource envelope

* ~41 s/case on GPU for both members (measured); ~175 s/case on CPU for one.
* Peak VRAM ~3 GB — members run sequentially, not side by side.
* Image ~4 GB compressed; model tarball ~7.6 GB.
* Base image is CUDA 12.6, required by the T4 instances Grand Challenge uses.
* The server must listen on port **4743**, `/health` must return 200 without
  redirecting, and `/invoke` must return **201**. The
  `org.grand-challenge.api-method="invoke"` label in the Dockerfile is required;
  `do_test_run.sh` refuses to continue without it.

---

## Files

| file | role |
|---|---|
| `inference.py` | **the method.** `init_model()` + `run()`; ensemble, decision layer, failure guard |
| `app.py` | FastAPI server implementing Grand Challenge's `/health` + `/invoke` contract. Unmodified from the organisers' template — you should not need to touch it |
| `Dockerfile` | CUDA 12.6 PyTorch base, venv with `--system-site-packages`, the required API label |
| `requirements.txt` | pinned to the exact versions in the image that passed the container test; torch/CUDA come from the base image and are deliberately absent |
| `do_build.sh` | build |
| `do_test_run.sh` | build + boot + health + invoke + collect |
| `do_save.sh` | export image tarball and `model.tar.gz` |
| `model/README.md` | required checkpoint layout (weights not shipped) |
| `test/input/interf0/` | `inputs.json` and `stroke-metadata.json` fixtures; drop a T1w into `images/t1-brain-mri/` |

The absolute paths in `inference.py` (`/input`, `/output`, `/opt/ml/model`) are
Grand Challenge's runtime contract, not machine-specific paths, and are left as
they are.

---

## Provenance

`app.py`, `Dockerfile`, `do_build.sh`, `do_save.sh`, `do_test_run.sh` and the
`test/input/` fixtures derive from the official **ISLES'26 algorithm submission
template** (Apache-2.0). `inference.py` is ours apart from the template's
docstring and I/O helpers. `do_test_run.sh` has been tidied relative to the
template — dead commented-out blocks removed, machine-specific choices moved
behind the environment variables above — but its behaviour is unchanged.

Reproducing the evaluation numbers quoted here additionally requires the
organisers' scorer; see `analysis/official/` for how to obtain it.
