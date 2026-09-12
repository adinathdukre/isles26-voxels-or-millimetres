# Model weights (not in this repository)

The shipped submission is a 10-checkpoint ensemble — two members x five folds —
of nnU-Net v2 ResEnc-M 3d_fullres networks. That is roughly **7.6 GB** of
`.pth` files, so they are not tracked here.

On Grand Challenge the weights are uploaded separately from the container image
(Your algorithm > Models) as a `model.tar.gz`, which is extracted to
`/opt/ml/model/` at runtime. `do_test_run.sh` reproduces that by bind-mounting
this directory read-only at the same path, and `do_save.sh` packs it into the
tarball you upload.

## Required layout

Populate this directory (or point `ISLES_MODEL_DIR` elsewhere) like so:

```
model/
├── nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres/
│   ├── dataset.json
│   ├── plans.json
│   ├── fold_0/checkpoint_final.pth
│   ├── fold_1/checkpoint_final.pth
│   ├── fold_2/checkpoint_final.pth
│   ├── fold_3/checkpoint_final.pth
│   └── fold_4/checkpoint_final.pth
└── topk10__nnUNetResEncUNetMPlans__3d_fullres/
    ├── dataset.json
    ├── plans.json
    ├── fold_0/checkpoint_final.pth
    ├── ...
    └── fold_4/checkpoint_final.pth
```

Each subdirectory is exactly what `nnUNetv2_train` writes under
`nnUNet_results/Dataset001_ISLES26/`, minus the training logs and the
`validation/` folders — `dataset.json`, `plans.json` and the five
`fold_*/checkpoint_final.pth` files are all `nnUNetPredictor` needs. The
directory names are the ones `NNUNET_MODEL_SUBDIRS` in `inference.py` looks for;
see `training/` for how the two members were trained.

`checkpoint_final.pth`, not `checkpoint_best.pth` — the reason is in the comment
above `CHECKPOINT_NAME` in `inference.py`.

## One rename you must apply to the TopK10 member

nnU-Net rebuilds the architecture by importing the trainer class named inside
each checkpoint. The second member was trained with a custom
`nnUNetTrainerDiceTopK10_250epochs` class that does not exist in the container
image, so the shipped TopK10 checkpoints have their `trainer_name` field
rewritten to the stock `nnUNetTrainer`:

```python
import torch

ckpt = torch.load(path, map_location="cpu", weights_only=False)
ckpt["trainer_name"] = "nnUNetTrainer"
torch.save(ckpt, path)
```

This is safe because TopK10 changes only the loss function, not the network —
predictions were verified bit-for-bit identical before and after the rename. The
source directory name (`nnUNetTrainerDiceTopK10_250epochs__...`) is likewise
shortened to `topk10__...` when copied here; that folder name is what
`inference.py` expects.
