# Custom nnU-Net trainers

Every ablation arm in the paper is a trainer class in one of these files. nnU-Net
finds them through the `nnUNet_extTrainer` environment variable — see
[`../README.md`](../README.md#3-registering-the-custom-trainers) for how that
works and for the `__init__` signature rule you must follow when adding one.

**The one that ships:** `nnUNetTrainerDiceTopK10_250epochs` in
[`isles_wave4.py`](isles_wave4.py). It is the 0.35-weight member of the shipped
ensemble. The 0.65-weight member is `nnUNetTrainer_500epochs`, which is **stock
nnU-Net** (`nnunetv2/training/nnUNetTrainer/variants/training_length/`) and
needs no code from this directory at all.

## Index

| File | Classes | Axis it tests |
|---|---|---|
| [`isles_losses.py`](isles_losses.py) | `MemoryEfficientSoftTverskyLoss`, `MemoryEfficientFocalTverskyLoss`, `tversky_factory`, `focal_tversky_factory` | Loss building blocks, not trainers. Drop-in `dice_class` replacements for `DC_and_CE_loss`. |
| [`isles_blob.py`](isles_blob.py) | `BlobLoss`, `_BlobDSWrapper`, `nnUNetTrainerBlobLoss_250epochs`, `nnUNetTrainerBlobLossW1_250epochs` | blob loss (Kofler et al. 2023): every lesion instance counts the same regardless of size. |
| [`isles_wave1.py`](isles_wave1.py) | `nnUNetTrainer_250epochs_repro`, `nnUNetTrainerTverskyCE_250epochs`, `nnUNetTrainerTverskyCE_a02b08_250epochs`, `nnUNetTrainerFocalTverskyCE_250epochs`, `nnUNetTrainer_250epochs_batchDice`, `nnUNetTrainer_250epochs_oversample066`, `nnUNetTrainer_250epochs_oversample090`, `nnUNetTrainerDA5_250epochs` | Single-change ablations: one lever each, so every result is attributable. Includes the same-machine control retrain. |
| [`isles_wave2.py`](isles_wave2.py) | `nnUNetTrainerTverskyBatchDice_250epochs`, `nnUNetTrainerTverskyOversample_250epochs`, `nnUNetTrainerBlobTversky_250epochs`, `nnUNetTrainerBlobBatchDice_250epochs`, `nnUNetTrainerDA5Tversky_250epochs`, `nnUNetTrainerDA5Blob_250epochs`, `nnUNetTrainerFiLMBlob_250epochs`*, `nnUNetTrainerFiLMTversky_250epochs`* | Combinations of the wave-1 levers, plus the combined method (metadata conditioning + instance-aware loss). |
| [`isles_wave4.py`](isles_wave4.py) | `SoftClDiceLoss`, `nnUNetTrainerClDice_250epochs`, `nnUNetTrainerDiceCE_w2_250epochs`, **`nnUNetTrainerDiceTopK10_250epochs`**, `nnUNetTrainerDiceCEnoSmooth_250epochs`, `nnUNetTrainerNoDA_250epochs`, `nnUNetTrainerDA5ord0_250epochs`, `nnUNetTrainerOnlyMirror01_250epochs` | Connectivity loss (clDice), loss rebalancing, and the *less*-augmentation end of the augmentation axis. |
| [`isles_primus.py`](isles_primus.py) | `nnUNetTrainerPrimusM_250epochs`, `nnUNetTrainerPrimusB_250epochs`, `nnUNetTrainerPrimusS_250epochs` | Primus (Wald et al. 2025), a ViT-style backbone, under the same split and budget as everything else. |

\* The two metadata-conditioning combination classes are defined inside a
`try: … except ImportError` block, because nnU-Net imports every module in every
external trainer directory and the FiLM trainer they subclass is not part of this
repository. In a fresh checkout that import fails, those two classes are simply
never defined, and the other six in the file load normally — which is exactly the
behaviour the `try/except` exists to give. Supply the FiLM trainer on
`PYTHONPATH` and they appear; `training/env.sh` is where that path is set.

## Reading order

Start with `isles_losses.py` (the Tversky family, written to nnU-Net's
`dice_class` contract), then `isles_wave1.py` (how a one-line trainer subclass
turns a loss into an experiment arm), then `isles_blob.py` and `isles_wave4.py`
for the two losses with real machinery behind them.

Each file's module docstring says **why that arm was tried**, in terms of
something measured earlier in the project rather than something fashionable in
the literature. Several of them also record the bug that made the first version
of the arm meaningless. Those notes are the point; do not strip them.
