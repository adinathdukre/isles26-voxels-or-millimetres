"""
Wave-1 trainers for the ISLES'26 sweep: single-change ablations against the
250-epoch ResEnc-M baseline (fold 0, mean Dice 0.6668), so every result is
attributable to exactly one change.

Discovered by nnU-Net through the nnUNet_extTrainer env var (see training/env.sh).

NOTE ON SIGNATURES: nnUNetTrainer.__init__ rebuilds its own init kwargs with
`inspect.signature(self.__init__)` + `locals()`, so every subclass must repeat
the full explicit signature -- a `*args, **kwargs` passthrough raises KeyError.
"""
import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5

from isles_losses import tversky_factory, focal_tversky_factory


# ---------------------------------------------------------------------------
# Control: identical to the baseline recipe, retrained on THIS machine and THIS
# nnU-Net version (2.8.0 vs. whatever produced the 0.6668 reference). Without
# it, every comparison below is confounded by the version/hardware change.
# ---------------------------------------------------------------------------
class nnUNetTrainer_250epochs_repro(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


# ---------------------------------------------------------------------------
# Loss ablations
# ---------------------------------------------------------------------------
class _TverskyTrainerBase(nnUNetTrainer):
    ALPHA = 0.3
    BETA = 0.7
    GAMMA = None  # set for focal variants

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not used for ISLES'26"
        dice_class = (focal_tversky_factory(self.ALPHA, self.BETA, self.GAMMA)
                      if self.GAMMA is not None else tversky_factory(self.ALPHA, self.BETA))
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label, dice_class=dice_class)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyCE_250epochs(_TverskyTrainerBase):
    """Tversky(0.3, 0.7) + CE -- penalizes false negatives ~2.3x harder than false positives."""
    ALPHA, BETA = 0.3, 0.7


class nnUNetTrainerTverskyCE_a02b08_250epochs(_TverskyTrainerBase):
    """More aggressive recall bias; brackets the alpha/beta axis."""
    ALPHA, BETA = 0.2, 0.8


class nnUNetTrainerFocalTverskyCE_250epochs(_TverskyTrainerBase):
    """Focal Tversky (alpha 0.3, beta 0.7, gamma 3/4) + CE."""
    ALPHA, BETA, GAMMA = 0.3, 0.7, 0.75


# ---------------------------------------------------------------------------
# Dice aggregation: per-sample (baseline) vs. over the whole batch.
# batch_dice makes the Dice term a batch-global statistic, which nnU-Net's own
# guidance recommends for small/sparse structures -- a patch holding only a
# handful of lesion voxels otherwise yields a near-degenerate per-sample Dice.
# ---------------------------------------------------------------------------
class nnUNetTrainer_250epochs_batchDice(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.configuration_manager.configuration['batch_dice'] = True


# ---------------------------------------------------------------------------
# Foreground oversampling: the fraction of training patches forced to be centred
# on a lesion voxel. nnU-Net's default is 0.33; small lesions occupy a vanishing
# fraction of the volume, so most random patches contain no lesion at all.
# ---------------------------------------------------------------------------
class nnUNetTrainer_250epochs_oversample066(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.oversample_foreground_percent = 0.66


class nnUNetTrainer_250epochs_oversample090(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.oversample_foreground_percent = 0.90


# ---------------------------------------------------------------------------
# Augmentation strength: nnU-Net's DA5 preset (much heavier spatial/intensity
# augmentation) as the cross-site generalization lever ISLES'26 is built around.
# ---------------------------------------------------------------------------
class nnUNetTrainerDA5_250epochs(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
