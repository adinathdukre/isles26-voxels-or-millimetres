"""
Wave-2 trainers: combinations of the wave-1 levers, plus the combined method.

The design bet, driven by what we measured rather than by what is fashionable:
the ISLES'26 failure is *detection* (1,817 false-positive components and 2,198
missed lesions out of ~4,900 across the 1,453-case out-of-fold set), so the
three levers worth stacking are

  1. metadata conditioning (FiLM)  -- measurably improves detection: fewer false
     positives AND more true lesions found, even though it is flat on Dice,
  2. an instance-aware loss (blob) -- makes every lesion count equally regardless
     of size, which global Dice does not,
  3. recall-biased Tversky         -- shifts the operating point toward finding
     lesions, with the false positives it invites cleaned up afterwards by the
     learned component filter (see analysis/).

Only combinations whose single-change ablations actually won in wave 1 should be
launched; the classes are all defined here so queueing one is a one-liner.
"""
import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5

from isles_losses import tversky_factory
from isles_blob import BlobLoss, _BlobDSWrapper


def _build_tversky_loss(trainer, alpha, beta):
    loss = DC_and_CE_loss(
        {'batch_dice': trainer.configuration_manager.batch_dice, 'smooth': 1e-5,
         'do_bg': False, 'ddp': trainer.is_ddp},
        {}, weight_ce=1, weight_dice=1, ignore_label=trainer.label_manager.ignore_label,
        dice_class=tversky_factory(alpha, beta))
    if trainer.enable_deep_supervision:
        scales = trainer._get_deep_supervision_scales()
        w = np.array([1 / (2 ** i) for i in range(len(scales))])
        w[-1] = 0
        loss = DeepSupervisionWrapper(loss, w / w.sum())
    return loss


def _build_blob_loss(trainer, blob_weight, dice_class=MemoryEfficientSoftDiceLoss):
    def mk():
        return DC_and_CE_loss(
            {'batch_dice': trainer.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': trainer.is_ddp},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=trainer.label_manager.ignore_label, dice_class=dice_class)
    blob = BlobLoss(mk(), blob_weight=blob_weight)
    if trainer.enable_deep_supervision:
        scales = trainer._get_deep_supervision_scales()
        w = np.array([1 / (2 ** i) for i in range(len(scales))])
        w[-1] = 0
        return _BlobDSWrapper(blob, mk(), w / w.sum())
    return blob


# ---------------------------------------------------------------------------
# Loss x aggregation / sampling / augmentation combinations
# ---------------------------------------------------------------------------
class nnUNetTrainerTverskyBatchDice_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.configuration_manager.configuration['batch_dice'] = True

    def _build_loss(self):
        return _build_tversky_loss(self, 0.3, 0.7)


class nnUNetTrainerTverskyOversample_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.oversample_foreground_percent = 0.66

    def _build_loss(self):
        return _build_tversky_loss(self, 0.3, 0.7)


class nnUNetTrainerBlobTversky_250epochs(nnUNetTrainer):
    """Instance-aware loss whose per-instance term is itself recall-biased."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        return _build_blob_loss(self, 2.0, dice_class=tversky_factory(0.3, 0.7))


class nnUNetTrainerBlobBatchDice_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.configuration_manager.configuration['batch_dice'] = True

    def _build_loss(self):
        return _build_blob_loss(self, 2.0)


class nnUNetTrainerDA5Tversky_250epochs(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        return _build_tversky_loss(self, 0.3, 0.7)


class nnUNetTrainerDA5Blob_250epochs(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        return _build_blob_loss(self, 2.0)


# ---------------------------------------------------------------------------
# The combined method: metadata conditioning + instance-aware loss.
# Imported lazily so this module still loads if the FiLM stack is not on the
# path (nnU-Net imports every module in every external trainer directory).
# ---------------------------------------------------------------------------
try:
    from nnUNetTrainerFiLM import nnUNetTrainerFiLM

    class nnUNetTrainerFiLMBlob_250epochs(nnUNetTrainerFiLM):
        """FiLM chronicity conditioning + blob instance loss.

        FiLM improved detection on fold 0 (lesion-wise F1 0.605 vs 0.595, false
        positives 462 vs 524) while looking flat on Dice; blob loss attacks the
        same detection axis from the loss side. They are independent mechanisms,
        so stacking them is the natural combined method.
        """
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = 250

        def _build_loss(self):
            return _build_blob_loss(self, 2.0)

    class nnUNetTrainerFiLMTversky_250epochs(nnUNetTrainerFiLM):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = 250

        def _build_loss(self):
            return _build_tversky_loss(self, 0.3, 0.7)

except ImportError:  # FiLM stack not on sys.path for this invocation
    pass
