"""
Wave-4: further loss and augmentation arms, chosen from what we MEASURED rather
than from what is fashionable.

Two measured facts drive the selection:

  1. Our decision-layer gain concentrated in LARGE lesions (Detection F1
     0.555 -> 0.628 in the >=10,000 mm^3 bin), which means the dominant
     instance error is spurious satellite fragments beside a correctly
     segmented infarct, and fragmentation of one lesion into several predicted
     components. Under one-to-one matching at IoU>=0.25 that is charged twice.
     => clDice, a connectivity-preserving loss, is the mechanistically
        motivated candidate. Every other loss we tried (blob, Tversky x2,
        focal Tversky, batch Dice) attacked size imbalance instead, and none
        of them beat the seed-noise floor.

  2. DA5 (MORE augmentation) was worse than baseline on every metric, and
     removing mirroring entirely was the worst arm of twelve. Both point at
     the same open question: is nnU-Net's DEFAULT augmentation already at or
     past the optimum for this dataset? That question is answered by arms with
     LESS augmentation, not more -- which is why NoDA and restricted mirroring
     are here.

Everything else is held fixed: fold 0, 250 epochs, ResEnc-M, same split.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5ord0
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import nnUNetTrainerNoDA
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import (
    nnUNetTrainerDiceCELoss_noSmooth,
)
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTopkLoss import (
    nnUNetTrainerDiceTopK10Loss,
)
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainer_onlyMirror01,
)


# ---------------------------------------------------------------------------
# clDice (Shit et al., CVPR 2021): centreline-Dice, a connectivity-preserving
# term. Built on a differentiable "soft skeleton" obtained by iterated
# morphological opening; a prediction that breaks one lesion into two pieces
# loses skeleton overlap even when its voxel-wise Dice is unchanged.
# ---------------------------------------------------------------------------
def _soft_erode(x):
    """Min-filter erosion as the minimum of three directional erosions.

    Written out rather than chained: `-a.min(b)` parses as `-(a.min(b))`, so the
    compact one-liner form silently erodes the wrong sign and yields an empty
    skeleton (clDice then returns exactly 0 for every input).
    """
    p1 = -F.max_pool3d(-x, (3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(x):
    return F.max_pool3d(x, (3, 3, 3), 1, (1, 1, 1))


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x, iters: int = 3):
    """Differentiable skeleton by iterated opening (Shit et al., CVPR 2021)."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        skel = skel + F.relu(x - x1) - skel * F.relu(x - x1)
    return skel


class SoftClDiceLoss(nn.Module):
    def __init__(self, iters: int = 3, smooth: float = 1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, net_output, target):
        probs = torch.softmax(net_output, 1)[:, 1:2].float()   # lesion channel
        if target.ndim == net_output.ndim and target.shape[1] == 1:
            tgt = (target == 1).float()
        else:
            tgt = (target.unsqueeze(1) == 1).float()

        sk_p = soft_skeletonize(probs, self.iters)
        sk_t = soft_skeletonize(tgt, self.iters)
        # topology precision / sensitivity
        tprec = ((sk_p * tgt).sum() + self.smooth) / (sk_p.sum() + self.smooth)
        tsens = ((sk_t * probs).sum() + self.smooth) / (sk_t.sum() + self.smooth)
        cl = 2.0 * tprec * tsens / (tprec + tsens)
        return 1.0 - cl


class _ClDiceCombined(nn.Module):
    """Dice+CE (unchanged) plus a clDice term at full resolution only."""

    def __init__(self, base, cldice_weight: float = 0.5):
        super().__init__()
        self.base = base
        self.cldice = SoftClDiceLoss()
        self.w = cldice_weight

    def forward(self, net_output, target):
        return self.base(net_output, target) + self.w * self.cldice(net_output, target)


class _ClDiceDSWrapper(nn.Module):
    """Deep supervision; the connectivity term only fires on the full-res head."""

    def __init__(self, top, plain, weights):
        super().__init__()
        self.top, self.plain, self.weights = top, plain, weights

    def forward(self, outputs, targets):
        total = self.weights[0] * self.top(outputs[0], targets[0])
        for w, o, t in zip(self.weights[1:], outputs[1:], targets[1:]):
            if w != 0:
                total = total + w * self.plain(o, t)
        return total


class nnUNetTrainerClDice_250epochs(nnUNetTrainer):
    CLDICE_WEIGHT = 0.5

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        def mk():
            return DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
                 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss)
        top = _ClDiceCombined(mk(), self.CLDICE_WEIGHT)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(scales))])
            w[-1] = 0
            return _ClDiceDSWrapper(top, mk(), w / w.sum())
        return top


# ---------------------------------------------------------------------------
# Loss rebalancing: nnU-Net weights Dice and CE equally. CE is the term that
# cares about every voxel individually, so up-weighting it is the cheapest
# probe of whether our soft map (and hence PR-AUC) is CE-starved.
# ---------------------------------------------------------------------------
class nnUNetTrainerDiceCE_w2_250epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
             'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=2, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(scales))])
            w[-1] = 0
            loss = DeepSupervisionWrapper(loss, w / w.sum())
        return loss


# ---------------------------------------------------------------------------
# 250-epoch versions of shipped variants, so they are comparable with the rest
# of the sweep.
# ---------------------------------------------------------------------------
class nnUNetTrainerDiceTopK10_250epochs(nnUNetTrainerDiceTopK10Loss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerDiceCEnoSmooth_250epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerNoDA_250epochs(nnUNetTrainerNoDA):
    """No augmentation at all -- the other end of the axis from DA5."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerDA5ord0_250epochs(nnUNetTrainerDA5ord0):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerOnlyMirror01_250epochs(nnUNetTrainer_onlyMirror01):
    """Mirror only the two in-plane axes.

    Removing mirroring entirely was the worst arm we tested, so the question is
    not whether to mirror but along which axes: left-right mirroring of a
    lateralised pathology is the one that is theoretically suspect.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
