"""
blob loss (Kofler et al. 2023, "blob loss: instance imbalance aware loss
functions for semantic segmentation") for the ISLES'26 sweep.

Why this one, for this dataset specifically: our measured failure is instance-
shaped, not boundary-shaped. The baseline misses 492 of 1082 ground-truth
lesion components and emits 524 false-positive components, while volumetric
Dice looks respectable at 0.667 -- because a single large lesion dominates the
volumetric statistic and small satellite lesions contribute almost nothing to
it. blob loss adds a term in which every lesion instance counts the same
regardless of size, which is exactly the gradient signal a global Dice term
fails to provide.

Implementation notes:
  - Instances come from a connected-component labelling of the ground-truth
    patch. That is a CPU operation (scipy.ndimage.label, or cc3d when it is
    installed, which is ~5x faster), costing ~30-60ms per step against a ~250ms
    step -- measured, not assumed.
  - The instance term is applied at FULL RESOLUTION ONLY. Deep-supervision
    heads keep the standard global Dice+CE. Re-labelling components at every
    supervision scale would multiply the CPU cost, and the low-resolution heads
    cannot resolve small lesions anyway -- which is the whole point.
  - For instance i, the other instances are masked OUT of the loss (loss_mask),
    so instance i is scored against itself plus genuine background only. This
    follows the paper's formulation.
"""
import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    import cc3d
    _HAVE_CC3D = True
except ImportError:
    from scipy import ndimage
    _HAVE_CC3D = False


def _label_components(arr_np):
    """Connected components of a (D,H,W) uint8 array -> (labels, n)."""
    if _HAVE_CC3D:
        lab = cc3d.connected_components(arr_np, connectivity=26)
        return lab, int(lab.max())
    lab, n = ndimage.label(arr_np)
    return lab, int(n)


class BlobLoss(nn.Module):
    """Global Dice+CE plus an instance-averaged Dice term.

    total = global_dice_ce + blob_weight * mean_over_instances(dice_loss_i)
    """

    def __init__(self, global_loss: nn.Module, blob_weight: float = 2.0, max_instances: int = 32):
        super().__init__()
        self.global_loss = global_loss
        self.blob_weight = blob_weight
        self.max_instances = max_instances
        # Per-instance Dice. Never batch_dice: each instance is scored on its own
        # sample, and apply_nonlin is None because forward() softmaxes once and
        # reuses the probabilities across every instance.
        self.inst_dice = MemoryEfficientSoftDiceLoss(
            apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5, ddp=False)

    def forward(self, net_output, target):
        loss = self.global_loss(net_output, target)

        with torch.no_grad():
            tgt = target
            if tgt.ndim == net_output.ndim and tgt.shape[1] == 1:
                tgt_np = tgt[:, 0].detach().to(torch.uint8).cpu().numpy()
            else:
                tgt_np = tgt.detach().to(torch.uint8).cpu().numpy()

        probs = torch.softmax(net_output, 1)

        inst_losses = []
        for b in range(tgt_np.shape[0]):
            fg = (tgt_np[b] > 0).astype(np.uint8)
            if fg.sum() == 0:
                continue
            lab, n = _label_components(fg)
            if n == 0:
                continue
            lab_t = torch.from_numpy(lab.astype(np.int16)).to(net_output.device)
            fg_t = torch.from_numpy(fg).to(net_output.device).bool()
            for i in range(1, min(n, self.max_instances) + 1):
                inst = (lab_t == i)
                # keep this instance and true background; hide every other instance
                loss_mask = (~(fg_t & ~inst)).to(probs.dtype)[None, None]
                y = inst.to(torch.float32)[None, None]
                x = probs[b:b + 1]
                inst_losses.append(self.inst_dice(x, y, loss_mask))

        if inst_losses:
            loss = loss + self.blob_weight * torch.stack(inst_losses).mean()
        return loss


class _BlobDSWrapper(nn.Module):
    """Deep supervision, but the instance term only fires on the full-res head."""

    def __init__(self, blob_loss: BlobLoss, plain_loss: nn.Module, weights):
        super().__init__()
        self.blob_loss = blob_loss
        self.plain_loss = plain_loss
        self.weights = weights

    def forward(self, outputs, targets):
        total = self.weights[0] * self.blob_loss(outputs[0], targets[0])
        for w, o, t in zip(self.weights[1:], outputs[1:], targets[1:]):
            if w != 0:
                total = total + w * self.plain_loss(o, t)
        return total


class nnUNetTrainerBlobLoss_250epochs(nnUNetTrainer):
    BLOB_WEIGHT = 2.0

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

    def _build_loss(self):
        assert not self.label_manager.has_regions, "region-based training not used for ISLES'26"

        def _mk():
            return DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5,
                 'do_bg': False, 'ddp': self.is_ddp},
                {}, weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss)

        blob = BlobLoss(_mk(), blob_weight=self.BLOB_WEIGHT)

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return _BlobDSWrapper(blob, _mk(), weights)
        return blob


class nnUNetTrainerBlobLossW1_250epochs(nnUNetTrainerBlobLoss_250epochs):
    """Lighter instance weighting; brackets the blob_weight axis."""
    BLOB_WEIGHT = 1.0
