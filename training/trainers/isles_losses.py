"""
Loss functions for the ISLES'26 sweep.

Each class matches the constructor signature nnU-Net's compound losses expect
for their `dice_class` argument
(apply_nonlin, batch_dice, do_bg, smooth, ddp), so they can be dropped straight
into DC_and_CE_loss without touching the trainer's deep-supervision wiring.

Sign convention follows nnU-Net: lower is better, and the Dice-family terms
return a negative number (-dice). Focal Tversky returns (1-TI)^gamma, which is
positive but still lower-is-better -- the two differ only by a constant offset
in the logged loss value, not in the gradient.

Motivation: the baseline's failure is recall-shaped, not delineation-shaped --
Detection F1 0.533 against Dice 0.645, with Dice 0.234 on <100-voxel lesions and
17/291 complete misses -- so the losses here all trade precision for recall in a
controlled way (Tversky beta > alpha) or re-weight hard/small examples (focal,
blob).
"""
from typing import Callable

import torch
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Soft Tversky index, computed like nnU-Net's MemoryEfficientSoftDiceLoss.

    TI = TP / (TP + alpha*FP + beta*FN).  alpha=beta=0.5 recovers Dice.
    alpha < beta penalizes false negatives harder -> higher recall on small
    lesions, which is the failure mode we measured.
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def _tversky(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        fp = sum_pred - intersect
        fn = sum_gt - intersect
        return (intersect + self.smooth) / (
            intersect + self.alpha * fp + self.beta * fn + float(self.smooth)).clamp_min(1e-8)

    def forward(self, x, y, loss_mask=None):
        return -self._tversky(x, y, loss_mask).mean()


class MemoryEfficientFocalTverskyLoss(MemoryEfficientSoftTverskyLoss):
    """Focal Tversky loss: (1 - TI)^gamma.

    gamma < 1 concentrates gradient on examples that are already poorly
    segmented (Abraham & Khan 2019 use 3/4). Clamped before the power so the
    gradient stays finite as TI -> 1.
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75):
        super().__init__(apply_nonlin, batch_dice, do_bg, smooth, ddp, alpha, beta)
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        ti = self._tversky(x, y, loss_mask)
        return ((1.0 - ti).clamp_min(1e-6) ** self.gamma).mean()


def tversky_factory(alpha: float, beta: float):
    """Build a dice_class usable by DC_and_CE_loss with fixed alpha/beta."""
    class _Tversky(MemoryEfficientSoftTverskyLoss):
        def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., ddp=True):
            super().__init__(apply_nonlin, batch_dice, do_bg, smooth, ddp, alpha=alpha, beta=beta)
    _Tversky.__name__ = f"Tversky_a{alpha}_b{beta}".replace(".", "")
    return _Tversky


def focal_tversky_factory(alpha: float, beta: float, gamma: float):
    class _FocalTversky(MemoryEfficientFocalTverskyLoss):
        def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., ddp=True):
            super().__init__(apply_nonlin, batch_dice, do_bg, smooth, ddp,
                             alpha=alpha, beta=beta, gamma=gamma)
    _FocalTversky.__name__ = f"FocalTversky_a{alpha}_b{beta}_g{gamma}".replace(".", "")
    return _FocalTversky
