"""
Primus (Wald et al. 2025, arXiv:2503.01835) for the ISLES'26 sweep -- the
nnU-Net team's own transformer architecture, shipped inside
dynamic_network_architectures, so no extra dependency.

Why it needs a custom trainer rather than a plans edit:
  * Primus takes `output_channels`, not nnU-Net's `num_classes`, and needs
    `input_shape` (the patch size) and `patch_embed_size` at construction.
  * It produces a SINGLE full-resolution output -- there are no deep-supervision
    heads -- so the trainer must subclass nnUNetTrainerNoDeepSupervision.
  * It is a plain ViT-style stack. nnU-Net's default SGD(0.01, nesterov) is
    tuned for convnets and destabilises transformers; the Primus paper uses
    AdamW. We follow the paper here rather than the framework default, because
    training a transformer with the convnet recipe would test the recipe, not
    the architecture.

Everything else is held fixed against the rest of the sweep: same fold, same
split, same preprocessed data, same 250-epoch budget, same patch size.
"""
import torch
from torch import nn

from dynamic_network_architectures.architectures.primus import PrimusB, PrimusM, PrimusS
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)

# 8^3 voxels per token: with nnU-Net's 128^3 patch that is a 16^3 = 4096-token
# sequence, the configuration the Primus paper uses for this patch size.
PATCH_EMBED_SIZE = (8, 8, 8)


class _PrimusTrainerBase(nnUNetTrainerNoDeepSupervision):
    PRIMUS_CLS = PrimusM
    INITIAL_LR = 3e-4          # Primus paper; nnU-Net's 1e-2 SGD value diverges here
    WEIGHT_DECAY = 5e-2
    EPOCHS = 250

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.EPOCHS
        self.initial_lr = self.INITIAL_LR
        self.weight_decay = self.WEIGHT_DECAY

    @staticmethod
    def _build(cls, plans_manager, configuration_manager, num_input_channels, num_output_channels):
        patch_size = tuple(configuration_manager.patch_size)
        for p, e in zip(patch_size, PATCH_EMBED_SIZE):
            assert p % e == 0, f"patch size {patch_size} not divisible by embed size {PATCH_EMBED_SIZE}"
        return cls(input_channels=num_input_channels,
                   output_channels=num_output_channels,
                   patch_embed_size=PATCH_EMBED_SIZE,
                   input_shape=patch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr,
                                      weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        """No-op.

        nnUNetTrainer.set_deep_supervision_enabled() does `mod.decoder.deep_supervision = x`,
        which assumes a U-Net-shaped network. Primus is an encoder + patch-decode
        head with no `.decoder` attribute and a single output, so the base
        implementation raises AttributeError before the first epoch. (The
        ConnectionResetError that surfaces alongside it is just the dataloader
        threads tearing down afterwards -- a red herring.)
        """
        return


class nnUNetTrainerPrimusM_250epochs(_PrimusTrainerBase):
    PRIMUS_CLS = PrimusM

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = False):
        return _PrimusTrainerBase._build(PrimusM, plans_manager, configuration_manager,
                                         num_input_channels, num_output_channels)


class nnUNetTrainerPrimusB_250epochs(_PrimusTrainerBase):
    PRIMUS_CLS = PrimusB

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = False):
        return _PrimusTrainerBase._build(PrimusB, plans_manager, configuration_manager,
                                         num_input_channels, num_output_channels)


class nnUNetTrainerPrimusS_250epochs(_PrimusTrainerBase):
    PRIMUS_CLS = PrimusS

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = False):
        return _PrimusTrainerBase._build(PrimusS, plans_manager, configuration_manager,
                                         num_input_channels, num_output_channels)
