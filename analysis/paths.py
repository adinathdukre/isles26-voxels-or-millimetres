"""Single source of truth for filesystem locations.

Every scorer under analysis/ that touches a volume, a weight or a ground truth
reads its roots from here, so nothing is hardcoded to the machine the
experiments happened to run on. (site_transfer.py reads only the per-case cache
and so imports nothing from this module.) Override with environment variables:

    export ISLES_ROOT=/path/to/your/checkout        # this repository
    export ISLES_NNUNET_RESULTS=/path/to/results    # nnUNet_results/Dataset001_ISLES26
    export ISLES_GT=/path/to/gt_segmentations       # reference segmentations

Defaults assume the nnU-Net layout described in training/README.md.
"""
import os

ROOT = os.environ.get("ISLES_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = "Dataset001_ISLES26"
RESULTS = os.environ.get("ISLES_NNUNET_RESULTS", os.path.join(ROOT, "nnUNet_results", DATASET))
GT = os.environ.get("ISLES_GT", os.path.join(ROOT, "nnUNet_preprocessed", DATASET, "gt_segmentations"))

# Planner suffix shared by every trained configuration in the paper.
PLANS = "__nnUNetResEncUNetMPlans__3d_fullres"
MEMBER_500EP = "nnUNetTrainer_500epochs" + PLANS
MEMBER_TOPK10 = "nnUNetTrainerDiceTopK10_250epochs" + PLANS
