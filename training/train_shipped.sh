#!/usr/bin/env bash
# Train the shipped ISLES'26 configuration: both ensemble members, five folds
# each, with --npz so the softmax maps exist for the decision layer and for
# ensembling afterwards.
#
#   member A   nnUNetTrainer_500epochs              stock nnU-Net    weight 0.65
#   member B   nnUNetTrainerDiceTopK10_250epochs    trainers/isles_wave4.py
#                                                                    weight 0.35
#
# This is a cleaned-up distillation of the per-fold job scripts the experiments
# actually ran under. Those waited on a specific PID and activated one user's
# conda environment on one machine; none of that is reproducible, so none of it
# survived. The nnUNetv2_train invocations are unchanged.
#
# Usage:
#   source training/env.sh
#   training/train_shipped.sh              # all five folds of both members
#   training/train_shipped.sh 0 2          # only folds 0 and 2
#
# Runtime: ~15 h/fold for the 500-epoch ResEnc-M member and roughly half that
# for the 250-epoch TopK10 member, on one card. Budget about 4-5 GPU-days for
# all ten trainings if you run them one at a time; the sweep ran them across
# eight shared GPUs with training/runner.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "$HERE/env.sh"

DATASET="${ISLES_DATASET_ID:-1}"
CONFIG=3d_fullres
PLANS=nnUNetResEncUNetMPlans
TRAINERS=(nnUNetTrainer_500epochs nnUNetTrainerDiceTopK10_250epochs)

FOLDS=("$@")
if [ "${#FOLDS[@]}" -eq 0 ]; then
    FOLDS=(0 1 2 3 4)
fi

echo "dataset=$DATASET plans=$PLANS config=$CONFIG folds=${FOLDS[*]}"
echo "nnUNet_results=$nnUNet_results"
echo "nnUNet_extTrainer=$nnUNet_extTrainer"

for tr in "${TRAINERS[@]}"; do
    for f in "${FOLDS[@]}"; do
        echo "=== $tr fold $f START $(date -u '+%Y-%m-%d %H:%M UTC') ==="
        nnUNetv2_train "$DATASET" "$CONFIG" "$f" -p "$PLANS" -tr "$tr" --npz
        echo "=== $tr fold $f DONE  $(date -u '+%Y-%m-%d %H:%M UTC') ==="
    done
done

echo "=== all requested folds done ==="
echo "Next: fit the decision layer (tau x minimum component size in mm^3) on the"
echo "pooled out-of-fold predictions -- see analysis/ -- then build the submission."
