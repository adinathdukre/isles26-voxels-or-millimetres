# Environment for ISLES'26 dataset conversion, training and the experiment sweep.
#
# Usage:  source training/env.sh
#
# Everything hangs off ISLES_ROOT. Set it yourself if the nnU-Net directories
# live somewhere other than this checkout; otherwise it defaults to the parent
# of this file, which is what a fresh clone wants. Every variable below is set
# with a :- default, so anything you already exported wins.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _isles_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    _isles_here="$(pwd)"
fi
export ISLES_ROOT="${ISLES_ROOT:-$_isles_here}"
unset _isles_here

# --- nnU-Net's own three directories -----------------------------------------
export nnUNet_raw="${nnUNet_raw:-$ISLES_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ISLES_ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$ISLES_ROOT/nnUNet_results}"

# --- what analysis/paths.py reads --------------------------------------------
# Exporting them here keeps the training and analysis halves of the repository
# pointing at the same files instead of drifting apart.
export ISLES_NNUNET_RESULTS="${ISLES_NNUNET_RESULTS:-$nnUNet_results/Dataset001_ISLES26}"
export ISLES_GT="${ISLES_GT:-$nnUNet_preprocessed/Dataset001_ISLES26/gt_segmentations}"

# --- custom trainers ----------------------------------------------------------
# nnU-Net (>= 2.8) discovers trainer classes outside its own package through
# this variable; see nnunetv2/utilities/find_objects.py. Colon-separated, so
# append your own directory here if you keep extra trainers elsewhere.
export nnUNet_extTrainer="${nnUNet_extTrainer:-$ISLES_ROOT/training/trainers}"

# nnU-Net puts the directory it is *currently scanning* on sys.path, but only
# that one. A trainer that imports a class out of a different trainer directory
# needs that directory importable the ordinary way as well, so the same list is
# prepended to PYTHONPATH. isles_wave2.py does this for its two
# metadata-conditioning arms; that import is wrapped in try/except, which means
# a missing PYTHONPATH does not crash anything -- it silently drops those arms,
# which is far more annoying to debug.
export PYTHONPATH="$ISLES_ROOT/training/trainers:${PYTHONPATH:-}"

# --- runtime knobs ------------------------------------------------------------
# torch.compile is a coin-flip on torch 2.12 + Blackwell and costs minutes of
# warmup per run; the sweep values wall-clock and reproducibility over the
# ~10% steady-state speedup, so it stays off.
export nnUNet_compile="${nnUNet_compile:-f}"

# 512 cores are shared with other users' jobs; 12 DA workers per training is
# nnU-Net's own default cap and keeps ~8 concurrent runs well inside RAM.
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"

# --- where the sweep keeps its scratch state ---------------------------------
# Job queue + per-job logs for training/runner.py. Pure runtime state; nothing
# in here belongs in version control.
export ISLES_SWEEP_DIR="${ISLES_SWEEP_DIR:-$ISLES_ROOT/sweep}"
