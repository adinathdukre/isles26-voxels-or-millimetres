#!/usr/bin/env python3
"""Search the 3-member ensemble weight simplex for the best submission model.

We have three members with all five folds: 500ep (carries Dice/Detection F1),
Dice+TopK10 (carries PR-AUC) and Dice+CE-x2 (improved every metric on fold 0).
Equal-weight 3-member was tested and lost to equal-weight 2-member; tuned
2-member (0.65/0.35) then beat both. The tuned 3-member case was never tried.

Two stages, so the search is affordable:
  1. coarse search on a random subset -- cheap, ranks the candidates
  2. the winner (and the incumbent) re-scored on all 1,453 -- the number we trust

Subsetting only ever decides *which* candidates to confirm; every reported figure
comes from the full cohort.
"""
import json
import os
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
import paths  # noqa: E402  -- roots come from ISLES_ROOT / ISLES_NNUNET_RESULTS / ISLES_GT

RESULTS = paths.RESULTS
GT = paths.GT
MEMBERS = [paths.MEMBER_500EP,
           paths.MEMBER_TOPK10,
           "nnUNetTrainerDiceCE_w2_250epochs" + paths.PLANS]
TAU, VMIN = 0.5, 30.0
CANDIDATES = [(0.65, 0.35, 0.00),   # incumbent, wired into the container
              (0.55, 0.30, 0.15),
              (0.50, 0.25, 0.25),
              (0.60, 0.20, 0.20),
              (0.45, 0.35, 0.20),
              (0.70, 0.15, 0.15)]
SUBSET = 350


def score(args):
    import SimpleITK as sitk
    import cc3d
    from eval_utils import (compute_dice_f1_instance_difference, compute_pr_auc,
                            compute_absolute_volume_difference)
    case, fold, weights = args
    g_img = sitk.ReadImage(os.path.join(GT, case + ".nii.gz"))
    g = sitk.GetArrayFromImage(g_img) > 0.5
    nu = float(np.prod(g_img.GetSpacing()))
    ml = np.float64(nu / 1000.0)
    maps = []
    for m in MEMBERS:
        with np.load(os.path.join(RESULTS, m, f"fold_{fold}", "validation", case + ".npz")) as z:
            a = z[z.files[0]]
        maps.append((a[1] if a.ndim == 4 else a).astype(np.float32))
    stack = np.stack(maps)

    out = {}
    for w in weights:
        arr = np.asarray(w, dtype=np.float32)
        arr = arr / arr.sum()
        prob = np.tensordot(arr, stack, axes=1).astype(np.float32)
        lab = cc3d.connected_components((prob >= TAU).astype(np.uint8), connectivity=26)
        n = int(lab.max())
        if n:
            sz = np.bincount(lab.ravel())
            keep = np.zeros(n + 1, bool)
            keep[[i for i in range(1, n + 1) if sz[i] >= VMIN / max(nu, 1e-6)]] = True
            pf = keep[lab]
        else:
            pf = prob >= TAU
        f1, dn, dice = compute_dice_f1_instance_difference(g, pf)
        out[w] = {"dice": float(dice), "f1": float(f1), "count_diff": int(dn),
                  "avd_ml": float(compute_absolute_volume_difference(g, pf, ml)),
                  "pr_auc": float(compute_pr_auc(g, prob))}
    return out


def run(cases, weights, workers):
    tasks = [(c, f, weights) for c, f in cases]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(score, tasks, chunksize=2))
    return {w: {k: float(np.mean([r[w][k] for r in res]))
                for k in ("dice", "f1", "count_diff", "avd_ml", "pr_auc")} for w in weights}


def show(agg, title):
    print(f"\n{title}")
    print(f"  {'weights (500/TopK/CE)':>24}{'Dice':>9}{'Det F1':>9}{'|dN|':>7}{'|dV|':>7}{'PR-AUC':>9}")
    for w, a in sorted(agg.items(), key=lambda kv: -kv[1]["f1"]):
        tag = "  <- incumbent" if w == (0.65, 0.35, 0.00) else ""
        print(f"  {str(w):>24}{a['dice']:9.4f}{a['f1']:9.4f}{a['count_diff']:7.2f}"
              f"{a['avd_ml']:7.2f}{a['pr_auc']:9.4f}{tag}")


def main():
    allc = []
    for f in range(5):
        vd = os.path.join(RESULTS, MEMBERS[0], f"fold_{f}", "validation")
        allc += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    rng = random.Random(0)
    sub = rng.sample(allc, SUBSET)

    print(f"stage 1: coarse search, {len(CANDIDATES)} candidates on {SUBSET} random cases", flush=True)
    agg = run(sub, CANDIDATES, 16)
    show(agg, "STAGE 1 (subset -- ranking only, not for reporting)")

    best = max(agg, key=lambda w: agg[w]["f1"])
    finals = list({best, (0.65, 0.35, 0.00)})
    print(f"\nstage 2: confirming {finals} on all {len(allc)} subjects", flush=True)
    full = run(allc, finals, 16)
    show(full, f"STAGE 2 (full cohort, n={len(allc)}) -- these are the numbers to trust")
    json.dump({str(k): v for k, v in full.items()},
              open(os.path.join(HERE, "best_ensemble.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
