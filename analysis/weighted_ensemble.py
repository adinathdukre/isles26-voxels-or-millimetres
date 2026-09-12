#!/usr/bin/env python3
"""Does an unequal ensemble weighting beat the 50/50 average we ship?

The shipped container averages 500ep and TopK10 with equal weight. That was never
tuned. The two members are not equally good: 500ep leads on Dice and Detection F1,
TopK10 contributes the PR-AUC. A weight favouring 500ep should trace a different
point on the same trade-off, and the rank objective may prefer it.

Scored at the shipped operating point (tau=0.5, v_min=30 mm^3) over the pooled
out-of-fold cohort, so results are directly comparable to SS8e.
"""
import json
import os
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
A = paths.MEMBER_500EP    # stronger member
B = paths.MEMBER_TOPK10   # PR-AUC contributor
WEIGHTS = [0.35, 0.50, 0.65, 0.80]           # weight on A
TAU, VMIN = 0.5, 30.0


def one(args):
    import SimpleITK as sitk
    import cc3d
    from eval_utils import (compute_dice_f1_instance_difference, compute_pr_auc,
                            compute_absolute_volume_difference)
    case, fold = args
    g_img = sitk.ReadImage(os.path.join(GT, case + ".nii.gz"))
    g = sitk.GetArrayFromImage(g_img) > 0.5
    nu = float(np.prod(g_img.GetSpacing()))
    ml = np.float64(nu / 1000.0)
    maps = []
    for m in (A, B):
        with np.load(os.path.join(RESULTS, m, f"fold_{fold}", "validation", case + ".npz")) as z:
            a = z[z.files[0]]
        maps.append((a[1] if a.ndim == 4 else a).astype(np.float32))

    out = {}
    for w in WEIGHTS:
        prob = (w * maps[0] + (1 - w) * maps[1]).astype(np.float32)
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


def main():
    tasks = []
    for f in range(5):
        vd = os.path.join(RESULTS, A, f"fold_{f}", "validation")
        tasks += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"weighting sweep over {len(tasks)} out-of-fold cases, weights {WEIGHTS}", flush=True)
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(one, tasks, chunksize=4))

    print(f"\n{'weight on 500ep':>16}{'Dice':>9}{'Det F1':>9}{'|dN|':>8}{'|dV|':>8}{'PR-AUC':>9}")
    agg = {}
    for w in WEIGHTS:
        agg[w] = {k: float(np.mean([r[w][k] for r in res]))
                  for k in ("dice", "f1", "count_diff", "avd_ml", "pr_auc")}
        tag = "  <- shipped" if w == 0.50 else ""
        a = agg[w]
        print(f"{w:>16.2f}{a['dice']:9.4f}{a['f1']:9.4f}{a['count_diff']:8.2f}"
              f"{a['avd_ml']:8.2f}{a['pr_auc']:9.4f}{tag}")

    base = agg[0.50]
    print("\n  change vs the shipped 50/50 (references: Dice .0043 F1 .0057 AUC .0034):")
    for w in WEIGHTS:
        if w == 0.50:
            continue
        d = {k: agg[w][k] - base[k] for k in base}
        print(f"    w={w:.2f}  Dice {d['dice']:+.4f}  F1 {d['f1']:+.4f}  "
              f"|dN| {d['count_diff']:+.3f}  |dV| {d['avd_ml']:+.3f}  AUC {d['pr_auc']:+.4f}")
    json.dump({str(k): v for k, v in agg.items()},
              open(os.path.join(HERE, "weighted_ensemble.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
