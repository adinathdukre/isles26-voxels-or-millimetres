#!/usr/bin/env python3
"""Does filtering components by CONFIDENCE, not just size, beat size alone?

The shipped decision layer keeps a 26-connected component if its volume reaches
30 mm^3. That is a purely geometric rule: a large, confidently-wrong blob survives
it, and SS8g showed the residual error is dominated by detection, not delineation.
Lesion precision is 0.614 -- roughly a third of everything we emit is spurious.

A component carries more information than its size. Every voxel in it already
exceeds tau=0.5, but components differ enormously in how far above 0.5 they sit.
A true infarct tends to have a confident core; a false positive tends to hover
just over the threshold. So we test a second criterion alongside size:

    keep  iff  volume >= v_min  AND  stat(prob inside component) >= p

for stat in {mean, max}. p=0 reproduces the shipped rule exactly, which is the
control -- if the sweep cannot beat its own p=0 column, the idea is dead.

Only the MASK metrics can move. The soft map is submitted unfiltered by design,
so PR-AUC is invariant here and is not reported.

Scored at the shipped operating point (0.65/0.35 ensemble, tau=0.5, v_min=30 mm^3)
over the pooled out-of-fold cohort, so numbers are comparable to SS8e/SS8k.
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
A = paths.MEMBER_500EP
B = paths.MEMBER_TOPK10
W = (0.65, 0.35)
TAU, VMIN = 0.5, 30.0

# p = 0.0 is the shipped size-only rule -- the control column.
RULES = ([("mean", p) for p in (0.0, 0.60, 0.65, 0.70, 0.75, 0.80)] +
         [("max", p) for p in (0.70, 0.80, 0.90, 0.95, 0.99)])


def one(args):
    import SimpleITK as sitk
    import cc3d
    from eval_utils import (compute_dice_f1_instance_difference,
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
    prob = (W[0] * maps[0] + W[1] * maps[1]).astype(np.float32)

    lab = cc3d.connected_components((prob >= TAU).astype(np.uint8), connectivity=26)
    n = int(lab.max())
    min_vox = VMIN / max(nu, 1e-6)

    # Per-component size and confidence, computed once and reused for every rule.
    stats = {}
    if n:
        flat, pf_ = lab.ravel(), prob.ravel()
        sz = np.bincount(flat, minlength=n + 1).astype(np.float64)
        ssum = np.bincount(flat, weights=pf_, minlength=n + 1)
        smax = np.zeros(n + 1, np.float32)
        np.maximum.at(smax, flat, pf_)
        with np.errstate(invalid="ignore", divide="ignore"):
            smean = np.where(sz > 0, ssum / np.maximum(sz, 1), 0.0)
        stats = {"size": sz, "mean": smean, "max": smax}

    out = {}
    for stat, p in RULES:
        if n:
            keep = np.zeros(n + 1, bool)
            idx = np.arange(1, n + 1)
            ok = (stats["size"][idx] >= min_vox) & (stats[stat][idx] >= p)
            keep[idx[ok]] = True
            pred = keep[lab]
        else:
            pred = prob >= TAU
        f1, dn, dice = compute_dice_f1_instance_difference(g, pred)
        out[f"{stat}:{p}"] = {"dice": float(dice), "f1": float(f1),
                              "count_diff": int(dn),
                              "avd_ml": float(compute_absolute_volume_difference(g, pred, ml))}
    return out


def main():
    tasks = []
    for f in range(5):
        vd = os.path.join(RESULTS, A, f"fold_{f}", "validation")
        tasks += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"confidence-filter sweep: {len(RULES)} rules over {len(tasks)} out-of-fold cases",
          flush=True)
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(one, tasks, chunksize=4))

    agg = {k: {m: float(np.mean([r[k][m] for r in res]))
               for m in ("dice", "f1", "count_diff", "avd_ml")} for k in res[0]}
    base = agg["mean:0.0"]
    print(f"\n  shipped size-only control: Dice {base['dice']:.4f}  Det F1 {base['f1']:.4f}  "
          f"|dN| {base['count_diff']:.3f}  |dV| {base['avd_ml']:.3f}")
    print(f"\n  {'rule':>12}{'Dice':>9}{'Det F1':>9}{'|dN|':>8}{'|dV|':>8}"
          f"{'dDice':>9}{'dF1':>9}")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["f1"]):
        tag = "  <- shipped" if k == "mean:0.0" else ""
        print(f"  {k:>12}{a['dice']:9.4f}{a['f1']:9.4f}{a['count_diff']:8.3f}"
              f"{a['avd_ml']:8.3f}{a['dice']-base['dice']:+9.4f}{a['f1']-base['f1']:+9.4f}{tag}")
    print("\n  two-seed reference floors: Dice 0.0043, Det F1 0.0057")
    print("  a rule is only interesting if dF1 clears +0.0057 without losing Dice.")
    json.dump(agg, open(os.path.join(HERE, "confidence_filter.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
