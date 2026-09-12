#!/usr/bin/env python3
"""Should we emit a constant soft map when we predict "no lesion"?

The organisers' scorer returns PR-AUC = 1.0 for an empty ground truth if the soft
map is perfectly constant, and ~0 if it varies at all. So emitting zeros whenever
our binary mask is empty is worth up to a full point per empty-GT case -- but it
destroys real ranking information on any case where we wrongly predict empty.

`EMIT_CONSTANT_MAP_WHEN_EMPTY` is currently False in the container. This measures
the bet directly on the pooled out-of-fold cohort: how often we predict empty, how
often that is correct, and the net PR-AUC change.
"""
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
MEMBERS = [paths.MEMBER_500EP, paths.MEMBER_TOPK10]
TAU, VMIN = 0.5, 30.0


def one(args):
    import SimpleITK as sitk
    import cc3d
    from eval_utils import compute_pr_auc
    case, fold = args
    g_img = sitk.ReadImage(os.path.join(GT, case + ".nii.gz"))
    g = sitk.GetArrayFromImage(g_img) > 0.5
    nu = float(np.prod(g_img.GetSpacing()))
    maps = []
    for m in MEMBERS:
        f = os.path.join(RESULTS, m, f"fold_{fold}", "validation", case + ".npz")
        with np.load(f) as z:
            a = z[z.files[0]]
        maps.append((a[1] if a.ndim == 4 else a).astype(np.float32))
    prob = np.mean(maps, axis=0, dtype=np.float32)

    lab = cc3d.connected_components((prob >= TAU).astype(np.uint8), connectivity=26)
    n = int(lab.max())
    if n:
        sz = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[[i for i in range(1, n + 1) if sz[i] >= VMIN / max(nu, 1e-6)]] = True
        pf = keep[lab]
    else:
        pf = prob >= TAU

    pred_empty = not pf.any()
    gt_empty = not g.any()
    naive = float(compute_pr_auc(g, prob))
    const = float(compute_pr_auc(g, np.zeros_like(prob))) if pred_empty else naive
    return pred_empty, gt_empty, naive, const


def main():
    tasks = []
    for f in range(5):
        vd = os.path.join(RESULTS, MEMBERS[0], f"fold_{f}", "validation")
        tasks += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"evaluating the empty-map bet over {len(tasks)} out-of-fold cases ...", flush=True)
    with ProcessPoolExecutor(max_workers=28) as ex:
        res = list(ex.map(one, tasks, chunksize=2))

    pe = np.array([r[0] for r in res])
    ge = np.array([r[1] for r in res])
    naive = np.array([r[2] for r in res])
    const = np.array([r[3] for r in res])

    print(f"\n  cases where we predict EMPTY      : {pe.sum()} / {len(res)} ({100*pe.mean():.2f}%)")
    print(f"  of those, ground truth also empty : {(pe & ge).sum()}  -> trick WINS (PR-AUC ~1.0)")
    print(f"  of those, ground truth NOT empty  : {(pe & ~ge).sum()}  -> trick LOSES real ranking info")
    print(f"  cases with empty GT overall       : {ge.sum()}")
    if (pe & ~ge).any():
        lost = naive[pe & ~ge]
        print(f"    PR-AUC we would throw away on those: mean {lost.mean():.4f}")
    if (pe & ge).any():
        gain = const[pe & ge] - naive[pe & ge]
        print(f"    PR-AUC gained on correct empties  : mean {gain.mean():+.4f}")
    print(f"\n  pooled PR-AUC without trick : {naive.mean():.4f}")
    print(f"  pooled PR-AUC with trick    : {const.mean():.4f}   ({const.mean()-naive.mean():+.4f})")
    print(f"\n  two-seed PR-AUC reference   : 0.0034")


if __name__ == "__main__":
    main()
