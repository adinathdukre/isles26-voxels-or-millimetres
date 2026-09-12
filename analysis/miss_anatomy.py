#!/usr/bin/env python3
"""Characterise the lesions the model misses -- the dominant residual error.

v2: also records the case id and its native voxel volume, to test whether the
1 mm resampling in preprocessing is what blinds the model to small lesions.

Recall is 0.408 and SS8f showed no operating point recovers it, which implies the
network never proposes those lesions. That claim was inferred, not measured. This
measures it: for every ground-truth lesion, decide whether it was detected (IoU
>= 0.25 against a predicted component, the challenge's rule) and, for the missed
ones, report

  * their size distribution -- are misses small, or is size not the story?
  * the model's peak probability inside the missed lesion.

The second number is decisive. If peak probability is near zero the network is
blind there and only better training/features can help. If it is appreciable but
below tau, the lesion IS proposed and something cheaper (denser sliding windows,
calibration, a smarter decision rule) could recover it.
"""
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
import paths  # noqa: E402  -- roots come from ISLES_ROOT / ISLES_NNUNET_RESULTS / ISLES_GT

RESULTS = paths.RESULTS
GT = paths.GT
RUN = paths.MEMBER_500EP
TAU, VMIN = 0.5, 30.0


def analyse(args):
    import SimpleITK as sitk
    import cc3d
    case, fold = args
    vd = os.path.join(RESULTS, RUN, f"fold_{fold}", "validation")
    g_img = sitk.ReadImage(os.path.join(GT, case + ".nii.gz"))
    g = sitk.GetArrayFromImage(g_img) > 0.5
    nu = float(np.prod(g_img.GetSpacing()))
    if not g.any():
        return []
    with np.load(os.path.join(vd, case + ".npz")) as z:
        arr = z[z.files[0]]
    prob = (arr[1] if arr.ndim == 4 else arr).astype(np.float32)
    if prob.shape != g.shape:
        return []

    pred = prob >= TAU
    plab = cc3d.connected_components(pred.astype(np.uint8), connectivity=26)
    psz = np.bincount(plab.ravel())
    keep = np.zeros(len(psz), bool)
    keep[[i for i in range(1, len(psz)) if psz[i] >= VMIN / max(nu, 1e-6)]] = True
    pred_f = keep[plab]
    plab_f = plab * pred_f

    glab = cc3d.connected_components(g.astype(np.uint8), connectivity=26)
    out = []
    for i in range(1, int(glab.max()) + 1):
        m = glab == i
        vol = float(m.sum()) * nu
        # challenge rule: detected if some predicted component reaches IoU >= 0.25
        best = 0.0
        for j in np.unique(plab_f[m]):
            if j == 0:
                continue
            pj = plab_f == j
            inter = np.logical_and(m, pj).sum()
            union = np.logical_or(m, pj).sum()
            best = max(best, inter / union if union else 0.0)
        out.append({"case": case, "nu": nu, "vol_mm3": vol, "iou": float(best),
                    "detected": bool(best >= 0.25),
                    "peak_prob": float(prob[m].max()),
                    "mean_prob": float(prob[m].mean())})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor
    tasks = []
    for f in range(5):
        vd = os.path.join(RESULTS, RUN, f"fold_{f}", "validation")
        tasks += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"analysing ground-truth lesions across {len(tasks)} subjects ...", flush=True)
    with ProcessPoolExecutor(max_workers=28) as ex:
        res = [l for sub in ex.map(analyse, tasks, chunksize=2) for l in sub]

    det = [l for l in res if l["detected"]]
    mis = [l for l in res if not l["detected"]]
    print(f"\n  ground-truth lesions: {len(res)}   detected {len(det)} "
          f"({100*len(det)/len(res):.1f}%)   MISSED {len(mis)} ({100*len(mis)/len(res):.1f}%)\n")

    bins = [(0, 30, "<30 mm3 (below filter)"), (30, 100, "30-100"), (100, 500, "100-500"),
            (500, 2000, "500-2k"), (2000, 10000, "2k-10k"), (10000, 1e12, ">=10k")]
    print(f"  {'GT lesion size':24s}{'n':>7}{'missed':>9}{'miss %':>9}")
    for lo, hi, lab in bins:
        sel = [l for l in res if lo <= l["vol_mm3"] < hi]
        if not sel:
            continue
        m = sum(1 for l in sel if not l["detected"])
        print(f"  {lab:24s}{len(sel):7d}{m:9d}{100*m/len(sel):8.1f}%")

    print("\n  DOES THE MODEL SEE THE MISSED LESIONS AT ALL?")
    for lab, sel in (("missed", mis), ("detected", det)):
        if not sel:
            continue
        pk = np.array([l["peak_prob"] for l in sel])
        print(f"    {lab:9s} peak prob inside lesion: "
              f"median {np.median(pk):.3f}   mean {pk.mean():.3f}   "
              f"frac >0.5 {100*(pk>0.5).mean():5.1f}%   frac <0.05 {100*(pk<0.05).mean():5.1f}%")
    if mis:
        pk = np.array([l["peak_prob"] for l in mis])
        vol = np.array([l["vol_mm3"] for l in mis])
        rec = ((pk > 0.5) & (vol >= 30)).sum()
        print(f"\n    missed lesions that DO reach prob>0.5 and are >=30 mm3: {rec} "
              f"({100*rec/len(mis):.1f}% of misses)")
        print("      -> these are proposed but lost to component matching, not blindness")
    json.dump(res, open(os.path.join(HERE, "miss_anatomy.json"), "w"))


if __name__ == "__main__":
    main()
