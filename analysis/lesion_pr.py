#!/usr/bin/env python3
"""Lesion-wise precision and recall, before and after the decision layer.

Detection F1 alone cannot say whether filtering helps by removing false-positive
components or hurts by deleting true small lesions. This pools TP/FP/FN over the
out-of-fold cohort at two settings -- no size filter, and the shipped 30 mm^3
rule -- and also counts how many genuinely matched lesions the filter destroys.

Deliberately separate from official_score.py: it calls Panoptica with exactly the
settings in the organizers' eval_utils (ConnectedComponentsInstanceApproximator +
NaiveThresholdMatching at IoU 0.25) but leaves that file untouched.
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
import paths  # noqa: E402  -- roots come from ISLES_ROOT / ISLES_NNUNET_RESULTS / ISLES_GT

RESULTS = paths.RESULTS
GT_DIR = paths.GT
# The 250-epoch baseline is the run this decomposition was reported on; point
# ISLES_RUN at any other results folder to redo it for a different model.
RUN = os.environ.get("ISLES_RUN", "nnUNetTrainer_250epochs" + paths.PLANS)
TAU, MIN_MM3 = 0.5, 30.0


def _init():
    warnings.filterwarnings("ignore")


def _counts(args):
    import SimpleITK as sitk
    import cc3d
    from panoptica import (Panoptica_Evaluator, InputType,
                           ConnectedComponentsInstanceApproximator,
                           NaiveThresholdMatching)
    case, fold = args
    vd = os.path.join(RESULTS, RUN, f"fold_{fold}", "validation")
    g_img = sitk.ReadImage(os.path.join(GT_DIR, case + ".nii.gz"))
    g = (sitk.GetArrayFromImage(g_img) > 0.5).astype(int)
    nu = float(np.prod(g_img.GetSpacing()))

    with np.load(os.path.join(vd, case + ".npz")) as z:
        arr = z["probabilities" if "probabilities" in z else z.files[0]]
    prob = (arr[1] if arr.ndim == 4 else arr).astype(np.float32)

    raw = (prob >= TAU).astype(np.uint8)
    lab = cc3d.connected_components(raw, connectivity=26)
    n = int(lab.max())
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(n + 1, bool)
        keep[[i for i in range(1, n + 1) if sizes[i] >= MIN_MM3 / max(nu, 1e-6)]] = True
        filt = keep[lab].astype(int)
    else:
        filt = raw.astype(int)

    ev = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.25))
    out = {}
    for tag, pred in (("raw", raw.astype(int)), ("filt", filt)):
        if g.sum() == 0 and pred.sum() == 0:
            out[tag] = (0, 0, 0)
            continue
        r = ev.evaluate(pred, g)["ungrouped"]
        out[tag] = (int(r.tp), int(r.fp), int(r.fn))
    return case, out


def main():
    tasks = []
    for f in range(5):
        vd = os.path.join(RESULTS, RUN, f"fold_{f}", "validation")
        tasks += [(fn[:-7], f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"scoring {len(tasks)} out-of-fold cases", flush=True)
    with ProcessPoolExecutor(max_workers=32, initializer=_init) as ex:
        res = dict(ex.map(_counts, tasks, chunksize=2))

    agg = {}
    for tag in ("raw", "filt"):
        tp = sum(res[c][tag][0] for c in res)
        fp = sum(res[c][tag][1] for c in res)
        fn = sum(res[c][tag][2] for c in res)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        agg[tag] = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                        f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        print(f"  {tag:5s} TP={tp:6d} FP={fp:6d} FN={fn:6d}  "
              f"precision={prec:.4f} recall={rec:.4f}")
    # true lesions the filter destroyed: matched before, unmatched after
    lost = sum(max(0, res[c]["raw"][0] - res[c]["filt"][0]) for c in res)
    fp_removed = sum(max(0, res[c]["raw"][1] - res[c]["filt"][1]) for c in res)
    agg["true_lesions_lost"] = lost
    agg["false_positives_removed"] = fp_removed
    agg["n_cases"] = len(res)
    print(f"\n  filter removed {fp_removed} false-positive components "
          f"and destroyed {lost} correctly matched lesions")
    json.dump(agg, open(os.path.join(HERE, "lesion_pr.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
