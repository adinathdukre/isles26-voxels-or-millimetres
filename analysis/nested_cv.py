#!/usr/bin/env python3
"""Nested cross-validation of the decision layer, per reviewer request.

The submitted paper fits (tau, v_min) on all 1,453 pooled out-of-fold predictions
and reports on the same pool. Predictions are leakage-free but the *operating
point* is not independently evaluated, so the reported gain may carry selection
optimism -- a limitation the paper discloses but does not quantify.

This quantifies it. For each fold k: fit (tau, v_min) on the other four folds by
the challenge's rank-then-aggregate objective, apply that setting to fold k, and
pool the five held-out results. No fold contributes to choosing the threshold it
is then scored under.

Also reports Detection F1 stratified by reference lesion size, and how many
genuine lesions the fixed 30 mm^3 rule destroys per size bin -- the reviewer's
question about systematically removing small infarcts.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

CACHE = os.path.join(HERE, "nested_cv_percase.json")
RUN = paths.MEMBER_500EP
TAUS = [0.5, 0.6, 0.7]
VMINS = [0, 10, 20, 30, 40, 60, 100]


def build_cache():
    """Score every case at every (tau, v_min); cache per-case metrics."""
    import official_score as osc
    from concurrent.futures import ProcessPoolExecutor

    cfg = {"thresholds": TAUS, "min_sizes": VMINS, "pmax_cuts": [0.0], "mm3": True}
    tasks = []
    for f in range(5):
        vd = os.path.join(osc.RESULTS, RUN, f"fold_{f}", "validation")
        tasks += [(fn[:-7], RUN, f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    print(f"scoring {len(tasks)} cases x {len(TAUS)*len(VMINS)} settings ...", flush=True)
    with ProcessPoolExecutor(max_workers=28, initializer=osc._init, initargs=(cfg,)) as ex:
        res = dict(ex.map(osc._score_case, tasks, chunksize=2))
    fold_of = {t[0]: t[2] for t in tasks}
    out = {c: {"fold": fold_of[c], "gt_vox": v["gt_vox"],
               "m": {k: {kk: vv for kk, vv in x.items() if kk in ("dice", "f1", "count_diff", "avd_ml", "pr_auc")}
                     for k, x in v.items() if isinstance(x, dict)}}
           for c, v in res.items()}
    json.dump(out, open(CACHE, "w"))
    return out


def mean_rank_fit(cases, data, variants):
    """Pick the variant with the best mean per-case rank across the 5 metrics."""
    from scipy.stats import rankdata
    agg = {v: [] for v in variants}
    for c in cases:
        cols = []
        for v in variants:
            r = data[c]["m"][v]
            cols.append([r["dice"], -r["avd_ml"], r["f1"], -r["count_diff"], r.get("pr_auc", 0.0)])
        arr = np.array(cols, float)
        ranks = np.zeros(len(variants))
        for m in range(arr.shape[1]):
            ranks += rankdata(-arr[:, m], method="average")
        ranks /= arr.shape[1]
        for v, rk in zip(variants, ranks):
            agg[v].append(rk)
    return min(agg, key=lambda v: float(np.mean(agg[v])))


def main():
    data = json.load(open(CACHE)) if os.path.exists(CACHE) else build_cache()
    variants = [f"{t}|{s}|0.0" for t in TAUS for s in VMINS]
    by_fold = defaultdict(list)
    for c, v in data.items():
        by_fold[v["fold"]].append(c)

    # ---- nested CV ------------------------------------------------------
    picks, held = {}, []
    for k in range(5):
        train = [c for f in range(5) if f != k for c in by_fold[f]]
        best = mean_rank_fit(train, data, variants)
        picks[k] = best
        held += [(c, best) for c in by_fold[k]]
    pooled_nested = {m: float(np.mean([data[c]["m"][v][m] for c, v in held]))
                     for m in ("dice", "f1", "count_diff", "avd_ml")}

    fixed = "0.5|30|0.0"
    pooled_fixed = {m: float(np.mean([data[c]["m"][fixed][m] for c in data]))
                    for m in ("dice", "f1", "count_diff", "avd_ml")}
    nofilt = "0.5|0|0.0"
    pooled_nofilt = {m: float(np.mean([data[c]["m"][nofilt][m] for c in data]))
                     for m in ("dice", "f1", "count_diff", "avd_ml")}

    print("\nNESTED CROSS-VALIDATION OF THE DECISION LAYER (n=1453)\n")
    print("  per-fold threshold chosen without seeing the evaluated fold:")
    for k in range(5):
        print(f"    fold {k}: {picks[k]}")
    print(f"\n{'setting':38s}{'Dice':>9}{'Det F1':>9}{'|dN|':>8}{'|dV|':>8}")
    print(f"{'no filtering':38s}{pooled_nofilt['dice']:9.4f}{pooled_nofilt['f1']:9.4f}"
          f"{pooled_nofilt['count_diff']:8.2f}{pooled_nofilt['avd_ml']:8.2f}")
    print(f"{'fixed 30 mm3 (paper, fitted on all)':38s}{pooled_fixed['dice']:9.4f}{pooled_fixed['f1']:9.4f}"
          f"{pooled_fixed['count_diff']:8.2f}{pooled_fixed['avd_ml']:8.2f}")
    print(f"{'NESTED (threshold never sees fold)':38s}{pooled_nested['dice']:9.4f}{pooled_nested['f1']:9.4f}"
          f"{pooled_nested['count_diff']:8.2f}{pooled_nested['avd_ml']:8.2f}")
    opt = pooled_fixed['f1'] - pooled_nested['f1']
    print(f"\n  selection optimism in Detection F1: {opt:+.4f}")
    print(f"  gain over no filtering, nested estimate: {pooled_nested['f1']-pooled_nofilt['f1']:+.4f}")

    # ---- stratification by reference lesion size ------------------------
    print("\nDOES THE 30 mm^3 RULE REMOVE GENUINE SMALL INFARCTS?")
    print("  (Detection F1 by ground-truth lesion volume; nu=1 mm^3 => vox == mm^3)\n")
    bins = [(0, 1, "empty GT"), (1, 100, "<100 vox"), (100, 1000, "100-1k"),
            (1000, 10000, "1k-10k"), (10000, 10**12, ">=10k")]
    print(f"  {'GT lesion size':16s}{'n':>6}{'no filter':>11}{'30 mm3':>10}{'delta':>9}")
    for lo, hi, lab in bins:
        cs = [c for c in data if lo <= data[c]["gt_vox"] < hi]
        if not cs:
            continue
        a = np.mean([data[c]["m"][nofilt]["f1"] for c in cs])
        b = np.mean([data[c]["m"][fixed]["f1"] for c in cs])
        print(f"  {lab:16s}{len(cs):6d}{a:11.4f}{b:10.4f}{b-a:+9.4f}")


if __name__ == "__main__":
    main()
