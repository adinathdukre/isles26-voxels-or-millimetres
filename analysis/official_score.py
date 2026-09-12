#!/usr/bin/env python3
"""
THE authoritative scorer for this sweep: calls the ORGANISERS' OWN evaluation
code rather than a reimplementation of the metric spec.

`official/eval_utils.py` is Ezequiel de la Rosa's evaluation module from the
ISLES'26 repository. Reading it corrected three things this sweep had wrong:

  1. Detection F1 is Panoptica Recognition Quality with
     NaiveThresholdMatching(matching_threshold=0.25) -- a GT lesion counts as
     detected only at IoU >= 0.25, matched one-to-one. Our earlier "any voxel
     overlap" rule is far more lenient and inflated every F1 number.
  2. PR-AUC is precision_recall_curve followed by auc(recall, precision), i.e.
     TRAPEZOIDAL integration -- not sklearn's average_precision_score, which is
     a step-function sum. They disagree.
  3. For an EMPTY ground truth, compute_pr_auc returns 1.0 if the soft map is
     perfectly constant and 0.0 if it varies at all. The design document says
     empty cases are excluded from AP; the released code does not do that. So a
     case we believe is lesion-free should ship a CONSTANT soft map -- worth a
     full 1.0-vs-0.0 swing on one of the five ranked metrics, at negligible
     downside if we are wrong (a constant map on a non-empty case scores near
     the lesion prevalence, which is tiny either way).

Decision variables swept per run: binarisation threshold, minimum component
size, and a component max-probability cut.

Usage:
  python3 analysis/official_score.py --runs runA,runB --fold 0
  python3 analysis/official_score.py --runs runA --fold 0 \
      --thresholds 0.3,0.5 --min-sizes 0,20,50 --pmax-cuts 0.0,0.8
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
import paths  # noqa: E402  -- roots come from ISLES_ROOT / ISLES_NNUNET_RESULTS / ISLES_GT

RESULTS = paths.RESULTS
GT_DIR = paths.GT
OUT_DIR = os.path.join(HERE, "official_scores")

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)
    warnings.filterwarnings("ignore")


def _variants():
    """Decision-layer settings to score.

    min_size is in VOXELS when `mm3` is False and in mm^3 when True. mm^3 is the
    one we ship: native voxel volume spans 0.031-5.27 mm^3 across this dataset,
    so a voxel-count rule means something different on every scanner.
    """
    return [(t, m, p) for t in _CFG["thresholds"] for m in _CFG["min_sizes"] for p in _CFG["pmax_cuts"]]


class ProbReadError(RuntimeError):
    """An .npz exists but could not be read. Never silently downgraded.

    Falling back to the stored binary mask here is what silently corrupted an
    earlier round of scores: under heavy concurrent I/O a handful of cases per
    run failed to load, took the .nii.gz path instead of `prob >= tau`, and
    shifted whole-run Dice by up to 0.004 and Detection F1 by up to 0.007 --
    larger than the seed-noise floor those numbers were being compared against.
    Two scorings of the same predictions with the same code disagreed. A run is
    now either fully soft-map-scored or it fails.
    """


def _load_prob(vd, case, shape):
    """Lesion-class probability volume for one case.

    Returns None only when the run genuinely has no .npz (a legitimate
    binary-mask-only run). A read that fails is retried, then raised.
    """
    npz_path = os.path.join(vd, case + ".npz")
    if not os.path.exists(npz_path):
        return None
    last = None
    for attempt in range(4):
        try:
            with np.load(npz_path) as z:
                key = "probabilities" if "probabilities" in z else z.files[0]
                arr = z[key]
            pr = (arr[1] if arr.ndim == 4 else arr).astype(np.float32)
            if pr.shape != shape:
                raise ProbReadError(
                    f"{npz_path}: shape {pr.shape} != ground truth {shape}")
            return pr
        except ProbReadError:
            raise
        except Exception as e:                       # transient / partial write
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise ProbReadError(f"{npz_path}: unreadable after 4 attempts ({last!r})")


def _score_case(args):
    import SimpleITK as sitk
    import cc3d
    from eval_utils import (compute_dice_f1_instance_difference, compute_pr_auc,
                            compute_absolute_volume_difference)
    case, run, fold = args
    # `run` may be a "+"-joined list of runs, in which case their probability
    # maps are averaged -- an ensemble scored exactly like a single model.
    members = run.split("+")
    vd = os.path.join(RESULTS, members[0], f"fold_{fold}", "validation")

    g_img = sitk.ReadImage(os.path.join(GT_DIR, case + ".nii.gz"))
    g = sitk.GetArrayFromImage(g_img) > 0.5
    voxel_mm3 = float(np.prod(g_img.GetSpacing()))
    voxel_ml = np.float64(voxel_mm3 / 1000.0)

    loaded = [_load_prob(os.path.join(RESULTS, m, f"fold_{fold}", "validation"),
                         case, g.shape) for m in members]
    prob = None
    if all(p is not None for p in loaded):
        prob = loaded[0] if len(loaded) == 1 else np.mean(loaded, axis=0, dtype=np.float32)
    elif any(p is not None for p in loaded):
        raise ProbReadError(f"{case}: ensemble members disagree on .npz availability")

    out = {"gt_vox": int(g.sum()), "soft": prob is not None}
    # PR-AUC depends only on the soft map, which every decision-layer variant
    # shares, so compute it once rather than once per variant (it is the most
    # expensive term: a sort over ~3M voxels).
    ap_naive = float(compute_pr_auc(g, prob)) if prob is not None else None
    ap_constant = None
    for th, ms, pcut in _variants():
        if prob is None:
            if th != 0.5 or pcut > 0:
                continue
            base = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(vd, case + ".nii.gz"))) > 0.5
        else:
            base = prob >= th

        lab = cc3d.connected_components(base.astype(np.uint8), connectivity=26)
        n = int(lab.max())
        if n:
            sizes = np.bincount(lab.ravel())
            keep = np.zeros(n + 1, dtype=bool)
            # ms is mm^3 when --mm3 is set, otherwise voxels
            min_vox = (ms / max(voxel_mm3, 1e-6)) if _CFG.get("mm3") else ms
            for i in range(1, n + 1):
                if sizes[i] < min_vox:
                    continue
                if pcut > 0 and prob is not None and prob[lab == i].max() < pcut:
                    continue
                keep[i] = True
            pf = keep[lab]
        else:
            pf = base

        f1, dn, dice = compute_dice_f1_instance_difference(g, pf)
        avd = float(compute_absolute_volume_difference(g, pf, voxel_ml))

        rec = {"dice": float(dice), "f1": float(f1), "count_diff": int(dn), "avd_ml": avd}
        if prob is not None:
            # The soft map ships unfiltered. The one exception under test: if we
            # predict no lesion at all, a CONSTANT map scores 1.0 on an empty
            # ground truth instead of 0.0 (see module docstring).
            if pf.any():
                rec["pr_auc"] = ap_naive
            else:
                if ap_constant is None:
                    ap_constant = float(compute_pr_auc(g, np.zeros_like(prob)))
                rec["pr_auc"] = ap_constant
            rec["pr_auc_naive"] = ap_naive
        out[f"{th}|{ms}|{pcut}"] = rec
    return case, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", default=None, help="comma-separated folds to pool (out-of-fold)")
    ap.add_argument("--thresholds", default="0.5")
    ap.add_argument("--min-sizes", default="0,20")
    ap.add_argument("--pmax-cuts", default="0.0")
    ap.add_argument("--mm3", action="store_true",
                    help="interpret --min-sizes as mm^3 rather than voxels (the shipped form)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", default="official")
    a = ap.parse_args()

    cfg = {"thresholds": [float(x) for x in a.thresholds.split(",")],
           "min_sizes": [int(x) for x in a.min_sizes.split(",")],
           "pmax_cuts": [float(x) for x in a.pmax_cuts.split(",")],
           "mm3": a.mm3}
    runs = [r for r in a.runs.split(",") if r]

    folds = [int(x) for x in str(a.folds).split(",")] if a.folds else [a.fold]
    per_run = {}
    for run in runs:
        # each fold's validation set is disjoint, so pooling folds gives
        # out-of-fold predictions for every subject -- the honest surface on
        # which to fit a decision rule
        tasks = []
        first = run.split("+")[0]  # ensembles are "runA+runB"; discover cases from the first member
        for f in folds:
            vd = os.path.join(RESULTS, first, f"fold_{f}", "validation")
            if not os.path.isdir(vd):
                continue
            tasks += [(fn[:-7], run, f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
        if not tasks:
            # A mistyped run name used to print one line and then an empty table
            # with exit status 0, which reads as "scored, nothing to report".
            # Say what is actually there instead.
            have = sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []
            print(f"  skip {run}: no fold_*/validation under {os.path.join(RESULTS, first)}")
            print(f"    a run is the directory nnU-Net writes under ISLES_NNUNET_RESULTS "
                  f"(<trainer>__<plans>__<configuration>); "
                  + (f"available here: {have}" if have else
                     f"nothing is there -- ISLES_NNUNET_RESULTS={RESULTS}"))
            print("    trained without --npz? then validation/ has no .npz and no PR-AUC.")
            continue
        cases = [t[0] for t in tasks]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cfg,)) as ex:
            per_run[run] = dict(ex.map(_score_case, tasks, chunksize=1))
        # A run must be scored the same way on every case. A mix of soft-map and
        # binary-mask cases is not a valid row: the two paths differ enough to
        # swamp the seed-noise floor these numbers get compared against.
        n_soft = sum(1 for v in per_run[run].values() if v.get("soft"))
        if 0 < n_soft < len(cases):
            raise SystemExit(
                f"ABORT {run}: {n_soft}/{len(cases)} cases have a soft map. "
                f"Scoring a mixed run would silently blend two decision paths. "
                f"Finish or delete the partial validation/*.npz set and rerun.")
        print(f"  scored {run}: {len(cases)} cases "
              f"({'soft map' if n_soft else 'binary mask only, no PR-AUC'})")

    # variant keys are the dict-valued ones; "gt_vox" and "soft" are per-case
    # bookkeeping and must not be treated as variants
    keys = sorted({k for res in per_run.values() for v in res.values()
                   for k, x in v.items() if isinstance(x, dict)})
    rows = []
    for run, res in per_run.items():
        cases = sorted(res)
        for k in keys:
            if k not in res[cases[0]]:
                continue
            get = lambda f: np.array([res[c][k][f] for c in cases], dtype=float)
            row = {"run": run, "variant": k, "n": len(cases),
                   "dice": float(get("dice").mean()), "f1": float(get("f1").mean()),
                   "count_diff": float(get("count_diff").mean()),
                   "avd_ml": float(get("avd_ml").mean())}
            if "pr_auc" in res[cases[0]][k]:
                row["pr_auc"] = float(get("pr_auc").mean())
                row["pr_auc_naive"] = float(get("pr_auc_naive").mean())
            rows.append(row)

    if not rows:
        raise SystemExit("ABORT: nothing was scored. Every --runs entry was skipped "
                         "above; a run name or --folds is wrong, or training has not "
                         "written validation/ yet.")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(rows, open(f"{OUT_DIR}/{a.tag}_fold{a.fold if not a.folds else a.folds.replace(',','')}.json", "w"), indent=2)

    have_ap = any("pr_auc" in r for r in rows)
    print(f"\n{'run | th|minsz|pmax':<74} {'Dice':>7} {'DetF1':>7} {'|dN|':>6} {'|dV|mL':>8}"
          + (f" {'PR-AUC':>8}" if have_ap else ""))
    for r in sorted(rows, key=lambda r: -r["f1"]):
        line = (f"{(r['run'][:48] + ' | ' + r['variant']):<74} {r['dice']:.4f} {r['f1']:>7.4f} "
                f"{r['count_diff']:>6.2f} {r['avd_ml']:>8.3f}")
        if "pr_auc" in r:
            line += f" {r['pr_auc']:>8.4f}"
        print(line)

    # rank-then-aggregate, exactly as the challenge describes
    variants = [(r["run"], r["variant"]) for r in rows]
    if len(variants) > 1:
        from scipy.stats import rankdata
        common = None
        for res in per_run.values():
            common = set(res) if common is None else (common & set(res))
        # Rank only on metrics EVERY variant provides. Runs trained before --npz
        # was standard have no probability maps and therefore no PR-AUC; mixing
        # 4- and 5-metric vectors would compare them on different scales (and
        # used to crash outright).
        use_ap = all("pr_auc" in per_run[run][c][k]
                     for run, k in variants for c in list(common)[:1])
        if not use_ap:
            print("  note: PR-AUC missing for at least one run "
                  "(no saved probabilities) -- ranking on the 4 mask metrics only")
        agg = {v: [] for v in variants}
        for c in sorted(common):
            cols = []
            for run, k in variants:
                r = per_run[run][c][k]
                m = [r["dice"], -r["avd_ml"], r["f1"], -r["count_diff"]]
                if use_ap:
                    m.append(r["pr_auc"])
                cols.append(m)
            arr = np.array(cols, dtype=float)
            ranks = np.zeros(len(variants))
            for m in range(arr.shape[1]):
                ranks += rankdata(-arr[:, m], method="average")
            ranks /= arr.shape[1]
            for v, rk in zip(variants, ranks):
                agg[v].append(rk)
        ranking = sorted(((v, float(np.mean(rk))) for v, rk in agg.items()), key=lambda x: x[1])
        print("\n--- simulated ISLES'26 rank-then-aggregate (lower is better) ---")
        for (run, k), s in ranking:
            print(f"  {s:6.3f}  {run[:52]:<52} {k}")
        json.dump([{"run": r, "variant": k, "mean_rank": s} for (r, k), s in ranking],
                  open(f"{OUT_DIR}/{a.tag}_fold{a.fold if not a.folds else a.folds.replace(',','')}_ranking.json", "w"), indent=2)


if __name__ == "__main__":
    main()
