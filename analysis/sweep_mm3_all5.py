#!/usr/bin/env python3
"""Build the per-case matrix that site_transfer.py consumes.

This scores every pooled out-of-fold subject under BOTH parameterisations of the
minimum-component-size rule -- voxel counts and mm^3 -- so that the two families
can be compared on identical predictions, and so that an operating point chosen
on one set of scanners can be replayed on another (see site_transfer.py).

Provenance note, because it matters for trust: the original producer of
`sweep_mm3_all5.json` was a scratchpad one-off that did not survive into the
repository. This is a rebuild of it on top of official_score.py, which is the
authoritative scorer -- so the metrics here come from the organisers' own
eval_utils.py and 26-connected cc3d components, exactly as everywhere else,
rather than from a second private copy of the metric code. The shipped
`sweep_mm3_all5.json` was produced from the 250-epoch baseline run; its column
means are recorded in analysis/README.md so a rebuild can be checked against
them.

The voxel family deliberately carries only {0, 20}: 20 voxels is the voxel rule
the rank objective selects on every fold, and 0 is "no filter". The mm^3 family
is swept more densely because that is the family the paper argues for, and
because 20 mm^3 vs 20 voxels is the matched-magnitude comparison that isolates
the units effect from the magnitude effect.

Usage:
  python3 analysis/sweep_mm3_all5.py                      # all five folds
  python3 analysis/sweep_mm3_all5.py --run <results_folder> --workers 16

--run defaults to the 250-epoch baseline the paper's units numbers were measured
on, which is NOT one of the two runs train_shipped.sh trains; pass the run you
actually have. The .json itself is not in this repository -- it is per-subject
derived data -- so this script has to be run once before site_transfer.py or
anything else that consumes the cache.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "official"))
import paths  # noqa: E402  -- roots come from ISLES_ROOT / ISLES_NNUNET_RESULTS / ISLES_GT
import official_score as osc  # noqa: E402

VOX_SIZES = [0, 20]
MM3_SIZES = [10, 20, 30, 40, 60, 80, 120]
TAU = 0.5

# official_score returns metrics under these names; site_transfer expects these prefixes.
FIELDS = (("dice", "dice"), ("f1", "f1"), ("avd", "avd_ml"), ("cdiff", "count_diff"))


def _pass(tasks, run, min_sizes, mm3, workers):
    """Score every case at every min_size in one parameterisation."""
    cfg = {"thresholds": [TAU], "min_sizes": min_sizes, "pmax_cuts": [0.0], "mm3": mm3}
    with ProcessPoolExecutor(max_workers=workers, initializer=osc._init,
                             initargs=(cfg,)) as ex:
        return dict(ex.map(osc._score_case, tasks, chunksize=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.environ.get(
        "ISLES_RUN", "nnUNetTrainer_250epochs" + paths.PLANS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default=os.path.join(HERE, "sweep_mm3_all5.json"))
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",")]
    tasks = []
    for f in folds:
        vd = os.path.join(osc.RESULTS, a.run, f"fold_{f}", "validation")
        if not os.path.isdir(vd):
            raise SystemExit(f"missing validation directory: {vd}")
        tasks += [(fn[:-7], a.run, f) for fn in sorted(os.listdir(vd)) if fn.endswith(".nii.gz")]
    if not tasks:
        raise SystemExit("no validation predictions found")

    print(f"{len(tasks)} out-of-fold cases | {len(VOX_SIZES)} voxel settings | "
          f"{len(MM3_SIZES)} mm^3 settings", flush=True)
    vox = _pass(tasks, a.run, VOX_SIZES, False, a.workers)
    print("  voxel pass done", flush=True)
    mm3 = _pass(tasks, a.run, MM3_SIZES, True, a.workers)
    print("  mm^3 pass done", flush=True)

    rows = []
    for case, _, _ in tasks:
        # the .nii.gz suffix is kept so the row key matches the prediction
        # filename; site_transfer's site_of() parses the subject id out of it
        row = {"case": case + ".nii.gz"}
        for tag, res, sizes in (("vox", vox, VOX_SIZES), ("mm3", mm3, MM3_SIZES)):
            for s in sizes:
                rec = res[case][f"{TAU}|{s}|0.0"]
                for prefix, field in FIELDS:
                    row[f"{prefix}_{tag}_{s}"] = float(rec[field])
        rows.append(row)

    json.dump(rows, open(a.out, "w"))
    print(f"\nwrote {a.out}  ({len(rows)} cases)")

    import numpy as np
    print(f"\n{'setting':>10}{'Dice':>9}{'Det F1':>9}{'|dN|':>8}")
    for tag, sizes in (("vox", VOX_SIZES), ("mm3", MM3_SIZES)):
        for s in sizes:
            print(f"{tag + '_' + str(s):>10}"
                  f"{np.mean([r[f'dice_{tag}_{s}'] for r in rows]):9.4f}"
                  f"{np.mean([r[f'f1_{tag}_{s}'] for r in rows]):9.4f}"
                  f"{np.mean([r[f'cdiff_{tag}_{s}'] for r in rows]):8.4f}")
    d = (np.mean([r["f1_mm3_20"] for r in rows]) - np.mean([r["f1_vox_20"] for r in rows]))
    print(f"\n  matched magnitude, 20 mm^3 - 20 voxels: {d:+.4f} Detection F1")


if __name__ == "__main__":
    main()
