#!/usr/bin/env python3
"""
The decisive experiment for the physical-unit decision-layer claim.

The claim is that a component-size threshold expressed in mm^3 generalises to
unseen scanners better than one expressed in voxels, because ISLES'26 is
native-space (voxel volume spans 0.031-5.27 mm^3 across the training set) and
the test set contains institutions we have never seen.

Protocol: partition the sites into disjoint fit/held-out halves, choose the
operating point on the fit sites by the challenge's own rank-then-aggregate
objective, then score that choice on the held-out sites. Repeat over many random
site partitions. This is the only way to ask "will the threshold we picked still
be right on a new scanner?" without the hidden test set.

Falsification condition, stated in advance: if the mm^3 family does not beat the
voxel family on held-out sites, the physical-unit claim is dead and gets reported
as a negative.

Input is the per-case sweep over both parameterisations
(analysis/sweep_mm3_all5.json, rebuilt by analysis/sweep_mm3_all5.py): every case
scored with the organisers' eval_utils.py at each candidate threshold.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

DATA = os.environ.get(
    "ISLES_MM3_SWEEP",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_mm3_all5.json"))
N_SPLITS = 400
RNG = np.random.default_rng(0)


def site_of(case):
    """sub-rNNNsNNN_ses-1 -> RNNN ; sub-<site><NNN>_ses-1 -> <SITE>"""
    m = re.match(r"sub-(r\d+)s\d+", case)
    if m:
        return m.group(1).upper()
    m = re.match(r"sub-([a-zA-Z]+)\d+", case)
    return m.group(1).upper() if m else "UNKNOWN"


def main():
    rows = json.load(open(DATA))
    variants = sorted({k[len("dice_"):] for k in rows[0] if k.startswith("dice_")})
    vox = [v for v in variants if v.startswith("vox_")]
    mm3 = [v for v in variants if v.startswith("mm3_")]
    print(f"{len(rows)} cases | {len(vox)} voxel settings | {len(mm3)} mm^3 settings")

    sites = np.array([site_of(r["case"]) for r in rows])
    uniq = sorted(set(sites))
    print(f"{len(uniq)} sites; largest: "
          f"{sorted(((s, int((sites == s).sum())) for s in uniq), key=lambda x: -x[1])[:5]}")

    # metric matrix: (cases, variants, 4), higher is better
    M = np.array([[[r[f"dice_{v}"], -r[f"avd_{v}"], r[f"f1_{v}"], -r[f"cdiff_{v}"]]
                   for v in variants] for r in rows], dtype=float)
    vidx = {v: i for i, v in enumerate(variants)}

    def mean_rank(case_mask, family):
        """Mean rank-then-aggregate score of each setting in `family`, over the given cases."""
        cols = [vidx[v] for v in family]
        sub = M[case_mask][:, cols, :]
        out = np.zeros(len(cols))
        for i in range(sub.shape[0]):
            r = np.zeros(len(cols))
            for m in range(4):
                r += rankdata(-sub[i, :, m], method="average")
            out += r / 4
        return out / max(sub.shape[0], 1)

    def f1_of(case_mask, variant):
        return float(np.mean([r[f"f1_{variant}"] for r, k in zip(rows, case_mask) if k]))

    def dice_of(case_mask, variant):
        return float(np.mean([r[f"dice_{variant}"] for r, k in zip(rows, case_mask) if k]))

    res = {"mm3": {"f1": [], "dice": [], "picked": []},
           "vox": {"f1": [], "dice": [], "picked": []}}

    for _ in range(N_SPLITS):
        perm = RNG.permutation(uniq)
        fit_sites = set(perm[: len(perm) // 2])
        fit = np.array([s in fit_sites for s in sites])
        held = ~fit
        if fit.sum() < 50 or held.sum() < 50:
            continue
        for fam_name, fam in (("mm3", mm3), ("vox", vox)):
            ranks = mean_rank(fit, fam)
            best = fam[int(np.argmin(ranks))]          # chosen on FIT sites only
            res[fam_name]["f1"].append(f1_of(held, best))    # scored on HELD-OUT sites
            res[fam_name]["dice"].append(dice_of(held, best))
            res[fam_name]["picked"].append(best)

    print(f"\n--- operating point fitted on half the sites, scored on the other half "
          f"({len(res['mm3']['f1'])} random partitions) ---")
    print(f"{'family':<6} {'held-out Det.F1':>16} {'held-out Dice':>14}  most-picked setting")
    for fam in ("mm3", "vox"):
        f1 = np.array(res[fam]["f1"])
        dc = np.array(res[fam]["dice"])
        picks = defaultdict(int)
        for p in res[fam]["picked"]:
            picks[p] += 1
        top = sorted(picks.items(), key=lambda x: -x[1])[:3]
        top_s = ", ".join(f"{k}({100*v/len(res[fam]['picked']):.0f}%)" for k, v in top)
        print(f"{fam:<6} {f1.mean():>10.4f} ± {f1.std():.4f} {dc.mean():>10.4f}  {top_s}")

    d = np.array(res["mm3"]["f1"]) - np.array(res["vox"]["f1"])
    print(f"\npaired difference (mm^3 - voxel) on held-out sites: "
          f"{d.mean():+.4f} ± {d.std():.4f}")
    print(f"mm^3 wins in {100*np.mean(d > 0):.1f}% of partitions, ties {100*np.mean(d == 0):.1f}%")
    verdict = "SUPPORTED" if d.mean() > 0 and np.mean(d > 0) > 0.5 else "NOT SUPPORTED"
    print(f"\nphysical-unit claim: {verdict}")


if __name__ == "__main__":
    main()
