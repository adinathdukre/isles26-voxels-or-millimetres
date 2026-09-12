#!/usr/bin/env python3
"""
Build Dataset504_ISLES26_MSL: Multi-Size Labeling (Shang et al., "Segmenting
Small Stroke Lesions with Novel Labeling Strategies", MLCN@MICCAI 2024,
arXiv:2408.02929).

Instead of one lesion class, each ground-truth connected component is assigned a
class by its voxel count. The network then gets an explicitly size-aware
objective, so a 40-voxel satellite lesion is no longer competing for gradient
against a 40,000-voxel infarct under the same label. At inference the classes
are merged back to a single lesion mask (argmax, then map 1..4 -> 1) and the
soft map is the sum of the four lesion channels.

This is the only published method the literature sweep found that beats
MAPPING (the ATLAS v2.0 winner) on ATLAS: reported +3.6% recall, +2.4% F1,
+1.3% Dice on the small-lesion subset.

Size bins match the ones this project already reports on, and are consistent
with the measured component distribution over the 1,453 training masks
(40.4% of components are <100 voxels, 69.4% <1000).

Images are symlinked, not copied -- only the labels differ from Dataset001.

It did not reproduce here: on fold 0, against the identical split, the merged
MSL prediction lost to the plain single-class baseline on the challenge's
rank-then-aggregate objective. Kept because it is the cleanest size-aware
labeling arm we ran, and because a negative replication is worth the fifty
lines.

Paths come from the environment (see training/env.sh): nnUNet_raw, or
ISLES_ROOT/nnUNet_raw as a fallback. ISLES_WORKERS caps the process pool.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.environ.get("ISLES_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.environ.get("nnUNet_raw", os.path.join(ROOT, "nnUNet_raw"))
SRC = os.path.join(RAW, "Dataset001_ISLES26")
DST = os.path.join(RAW, "Dataset504_ISLES26_MSL")

# Relabelling is CPU-bound and embarrassingly parallel; 48 was the machine this
# ran on, not a requirement.
WORKERS = int(os.environ.get("ISLES_WORKERS", min(48, os.cpu_count() or 8)))

# (upper bound exclusive, class index)
BINS = [(100, 1), (1000, 2), (10000, 3), (float("inf"), 4)]


def relabel(case):
    import SimpleITK as sitk
    import cc3d
    src = os.path.join(SRC, "labelsTr", case + ".nii.gz")
    dst = os.path.join(DST, "labelsTr", case + ".nii.gz")
    if os.path.exists(dst):
        return case, None
    img = sitk.ReadImage(src)
    a = sitk.GetArrayFromImage(img) > 0.5          # 41 masks carry float rounding noise
    lab = cc3d.connected_components(a.astype(np.uint8), connectivity=26)
    n = int(lab.max())
    out = np.zeros(a.shape, dtype=np.uint8)
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    if n:
        sizes = np.bincount(lab.ravel())
        cls_of = np.zeros(n + 1, dtype=np.uint8)
        for i in range(1, n + 1):
            s = sizes[i]
            for ub, c in BINS:
                if s < ub:
                    cls_of[i] = c
                    counts[c] += 1
                    break
        out = cls_of[lab]
    res = sitk.GetImageFromArray(out)
    res.CopyInformation(img)
    sitk.WriteImage(res, dst, useCompression=True)
    return case, counts


def main():
    os.makedirs(os.path.join(DST, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(DST, "labelsTr"), exist_ok=True)

    cases = sorted(f[:-7] for f in os.listdir(os.path.join(SRC, "labelsTr")) if f.endswith(".nii.gz"))
    print(f"{len(cases)} cases")

    for c in cases:
        s = os.path.realpath(os.path.join(SRC, "imagesTr", c + "_0000.nii.gz"))
        d = os.path.join(DST, "imagesTr", c + "_0000.nii.gz")
        if not os.path.exists(d):
            os.symlink(s, d)

    total = {1: 0, 2: 0, 3: 0, 4: 0}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (case, counts) in enumerate(ex.map(relabel, cases, chunksize=4)):
            if counts:
                for k, v in counts.items():
                    total[k] += v
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(cases)}")

    json.dump({
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion_xs": 1, "lesion_s": 2, "lesion_m": 3, "lesion_l": 4},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": "ISLES26_ATLAS3_MSL",
        "description": "Multi-Size Labeling (Shang et al. 2024): lesion components split into "
                       "4 classes by voxel count (<100, 100-1k, 1k-10k, >=10k). Merged back to a "
                       "single lesion class at inference.",
    }, open(os.path.join(DST, "dataset.json"), "w"), indent=2)

    print(f"component counts by class: {total} (total {sum(total.values())})")
    print(f"wrote {DST}")
    print("NEXT: copy Dataset001's splits_final.json into the preprocessed dir for this "
          "dataset after planning, or the folds will not line up and no comparison is valid.")


if __name__ == "__main__":
    main()
