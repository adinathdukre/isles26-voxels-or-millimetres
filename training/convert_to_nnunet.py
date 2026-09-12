"""
Convert the extracted ISLES'26 (ATLAS v3.0 Raw) dataset into nnU-Net v2's
raw dataset format.

Source layout (per subject):
  ATLAS3_Training_Raw/<SITE>/sub-<id>/ses-1/anat/
    sub-<id>_ses-1_space-orig_desc-brain_T1w.nii.gz
    sub-<id>_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
    sub-<id>_ses-1_metadata.csv

Target layout (nnU-Net v2 raw):
  nnUNet_raw/Dataset001_ISLES26/
    imagesTr/<case>_0000.nii.gz
    labelsTr/<case>.nii.gz
    dataset.json
    case_metadata.csv

Images and labels are SYMLINKED, not copied: the raw release is ~1,453 volumes
and nnU-Net only ever reads them once, during preprocessing.

Two subjects have empty metadata rows but valid images/masks -- they are
still included as training cases here; only the metadata table skips them.
(Subject identifiers are deliberately not reproduced in this repository.)

Paths come from the environment (see training/env.sh):
    ISLES_ATLAS3_RAW   extracted ATLAS3_Training_Raw directory
                       (default: $ISLES_ROOT/data/ATLAS3_Training_Raw)
    nnUNet_raw         nnU-Net's raw directory
                       (default: $ISLES_ROOT/nnUNet_raw)
Both can be overridden on the command line with --src / --dst.
"""
import argparse
import csv
import glob
import json
import os

ROOT = os.environ.get("ISLES_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SRC = os.environ.get("ISLES_ATLAS3_RAW", os.path.join(ROOT, "data", "ATLAS3_Training_Raw"))
DEFAULT_DST = os.path.join(
    os.environ.get("nnUNet_raw", os.path.join(ROOT, "nnUNet_raw")), "Dataset001_ISLES26")


def find_subject_dirs(src_root):
    return sorted(glob.glob(os.path.join(src_root, "*", "sub-*", "ses-*", "anat")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="extracted ATLAS3_Training_Raw directory")
    ap.add_argument("--dst", default=DEFAULT_DST, help="nnU-Net raw dataset directory to create")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)
    images_tr = os.path.join(dst_root, "imagesTr")
    labels_tr = os.path.join(dst_root, "labelsTr")

    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)

    anat_dirs = find_subject_dirs(src_root)
    print(f"Found {len(anat_dirs)} subject anat dirs under {src_root}")

    case_ids = []
    metadata_rows = []
    skipped = []

    for d in anat_dirs:
        t1_matches = glob.glob(os.path.join(d, "*_T1w.nii.gz"))
        mask_matches = glob.glob(os.path.join(d, "*_mask.nii.gz"))
        meta_matches = glob.glob(os.path.join(d, "*_metadata.csv"))

        if len(t1_matches) != 1 or len(mask_matches) != 1:
            skipped.append((d, "missing image or mask"))
            continue

        t1_path = t1_matches[0]
        mask_path = mask_matches[0]

        # case id: sub-<id> derived from the T1w filename prefix
        base = os.path.basename(t1_path)
        case_id = base.split("_ses-")[0] + "_" + base.split("_")[1]  # sub-XXXX_ses-1

        dst_img = os.path.join(images_tr, f"{case_id}_0000.nii.gz")
        dst_lbl = os.path.join(labels_tr, f"{case_id}.nii.gz")

        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(t1_path), dst_img)
        if not os.path.exists(dst_lbl):
            os.symlink(os.path.abspath(mask_path), dst_lbl)

        case_ids.append(case_id)

        row = {"case_id": case_id, "SESSION_ID": case_id}
        if meta_matches:
            with open(meta_matches[0], newline="") as fh:
                reader = csv.DictReader(fh)
                data_rows = list(reader)
                if data_rows:
                    row.update(data_rows[0])
        metadata_rows.append(row)

    print(f"Linked {len(case_ids)} cases into imagesTr/labelsTr")
    if skipped:
        print(f"Skipped {len(skipped)} subject dirs:")
        for s in skipped:
            print("  ", s)

    dataset_json = {
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "ISLES26_ATLAS3_Raw",
        "description": "ISLES'26 native-space T1w ischemic stroke lesion segmentation (ATLAS v3.0 raw release)",
        "reference": "ATLAS v2.0 (Liew et al. 2022) + SOOP (Absher et al. 2024) + 329 new cases, released via NITRC ATLAS v3.0",
    }
    with open(os.path.join(dst_root, "dataset.json"), "w") as fh:
        json.dump(dataset_json, fh, indent=2)

    with open(os.path.join(dst_root, "case_metadata.csv"), "w", newline="") as fh:
        fieldnames = ["case_id", "SESSION_ID", "ATLAS2_DATASET", "DAYS_POST_STROKE", "CHRONICITY", "SITE"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)

    print(f"Wrote dataset.json (numTraining={len(case_ids)}) and case_metadata.csv under {dst_root}")


if __name__ == "__main__":
    main()
