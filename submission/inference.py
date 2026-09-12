"""
ISLES'26 algorithm for Grand Challenge — nnU-Net v2 (ResEnc-M) baseline.

  init_model() — loads the trained nnU-Net predictor once at server startup.
  run(model)   — reads /input, runs nnU-Net inference, writes /output.

Your algorithm's interfaces:

  interf0:
    Inputs:
      - /input/images/t1-brain-mri
      - /input/stroke-metadata.json
    Outputs:
      - /output/images/stroke-lesion-segmentation
      - /output/images/lesion-probability-map

This baseline is a single-channel (T1w-only) nnU-Net model; the stroke
metadata is logged but not consumed by the model.

For more details see:
  https://grand-challenge.org/documentation/algorithms/
  https://grand-challenge.org/documentation/runtime-environment/
"""

import glob
import json
import traceback
from pathlib import Path

import cc3d
import numpy
import SimpleITK
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_DIR = Path("/opt/ml/model")

# Must match the folder name nnU-Net wrote under nnUNet_results, and the
# checkpoint/fold(s) we chose to package (see model/README.md).
# Ensemble members. Their probability maps are averaged before thresholding.
# Both are ResEnc-M/3d_fullres and differ only in training: a 500-epoch schedule
# and a Dice+TopK10 loss. Averaging them raises pooled out-of-fold PR-AUC by
# 0.0088 (2.6x the two-seed reference) while Detection F1 and Dice move by less
# than their own references, and it wins the rank-then-aggregate objective.
#
# NOTE: nnU-Net imports the trainer class named inside each checkpoint to rebuild
# the architecture. nnUNetTrainerDiceTopK10_250epochs is a custom class that does
# not exist in this image, so the shipped TopK10 checkpoints have `trainer_name`
# rewritten to the stock nnUNetTrainer. That is safe because TopK10 changes only
# the loss -- verified bit-for-bit identical predictions before/after the rename.
NNUNET_MODEL_SUBDIRS = [
    "nnUNetTrainer_500epochs__nnUNetResEncUNetMPlans__3d_fullres",
    "topk10__nnUNetResEncUNetMPlans__3d_fullres",
]

# Relative weight of each member when averaging probability maps, in the order above.
# Fitted on the pooled out-of-fold cohort (n=1,453); see the note at the averaging step.
ENSEMBLE_WEIGHTS = [0.65, 0.35]

# checkpoint_final, NOT checkpoint_best. nnU-Net's own validation
# (nnUNetv2_train --val, run_training.py:79-82) loads checkpoint_final, so every
# number this pipeline was tuned against -- the operating point below included --
# describes the final-epoch weights. Deploying checkpoint_best would ship a model
# we never measured. Use --val_best if you ever want to switch, and re-tune.
CHECKPOINT_NAME = "checkpoint_final.pth"
USE_FOLDS = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------------------
# Instance decision layer.
#
# ISLES'26 ranks on five per-case metrics (Dice, absolute volume difference,
# PR-AUC on the soft map, lesion-wise Detection F1, absolute lesion-count
# difference), rank-then-aggregate. Three of those five are hurt by spurious
# connected components, and only one (Dice) rewards raw overlap -- so dropping
# small components is strongly net-positive.
#
# The threshold is in PHYSICAL UNITS, not voxels, because ISLES'26 is native
# space: measured across the 1,453 training volumes, voxel volume spans
# 0.031-5.27 mm^3 (a 168x spread; only 71.6% of cases lie within [0.9, 1.1]).
# A "20-voxel" rule -- the form every prior ATLAS post-processing rule takes,
# because ATLAS v2.0 was MNI-registered at 1 mm^3 -- would mean 0.63 mm^3 on one
# scanner and 105 mm^3 on another.
#
# 30 mm^3 was fitted on pooled 5-fold out-of-fold predictions for all 1,453
# training subjects against the challenge's own rank-then-aggregate objective,
# using the organisers' evaluation code. Measured effect:
#   Detection F1  0.5283 -> 0.5849  (+0.057)
#   |lesion count difference|  2.09 -> 1.86
#   Dice  0.6395 -> 0.6391  (-0.0004)
# An mm^3 setting was rank-optimal in all five folds independently.
MIN_LESION_MM3 = 30.0

# Binarisation cutoff for the lesion probability map (nnU-Net's argmax default is
# exactly 0.5 for this 2-class task).
#
# Fitted jointly with MIN_LESION_MM3 on POOLED OUT-OF-FOLD predictions for all
# 1,453 training subjects, scored with the organisers' evaluation code against
# the rank-then-aggregate objective. Out-of-fold ranks at 30 mm^3 (lower better):
#   tau 0.5 -> 12.379   <-- best
#   tau 0.6 -> 12.380
#   tau 0.7 -> 12.583
#   tau 0.8 -> 12.960
#   tau 0.9 -> 13.718
#
# A single-fold sweep suggested tau=0.8 was worth +0.008 Detection F1; it did not
# replicate on 1,453 cases. Raising tau is NOT a lever here -- the size filter
# above is. Do not re-tune this on one fold.
#
# A further caution if you ever do re-tune it: this container ensembles 5 folds,
# and averaging five softmaxes flattens the distribution relative to any single
# member, so a cutoff fitted on single-fold predictions would not transfer
# cleanly anyway.
BINARY_THRESHOLD = 0.5

# NOTE, deliberately NOT enabled: the organisers' released scorer returns
# PR-AUC = 1.0 for an empty ground truth if the soft map is perfectly constant,
# and 0.0 if it varies at all -- so emitting a constant map whenever we predict
# "no lesion" would be worth up to a full point per empty-GT case. The ISLES'26
# design document, however, states empty-GT cases are excluded from PR-AUC
# (NaN), which would make the trick a no-op, and the copy of the scorer we could
# obtain appears to be the ISLES'24 one. Since a constant map also throws away
# real ranking information whenever we wrongly predict empty, this stays off
# until the official ISLES'26 scorer confirms the behaviour.
EMIT_CONSTANT_MAP_WHEN_EMPTY = False


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def init_model():
    """Load and return the trained nnU-Net predictor.

    Called once by app.py during server startup (before /health returns 200).
    The predictor is reused across all /invoke calls.
    """
    _show_torch_cuda_info()

    available = torch.cuda.is_available()
    device = torch.device("cuda") if available else torch.device("cpu")
    print(f"Using device: {device}")

    # Mirroring test-time augmentation multiplies inference cost up to 8x
    # (2^3 flipped forward passes per patch). That is affordable on GPU but
    # blows well past the 7-minute/case CPU budget on Grand Challenge's
    # weaker CPU instances -- disable it when running on CPU.
    use_mirroring = available
    print(f"Using mirroring (TTA): {use_mirroring}")

    # Ensembling doubles inference cost. On GPU that is ~41 s/case (measured) and
    # entirely affordable. On CPU a single member already takes ~175 s, so the pair
    # blows past the per-case budget -- the same reason mirroring TTA is disabled
    # above. Fall back to the first (strongest) member when there is no GPU.
    subdirs = NNUNET_MODEL_SUBDIRS if available else NNUNET_MODEL_SUBDIRS[:1]
    if len(subdirs) < len(NNUNET_MODEL_SUBDIRS):
        print(f"no GPU: using {len(subdirs)}/{len(NNUNET_MODEL_SUBDIRS)} ensemble members "
              f"to stay inside the CPU runtime budget")

    predictors = []
    for subdir in subdirs:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=use_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(MODEL_DIR / subdir),
            use_folds=USE_FOLDS,
            checkpoint_name=CHECKPOINT_NAME,
        )
        predictors.append(predictor)
        print(f"loaded ensemble member: {subdir} (folds {USE_FOLDS})")
    return predictors


def run(model):
    """This is called each time the /invoke endpoint is called.
    This should read input from /input, run inference, and write output to /output.
    """
    interface_key = get_interface_key()

    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]

    return handler(model)


def interf0_handler(predictors):
    t1_image_path = find_input_image(location=INPUT_PATH / "images/t1-brain-mri")

    # Reference geometry is read FIRST, before anything that can fail. Every output
    # we can possibly write needs it -- including the degraded fallback below -- so
    # it must not sit behind inference. GetSize() is header-only; no pixels loaded.
    reference_image = SimpleITK.ReadImage(str(t1_image_path))
    voxel_mm3 = float(numpy.prod(reference_image.GetSpacing()))
    volume_shape = tuple(reversed(reference_image.GetSize()))  # SITK (X,Y,Z) -> numpy (Z,Y,X)

    # =========================================================================
    # METADATA NOTES:
    # 1) Unlike the training dataset where the acquisition site key is referred
    #    to as 'SITE', in Grand Challenge inputs it is named 'CENTER'.
    # 2) Metadata values (e.g., DAYS_POST_STROKE, CHRONICITY) may be `null`.
    # This baseline is T1w-only and does not consume this metadata; it is
    # logged here for traceability / future extension.
    # =========================================================================
    # Logged only -- this model is T1w-only. A missing or malformed metadata file
    # must never be able to fail a case, so reading it is advisory.
    try:
        input_stroke_metadata = load_json_file(location=INPUT_PATH / "stroke-metadata.json")
        print("=+=" * 10)
        print("Loaded Stroke Metadata:")
        print(json.dumps(input_stroke_metadata, indent=2))
        print("=+=" * 10)
    except Exception as exc:
        print(f"WARNING: could not read stroke-metadata.json ({exc!r}); continuing.")

    # One pathological case must not take the whole submission with it. nnU-Net can
    # raise on unexpected headers, and CUDA can OOM on an unusually large volume; if
    # either happens we still emit geometrically valid outputs, so this case scores
    # 0 while every other case is unaffected. An uncaught raise here would fail the
    # container instead, which scores nothing anywhere.
    try:
        # Load via nnU-Net's own SimpleITK reader so the array axis order and
        # spacing convention exactly match what the model was trained on. Inside the
        # guard: a corrupt volume must degrade this case, not fail the container.
        input_image, image_properties = SimpleITKIO().read_images([str(t1_image_path)])

        # Average the ensemble members' probability maps. Each member is itself a
        # 5-fold ensemble; members run sequentially so peak VRAM stays ~3 GB.
        member_maps = []
        for predictor in predictors:
            _, probabilities = predictor.predict_single_npy_array(
                input_image, image_properties, None, None, True,
            )
            # probabilities has shape (num_classes, Z, Y, X) for this 2-class
            # (background=0, lesion=1) task; take the lesion-class channel.
            member_map = numpy.asarray(probabilities)
            if member_map.ndim == 4:
                member_map = member_map[1]
            member_maps.append(member_map.astype(numpy.float32))

        # Weighted, not equal. The members are not equally good: 500ep carries Dice and
        # Detection F1, TopK10 contributes PR-AUC. Sweeping the weight over all 1,453
        # out-of-fold subjects, 0.65/0.35 beats the equal average on four of five metrics
        # and matches the best single model on every mask metric while still gaining
        # +0.0091 PR-AUC (2.7x its two-seed reference).
        if len(member_maps) == 1:
            lesion_probability_map = member_maps[0]
        else:
            w = numpy.asarray(ENSEMBLE_WEIGHTS[:len(member_maps)], dtype=numpy.float32)
            w = w / w.sum()
            lesion_probability_map = numpy.tensordot(w, numpy.stack(member_maps), axes=1
                                                     ).astype(numpy.float32)
        print(f"ensembled {len(member_maps)} member(s), weights "
              f"{[round(float(x), 3) for x in (numpy.asarray(ENSEMBLE_WEIGHTS[:len(member_maps)]) / numpy.sum(ENSEMBLE_WEIGHTS[:len(member_maps)]))]}")

        # Binarise from the probability map at BINARY_THRESHOLD rather than taking
        # nnU-Net's argmax, so the cutoff is a tunable parameter of the decision
        # layer. At 0.5 the two are identical for this 2-class task.
        # Threshold the AVERAGED map. (A single member's argmax is not a valid
        # shortcut once the maps are averaged.)
        thresholded = (lesion_probability_map >= BINARY_THRESHOLD).astype(numpy.uint8)

        binary_segmentation_mask = apply_instance_decision_layer(thresholded, voxel_mm3)

        # The soft map is submitted SEPARATELY from the binary mask and is used
        # exclusively for PR-AUC, so it ships unfiltered -- filtering it could only
        # destroy ranking information that the binary mask's threshold already
        # discarded.
        if EMIT_CONSTANT_MAP_WHEN_EMPTY and not binary_segmentation_mask.any():
            lesion_probability_map = numpy.zeros_like(lesion_probability_map)
    except Exception:
        traceback.print_exc()
        print(f"ERROR: inference failed for {t1_image_path}; emitting empty outputs "
              f"of shape {volume_shape} so the run completes.", flush=True)
        binary_segmentation_mask = numpy.zeros(volume_shape, dtype=numpy.uint8)
        lesion_probability_map = numpy.zeros(volume_shape, dtype=numpy.float32)

    # A NaN or inf reaching the writer would produce an unreadable map; clamp rather
    # than trust the network's output range.
    lesion_probability_map = numpy.nan_to_num(
        lesion_probability_map, nan=0.0, posinf=1.0, neginf=0.0
    ).astype(numpy.float32)
    numpy.clip(lesion_probability_map, 0.0, 1.0, out=lesion_probability_map)

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary_segmentation_mask,
        reference_image=reference_image,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=lesion_probability_map,
        reference_image=reference_image,
    )

    return 0


def apply_instance_decision_layer(mask, voxel_mm3):
    """Drop predicted lesion components smaller than MIN_LESION_MM3.

    Deliberately NOT nnU-Net's own `determine_postprocessing`, whose only rule is
    remove-all-but-largest-connected-component: ground truth here averages 3.18
    lesion components per case (max 27), so keeping only the largest measured
    -0.029 Dice and -0.067 Detection F1 on our out-of-fold predictions.

    26-connectivity matches the convention used by the challenge's Panoptica-based
    Detection F1 (ConnectedComponentsInstanceApproximator).
    """
    if not mask.any():
        return mask
    min_voxels = MIN_LESION_MM3 / max(voxel_mm3, 1e-6)
    labels = cc3d.connected_components(mask.astype(numpy.uint8), connectivity=26)
    n = int(labels.max())
    if n == 0:
        return mask
    sizes = numpy.bincount(labels.ravel())
    keep = numpy.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        keep[i] = sizes[i] >= min_voxels
    filtered = keep[labels].astype(numpy.uint8)
    print(f"instance decision layer: {n} components -> {int(keep.sum())} kept "
          f"(min {MIN_LESION_MM3} mm^3 = {min_voxels:.1f} voxels at {voxel_mm3:.3f} mm^3/voxel)")
    return filtered


def get_interface_key():
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def find_input_image(*, location):
    input_files = (
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if not input_files:
        raise FileNotFoundError(f"No valid image file found in {location}")
    return input_files[0]


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)

    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
