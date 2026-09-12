# `analysis/official/` — the organisers' evaluation code goes here

Every scorer in `analysis/` is deliberately built on the **ISLES'26 organisers'
own** evaluation module rather than on a reimplementation of the metric spec.
That decision is the reason several numbers in the paper are what they are, and
it is documented in the docstring of `analysis/official_score.py`.

That module is **not redistributed in this repository.** It is authored by
Ezequiel de la Rosa (ISLES organiser) and ships without a license header, so it
is not ours to re-publish.

## What you need to do

Obtain `eval_utils.py` from the official ISLES'26 challenge materials — the
challenge's own evaluation / metrics repository, linked from the ISLES'26 pages
on Grand Challenge — and place it here:

```
analysis/official/eval_utils.py
```

Nothing else is required. The scorers put `analysis/official` on `sys.path` and
import from it directly:

```python
from eval_utils import (compute_dice_f1_instance_difference,
                        compute_pr_auc,
                        compute_absolute_volume_difference)
```

## The three functions that are used

| Function | Returns | Imported by |
|---|---|---|
| `compute_dice_f1_instance_difference(gt, pred, empty_value=1.0)` | `(lesion-wise F1, absolute lesion-count difference, Dice)` | `official_score.py`, `weighted_ensemble.py`, `best_ensemble.py`, `confidence_filter.py` |
| `compute_pr_auc(gt, prob_map, empty_value=1.0)` | area under the precision–recall curve of the **soft** map | `official_score.py`, `weighted_ensemble.py`, `best_ensemble.py`, `empty_map_risk.py` |
| `compute_absolute_volume_difference(gt, pred, voxel_size_ml)` | absolute volume difference in mL | `official_score.py`, `weighted_ensemble.py`, `best_ensemble.py`, `confidence_filter.py` |

`nested_cv.py` and `sweep_mm3_all5.py` reach all three through
`official_score._score_case` rather than importing them directly, so they need
the file too. The three scripts that do **not**: `site_transfer.py` reads the
per-case JSON `sweep_mm3_all5.py` wrote, `lesion_pr.py` calls Panoptica itself
(see below), and `miss_anatomy.py` does its own IoU matching.

Two details of that file drive decisions throughout `analysis/`, and are worth
reading before you trust any number here:

1. Detection F1 is Panoptica **Recognition Quality** with
   `NaiveThresholdMatching(matching_threshold=0.25)` — a ground-truth lesion
   counts as detected only at IoU ≥ 0.25, matched one-to-one. An "any voxel
   overlap" rule is far more lenient and inflates every F1 number.
2. PR-AUC is `precision_recall_curve` followed by `auc(recall, precision)` —
   **trapezoidal** integration, not `average_precision_score`'s step-function
   sum. The two disagree.

## Dependencies

`eval_utils.py` itself needs `panoptica`, `scikit-learn` and `numpy`. The
scorers additionally need `SimpleITK`, `connected-components-3d` (`cc3d`) and
`scipy`.

```
pip install -r ../../requirements.txt
```

Install the pinned versions rather than the latest: `panoptica` supplies the
instance matching behind every Detection F1 here and its matching semantics have
moved between majors, so an unpinned install reports different numbers on the
same masks without complaining.

`analysis/lesion_pr.py` is the one script that calls `panoptica` directly rather
than through `eval_utils` — it uses exactly the same evaluator settings
(`ConnectedComponentsInstanceApproximator` + `NaiveThresholdMatching` at IoU
0.25) but needs the raw TP/FP/FN counts, which `eval_utils` does not expose.
