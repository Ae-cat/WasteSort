# Week 3 carryover: prediction logs and image tests

## Status (2026-10-25)

Prediction logging and the held-out test run are complete. Independent external-image
performance is **pending: no externally captured image set was found in this checkout**.
The 719 RealWaste test images are held out from training, but are not external photos.
No external accuracy or successful external experiment is claimed.

The local checkout is Ae-cat/WasteSort, branch `sichen`, initially clean at
`a176f21`. It already contains John's dataset and weighted-loss work. Remote freshness
could not be checked because github.com DNS resolution failed. No push was performed.
The run used the local working implementation added on top of that revision; the
`git_revision` in summary.json identifies the base commit, not a committed script version.

## Reproduced held-out results

CPU inference with the existing `models/resnet18_realwaste.pt`: epoch 5, stored
validation accuracy 76.338%. No retraining or checkpoint modification was performed.
John's `ml/dataset.py` and `ml/splits.csv` supply 3,323 train / 710 val / 719 test
images. Checkpoint class ordering is checked against the dataset.

- Correct: **578 / 719**; accuracy **80.389%**, matching the existing baseline report.
- Errors: **141 / 719**.
- Disposal mapping accuracy: **93.602%** (673 / 719), using the repository's
  Recyclable / Compost / Landfill mapping, not a claim about local disposal rules.

| Class | Correct / support | Recall |
| --- | ---: | ---: |
| Cardboard | 57 / 70 | 81.429% |
| Food Organics | 57 / 63 | 90.476% |
| Glass | 52 / 63 | 82.540% |
| Metal | 102 / 119 | 85.714% |
| Miscellaneous Trash | 43 / 75 | 57.333% |
| Paper | 61 / 75 | 81.333% |
| Plastic | 103 / 139 | 74.101% |
| Textile Trash | 38 / 49 | 77.551% |
| Vegetation | 65 / 66 | 98.485% |

Per-class values are recall: correct predictions divided by actual examples of that
class. Miscellaneous Trash remains weakest. Metal → Plastic occurs 13 times,
Plastic → Metal 12 times, and Paper → Cardboard 12 times. This is a reproduction
of an already examined test set, not a fresh model-selection holdout.

Evidence: [719 prediction records](../reports/week3_held_out_test/predictions.csv)
and [machine-readable metrics and confusion matrix](../reports/week3_held_out_test/summary.json).
The summary records checkpoint/split SHA-256, class order, preprocessing, software
versions and device. Confusion-matrix rows are true classes, columns predicted classes.

## Repeatable commands

Run from the repository root with dependencies from requirements.txt installed.
Use a new output directory for every run; existing results are never overwritten.

```bash
python3 ml/test_images.py --kind realwaste_held_out_test --output-dir reports/held_out_repeat
python3 ml/predict.py --image 'data/realwaste-main/RealWaste/Metal/Metal_1.jpg' --true-label Metal --log reports/single_prediction.csv
python3 -m unittest discover -s tests -v
```

The reusable `ml/prediction_log.py` writes image path, optional true label, predicted
class, softmax confidence, correctness, UTC timestamp, dataset kind, model name,
checkpoint path and checkpoint SHA-256. Missing labels produce blank correctness,
never an assumed failure. Confidence is the model's uncalibrated softmax score,
not a guarantee of correctness. Single-image logs use `single_image_unverified`
and do not claim external provenance.

## Independent external images: collection and pending result template

Copy [the header-only manifest](templates/external_images.csv) beside your external
photos, for example to `data/external_unseen/manifest.csv`. Add one row per photo:

```csv
image_path,true_label,source
photos/can_01.jpg,Metal,New phone photo by collector on YYYY-MM-DD at kitchen
photos/unknown_01.jpg,,New phone photo by collector on YYYY-MM-DD at desk
```

These are illustrative rows, not supplied or tested images. Paths are relative to
the manifest. Use exact class names from the table, or leave true_label empty.
Collect independent new photos, record who/when/where, and label them before viewing
predictions. Include varied lighting/backgrounds and multiple examples per class;
record the sample count and collection limits rather than generalizing a small set.

```bash
python3 ml/test_images.py --kind external_unseen --manifest data/external_unseen/manifest.csv --output-dir reports/external_unseen_run1
```

The runner requires source text and checks SHA-256 against **all three RealWaste
splits**, rejecting exact copies and duplicate external files. All RealWaste files
must therefore be available for this check. Hashes cannot detect crops, re-encodes,
or different photos of the same scene; the collector must verify independence.
An empty manifest, unknown label, missing/corrupt image, mismatch of checkpoint class
order, or overlapping split paths fails rather than silently dropping examples.
Accuracy is computed only over labeled images; an entirely unlabeled set has null
accuracy. Results for external images are saved separately, never pooled with test.

| External result field | Current value |
| --- | --- |
| Source and collection dates | Pending collection |
| Total / labeled images | Not collected |
| Accuracy / per-class recall | Not measured |
| Failure examples and lighting/background limits | Pending actual predictions |
| Prediction CSV / summary JSON | Generated by command after photos are supplied |

No new photos were synthesized or borrowed from RealWaste to fill this gap.

## Validation performed

`python3 -m unittest discover -s tests -v`: all 5 tests passed, including dataset
loading, inference, metrics, unlabeled denominators, CSV overwrite protection,
external provenance and renamed RealWaste-copy rejection. The full held-out run
produced 719 unique rows with 578 correct and confidence values in [0, 1].
Single-image CLI logging was also exercised on Metal_1.jpg (Metal, 81.4% confidence);
this is a RealWaste smoke test, not an external result. The empty external manifest
was run through the CLI and correctly failed without producing a result directory.
`git diff --check` passed.
