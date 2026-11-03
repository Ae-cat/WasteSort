# Week 3 Prediction Log

## Status and artifact discovery

**Pending, checked November 3, 2026.**
No predictions or repeat agreement are claimed in this log.
`models/resnet18_realwaste.pt` and the local `data/realwaste-main/RealWaste/` directory are absent.
The existing command below was attempted and exited with `FileNotFoundError` for `models/resnet18_realwaste.pt` before inference:

```sh
.venv/bin/python ml/predict.py --image 'data/realwaste-main/RealWaste/Cardboard/Cardboard_101.jpg'
```

The repository's [data plan](data_plan.md) identifies RealWaste.
The official [UCI dataset page](https://archive.ics.uci.edu/dataset/908/realwaste) provides the [656.6 MB archive](https://archive.ics.uci.edu/static/public/908/realwaste.zip).
Extract it under `data/` to preserve the paths recorded in `ml/splits.csv`.
Downloading the dataset is deferred until the trained checkpoint is available.

Repository documentation, source, all locally available Git history, and the project, Downloads, Desktop, and Documents locations were checked for checkpoint files or a checkpoint download URL.
No `.pt` or `.pth` checkpoint was found in those locations, and no trained-checkpoint URL was recorded in the repository.
`ml/predict_ariella.py` explicitly directs teammates to obtain the ignored checkpoint from the model owner separately.
A GitHub release-asset lookup could not connect to `api.github.com`, so remote release availability is unverified.
Obtain the existing team trained checkpoint from its owner, with its identity or checksum, rather than retraining or substituting ImageNet weights.
Checkpoint filename, SHA-256, training identity, and class order: **pending**.

## Deterministic test inputs

Selection uses only the committed `ml/splits.csv` rows with `split == "test"`.
Sort class labels lexicographically, select the lexicographically smallest path in each class, then append the smallest unselected path.
This produces ten distinct inputs covering all nine classes out of 719 test rows.
All ten selected image files were absent on the laptop when checked.
Paths below are relative to `data/realwaste-main/RealWaste/`.

| # | True class | Image path | Predicted class | Category | Confidence |
|---|---|---|---|---|---|
| 1 | Cardboard | `Cardboard/Cardboard_101.jpg` | Pending | Pending | Pending |
| 2 | Food Organics | `Food Organics/Food Organics_1.jpg` | Pending | Pending | Pending |
| 3 | Glass | `Glass/Glass_104.jpg` | Pending | Pending | Pending |
| 4 | Metal | `Metal/Metal_100.jpg` | Pending | Pending | Pending |
| 5 | Miscellaneous Trash | `Miscellaneous Trash/Miscellaneous Trash_110.jpg` | Pending | Pending | Pending |
| 6 | Paper | `Paper/Paper_103.jpg` | Pending | Pending | Pending |
| 7 | Plastic | `Plastic/Plastic_104.jpg` | Pending | Pending | Pending |
| 8 | Textile Trash | `Textile Trash/Textile Trash_110.jpg` | Pending | Pending | Pending |
| 9 | Vegetation | `Vegetation/Vegetation_108.jpg` | Pending | Pending | Pending |
| 10 | Cardboard | `Cardboard/Cardboard_11.jpg` | Pending | Pending | Pending |

Reproduce the selection and run the existing command once per image, then five times for image 1, after receiving both artifacts:

```sh
shasum -a 256 models/resnet18_realwaste.pt
.venv/bin/python - <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

with Path("ml/splits.csv").open(newline="") as source:
    rows = [r for r in csv.DictReader(source) if r["split"] == "test"]
classes = sorted({r["label"] for r in rows})
selected = [min((r for r in rows if r["label"] == label),
                key=lambda r: r["path"]) for label in classes]
selected.append(min((r for r in rows if r not in selected),
                    key=lambda r: r["path"]))
assert len(classes) == 9 and len({r["path"] for r in selected}) == 10
for row in selected + [selected[0]] * 5:
    subprocess.run([sys.executable, "ml/predict.py", "--image", row["path"],
                    "--checkpoint", "models/resnet18_realwaste.pt"], check=True)
PY
```

`ml/predict.py` retains its existing automatic device selection; record the device used when filling this log.
The separate laptop live latency run must use CPU.

## Repeated prediction of image 1

| Repeat | Predicted class | Category | Confidence |
|---|---|---|---|
| 1 | Pending | Pending | Pending |
| 2 | Pending | Pending | Pending |
| 3 | Pending | Pending | Pending |
| 4 | Pending | Pending | Pending |
| 5 | Pending | Pending | Pending |

Five-run agreement: **pending** for label, category, and confidence.
These selected dataset images do not replace physical item testing.
Adam still needs the Section 5 item assignment and must photograph the actual assigned objects; no physical photographs or physical-test results were produced.
