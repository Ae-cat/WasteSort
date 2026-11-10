# Week 5 controlled experiment — preparation status

John branch `5eedb5b` merged locally into sichen as `131f937`. Backup branch:
`backup/sichen-before-week5-20261103`. No remote push. README conflicts resolved
by retaining local Week 3/4 documentation links.

## Data and environment

- Original RealWaste: 3323 train, 710 validation, 719 test; all referenced files exist.
- John's combined manifest: 11426 train, 710 validation, 719 test. Validation/test
  ordered (label, filename) pairs exactly match the original manifest.
- TrashNet extracted from `/Users/qisichen/Downloads/dataset-resized.zip`.
- Kaggle archive downloaded from the link supplied by John:
  https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- Combined data built successfully: 11426 / 710 / 719. Regenerated CSV is
  byte-identical to John's committed manifest. Full preflight passed.
- Existing interpreter: `/Users/qisichen/WasteSort/.venv/bin/python`.
- Missing imagehash and PyWavelets packages are available locally in `.week5-deps`;
  existing NumPy/Pillow/SciPy are retained. Use `PYTHONPATH=.week5-deps`.
- Current process reports MPS unavailable; the runner records the actual device.

## Once the missing archive is supplied

Extract it so the source directory contains biological, cardboard, paper, plastic,
metal, white-glass, green-glass, brown-glass, clothes, and shoes folders.
From the repository root run:

```sh
PYTHONPATH=.week5-deps /Users/qisichen/WasteSort/.venv/bin/python ml/build_combined.py
PYTHONPATH=.week5-deps /Users/qisichen/WasteSort/.venv/bin/python ml/week5_experiment.py
PYTHONPATH=.week5-deps /Users/qisichen/WasteSort/.venv/bin/python ml/week5_experiment.py --run
```

The first command now refuses missing source roots. The second performs preflight
without training: expected counts, fixed validation/test identities, unique file
identities, image readability, exact cross-split duplicate checks, and unchanged
RealWaste image bytes. Any failure must be resolved before training.

## Controlled design

1. Frozen backbone + fc, reproducing John's settings with a recorded seed.
2. Only layer4 + fc trainable; otherwise identical settings.
3. Same layer4 + fc treatment, adding only RandomHorizontalFlip(p=0.5).

All runs: pretrained ResNet18, shared initial model state, seed 42, 5 epochs,
batch size 32, Adam lr 0.001, unweighted cross entropy, unchanged validation/test
preprocessing. Training order uses a separate generator so augmentation does not
change sample ordering. BatchNorm running statistics update in all training runs,
matching John's train.py. This is parameter freezing, not fully fixed features.
Checkpoint selection uses validation accuracy only, earliest epoch on ties.
Test results never select configurations/checkpoints. Single-seed differences
are descriptive, not a statistical significance claim.

Outputs go to a new `reports/week5_controlled_seed42` directory; an existing output
directory is refused. Each run records best checkpoint, epoch history, trainable
parameter names, test predictions, 9-class/3-bin accuracy, per-class recall,
per-class bin accuracy, and confusion matrices. No confidence/review threshold is
applied. Global metadata includes settings, versions, source hashes and split hash.

John's reported 73.6% / 91.0% combined and 80.4% / 93.6% original scores remain
historical references, not newly reproduced results. John's combined checkpoint
is absent. Webcam comparison also requires a separately collected labeled dataset;
RealWaste alone cannot establish webcam benefit.

## Protected original checkpoint hashes (SHA-256)

- models/resnet18_realwaste.pt: ec029f793f7ee5e126eda72e9b8bb6de6cba3aa5502d20d20dcab10be5e3b17f
- models/week4_epochs_seed42/best_5epochs.pt: afef8b74e2a29d69ac90acfb11fb2efaabc9ea15d71edea9b85bf9637afd03f3
- models/week4_epochs_seed42/best_10epochs.pt: 49e9e38820e7d6c1219b667eb10f1bd66ffec03aab00f26f09b7e21a2f3f3e73
