# Week 5: Controlled combined-data experiments

All runs used 11,426 training images, the unchanged 710-image validation set, and the same 719-image RealWaste test set. The rebuilt combined manifest is byte-identical to John’s manifest.

## Settings and parameter freezing

ResNet18 pretrained on ImageNet; shared initial weights; seed 42; 5 epochs; batch size 32; Adam lr 0.001; unweighted cross entropy. Best validation accuracy selects the checkpoint, with earliest epoch winning ties. Test results do not select checkpoints.

- Frozen: freeze conv1, bn1 and layer1–layer4 parameters; train only fc.
- Layer4: train layer4 and fc; earlier parameters remain frozen.
- Augmentation: same layer4 treatment, adding only RandomHorizontalFlip(p=0.5).
- All runs retain BatchNorm running-statistic updates during training, matching John’s train.py.

## Results

| Run | 9-class accuracy | 3-bin accuracy | Selected epoch |
|---|---:|---:|---:|
| frozen | 73.85% | 90.54% | 5 |
| layer4 | 85.95% | 95.13% | 4 |
| layer4_aug | 83.73% | 94.85% | 3 |

| Class recall | Frozen | Layer4 | Layer4 + flip |
|---|---:|---:|---:|
| Cardboard | 60.00% | 87.14% | 88.57% |
| Food Organics | 79.37% | 90.48% | 95.24% |
| Glass | 71.43% | 98.41% | 95.24% |
| Metal | 82.35% | 81.51% | 89.08% |
| Miscellaneous Trash | 38.67% | 73.33% | 68.00% |
| Paper | 77.33% | 89.33% | 85.33% |
| Plastic | 78.42% | 83.45% | 65.47% |
| Textile Trash | 73.47% | 79.59% | 93.88% |
| Vegetation | 96.97% | 96.97% | 93.94% |

Layer4 improves overall accuracy by 12.10 percentage points and bin accuracy by 4.59 points versus the paired frozen run. Horizontal flipping reduces overall accuracy, especially Plastic, but improves Textile and Landfill recall. Landfill-bin recall is 64.52%, 80.65%, and 91.13% respectively.

These are single-seed results, not evidence of statistical stability. Webcam performance remains untested. The original historical baseline (80.4% / 93.6%) was not retrained in this experiment.

## Saved artifacts

- **frozen**: [checkpoint](../reports/week5_controlled_seed42/frozen/best.pt), [summary](../reports/week5_controlled_seed42/frozen/summary.json), [predictions](../reports/week5_controlled_seed42/frozen/predictions.csv), [history](../reports/week5_controlled_seed42/frozen/history.json).
- **layer4**: [checkpoint](../reports/week5_controlled_seed42/layer4/best.pt), [summary](../reports/week5_controlled_seed42/layer4/summary.json), [predictions](../reports/week5_controlled_seed42/layer4/predictions.csv), [history](../reports/week5_controlled_seed42/layer4/history.json).
- **layer4_aug**: [checkpoint](../reports/week5_aug_retry_seed42/layer4_aug/best.pt), [summary](../reports/week5_aug_retry_seed42/layer4_aug/summary.json), [predictions](../reports/week5_aug_retry_seed42/layer4_aug/predictions.csv), [history](../reports/week5_aug_retry_seed42/layer4_aug/history.json).

Augmentation was interrupted after epoch 3, then restarted from the same initial weights for a complete 5-epoch run. Only the completed retry is included in the comparison; the incomplete run is excluded. Original checkpoints remain unchanged.

The runner refuses existing output directories. Run `python ml/week5_experiment.py` to validate data; add `--run` for all three runs. `--mode layer4_aug --initial-state PATH --output-dir NEW_DIR` runs only augmentation from a saved shared initialization. Dependencies are in requirements.txt. Obtain the datasets separately; raw images are not committed.
