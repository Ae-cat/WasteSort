# Week 4: Compost review threshold for the 10-epoch candidate

## Decision and scope

John's branch was fetched at `509ec23` and merged into local `sichen` in `50098df`.
The README conflict was resolved by retaining both teams' instructions. Local main
was not changed; no branch was pushed. The pre-existing untracked .DS_Store was
left untouched.

**Recommend 60% for the current 10-epoch candidate, for predicted Compost only.**
The condition is `predicted_bin == "Compost" and top_class_confidence < 0.60`.
Food Organics and Vegetation map to Compost. Confidence means the maximum nine-class
softmax score, not the sum of the two Compost-class probabilities. A score exactly
0.60 passes. This is a review policy, not a new trained model or a calibrated
probability of correctness.

This recommendation is now evaluated on validation and test, but is **not wired
into the image/camera prediction UI**, and is not a team-approved deployment
requirement. Raw classification and bin accuracy are unchanged. No checkpoint was
modified or retrained.

## Validation selection (710 images)

John's original sweep was run first with the default validation split. It was then
run again to export auditable per-image records. The selection was recorded in
[selection.json](../reports/week4_threshold_10epochs/selection.json) before test
confirmation. Our proposed operating criterion was: among John's existing grid,
maximize caught wrong-Compost predictions while flagging no more than 10% of
correct Compost predictions on validation. This is an explicit engineering
tradeoff inspired by the team's low-review-burden goal, not an agreed product SLA.

| Threshold | Wrong Compost caught / 17 | Correct Compost flagged / 126 |
| --- | ---: | ---: |
| 40% | 2 (11.8%) | 2 (1.6%) |
| 50% | 5 (29.4%) | 4 (3.2%) |
| **60%** | **8 (47.1%)** | **8 (6.3%)** |
| 70% | 9 (52.9%) | 16 (12.7%) |
| 80% | 16 (94.1%) | 28 (22.2%) |
| 90% | 17 (100%) | 44 (34.9%) |

Moving from 60% to 70% catches one additional error while flagging eight additional
correct cases. A general 60% threshold across all predictions would flag 151/650
correct-bin validation predictions (23.2%), versus 8 correct predictions for the
Compost-only rule. Therefore retain the narrow rule.

The validation catch rate is 47.1%, not a claim that most errors are caught there.
Of the 17 wrong Compost predictions, 14 are actually Landfill and 6 of those 14 are
caught. The other three wrong Compost predictions are actually Recyclable; two of
those are caught. Correct-Compost cases are not synonymous with Food Organics:
only 4/61 actual Food Organics validation images are flagged by this policy.

## Fixed test confirmation (719 held-out RealWaste images)

Only the locked 0.60 threshold was evaluated on test; no test sweep was used to
select or retune it. This is the already-used RealWaste test split, **not independent
external photographs**.

| Measure | Result |
| --- | ---: |
| Wrong Compost predictions caught | 6 / 10 (60.0%) |
| Wrong Compost predictions still accepted | 4 / 10 |
| Correct Compost predictions flagged | 13 / 123 (10.6%) |
| Total images sent for review | 19 / 719 (2.6%) |
| Images still automatically accepted | 700 / 719 (97.4%) |
| Raw bin accuracy, all images | 676 / 719 (94.02%) |
| Bin accuracy among accepted images only | 663 / 700 (94.71%) |

All ten wrong Compost predictions on this test run are actually Landfill. Six are
flagged; four remain. The 13 correctly predicted Compost items flagged include
8 Food Organics and 5 Vegetation images. Denominators matter: flagged actual Food
Organics is 8/63 (12.7%); flagged actual Compost is 13/129 (10.1%); flagged correctly
predicted Compost is 13/123 (10.6%). These are different quantities.

Review does not automatically correct an error. The 94.71% figure describes only
the accepted subset and must not be reported as improved full-set model accuracy.
No human-review outcomes were measured. Small counts and the absence of external
photos limit generalization; this is a candidate-specific operating recommendation.

## Evidence and reproducible commands

- [Original John-script validation output](../reports/week4_threshold_10epochs/validation_sweep.txt)
- [Validation metrics](../reports/week4_threshold_10epochs/validation/summary.json) and [predictions](../reports/week4_threshold_10epochs/validation/predictions.csv)
- [Fixed test metrics](../reports/week4_threshold_10epochs/test_fixed_060/summary.json) and [review decisions](../reports/week4_threshold_10epochs/test_fixed_060/predictions.csv)

The summaries include checkpoint/split SHA-256, UTC time, split and confidence
semantics. The model is `models/week4_epochs_seed42/best_10epochs.pt`; its weights
remain local and ignored by Git. Use fresh output directories when repeating:

```bash
python3 ml/threshold_sweep.py --checkpoint models/week4_epochs_seed42/best_10epochs.pt --output-dir reports/threshold_val_repeat
python3 ml/threshold_sweep.py --checkpoint models/week4_epochs_seed42/best_10epochs.pt --split test --threshold 0.6 --output-dir reports/threshold_test_repeat
```

John's default sweep behavior is preserved. Added optional fixed-threshold mode,
CSV/JSON export, class-order validation and overwrite protection. Seven unit tests
passed, including strict threshold boundary, Compost-only scope, denominator checks
and empty accepted-set behavior. Existing prediction correctness tests also passed.
