# Week 4: Bin-Level Accuracy and Compost Safeguard (John)

## Bin-level accuracy

`ml/evaluate.py` now also reports how often items land in the right bin (Recyclable / Compost / Landfill), using `DISPOSAL_MAP` from `ml/model.py`. All results are on the same 719-image held-out test set.

| Model | Class accuracy | Bin accuracy |
| --- | ---: | ---: |
| Baseline (5 epochs, unweighted) | 80.389% | 93.602% |
| Class-weighted (5 epochs) | 79.277% | 92.211% |
| 10-epoch candidate (Frank) | 81.64% | 94.02% |

Bin confusion matrix, baseline (rows = true bin, columns = predicted bin):

| True ↓ / Predicted → | Recyclable | Compost | Landfill | Correct |
| --- | ---: | ---: | ---: | ---: |
| Recyclable (466) | 455 | 1 | 10 | 97.6% |
| Compost (129) | 0 | 126 | 3 | 97.7% |
| Landfill (124) | 21 | 11 | 92 | 74.2% |

Bin confusion matrix, class-weighted model:

| True ↓ / Predicted → | Recyclable | Compost | Landfill | Correct |
| --- | ---: | ---: | ---: | ---: |
| Recyclable (466) | 448 | 3 | 15 | 96.1% |
| Compost (129) | 1 | 124 | 4 | 96.1% |
| Landfill (124) | 18 | 15 | 91 | 73.4% |

- Landfill is the weak bin in both models (~74%). In the baseline, 32 of 124 Landfill items go to the wrong bin (21 into Recyclable, 11 into Compost). That is contamination, the costliest mistake for a sorter.
- Class weighting lowered bin accuracy too (93.6% → 92.2%), which supports keeping the unweighted baseline.
- Miscellaneous Trash has the lowest bin accuracy in both models (65.3% baseline, 62.7% weighted). Every other class is above 87%.

## Miscellaneous Trash → Food Organics errors

Script: `ml/inspect_errors.py` (grid saved to `reports/weighted_misc_as_food.png`)

Miscellaneous Trash predicted as Food Organics: 9 test images in the baseline, 11 in the class-weighted model. The analysis below uses the weighted model's 11.

- **Mostly hard cases:** dirty, earthy, or wet items (soil-like lumps, soggy paper or film, a stained wrapper, a lint ball) that look organic.
- **Possibly mislabeled or mixed:** `_325` looks like onion or garlic peel; `_465` and `_456` are a board covered in grass clippings.
- **Clear model error:** `_201`, a small brown plastic toy.
- **Same object photographed twice:** `_465`/`_456` and `_393`/`_423` (likely also `_304`/`_307`), so the 11 errors are about 8 distinct items.
- **Low confidence:** 9 of the 11 were predicted with 31–55% confidence. Only `_303` (85%) and `_325` (75%) were confident mistakes.

Conclusion: these errors come mostly from what is in the Miscellaneous Trash class, not a fixable model bug. A confidence safeguard fits better than retraining.

## Needs-review threshold

Script: `ml/threshold_sweep.py`. The threshold was chosen on the validation set and confirmed on the test set, using the class-weighted model.

**General threshold (all predictions):** too costly. At 60%, it flags 33% of correct predictions (val). Keep any general threshold low (~40%: catches 36% of wrong-bin errors, flags 8% of correct).

**Compost-only threshold:**

| Threshold | Wrong-bin caught (val) | Correct flagged (val) | Wrong-bin caught (test) | Correct flagged (test) |
| --- | ---: | ---: | ---: | ---: |
| 50% | 7/23 (30%) | 6/126 (5%) | 7/18 (39%) | 8/124 (6%) |
| **60%** | **13/23 (57%)** | **12/126 (10%)** | **15/18 (83%)** | **14/124 (11%)** |
| 70% | 17/23 (74%) | 20/126 (16%) | 15/18 (83%) | 25/124 (20%) |

**Recommendation:** show "Needs review" when the model predicts a Compost class (Food Organics or Vegetation) with less than 60% confidence. This catches most trash-into-compost errors while flagging about 1 in 10 real Compost items. The test catch rate is higher than validation mostly because of small numbers, so treat the validation figure (57%) as the realistic estimate.

**Caveat:** these numbers are for the class-weighted model. Frank is rerunning `ml/threshold_sweep.py` on the 10-epoch candidate, since a more confident model shifts them; use his numbers for the final threshold.