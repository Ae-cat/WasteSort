# WasteSort

WasteSort classifies an image into one of the nine RealWaste classes and maps
the prediction to Recyclable, Compost, or Landfill.

Week 2 setup and test instructions are recorded in
[`docs/week2_results.md`](docs/week2_results.md).
The meeting walkthrough is in [`docs/week2_demo.md`](docs/week2_demo.md).
AI-powered computer vision system for identifying waste and recommending disposal categories.

Evaluate a trained checkpoint on the held-out test split with:

```bash
python3 ml/evaluate.py
```

Week 3 prediction logs, reproduced held-out results, and the independent external
image collection/test workflow are in
[`docs/week3_carryover_results.md`](docs/week3_carryover_results.md).

Week 4 work:
- [Baseline error gallery and visual review](docs/week4_error_analysis.md)
- [Controlled 5-versus-10-epoch experiment](docs/week4_epoch_experiment.md)
