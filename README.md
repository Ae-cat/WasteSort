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
