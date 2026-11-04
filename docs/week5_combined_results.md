# Week 5: First Model on the Combined Dataset (John)

## Run

Same settings as the Week 3 baseline (frozen ResNet18 backbone, 5 epochs, batch size 32, learning rate 0.001). The only change is the training data: `ml/splits_combined.csv` (11,426 training images instead of 3,323). Validation and test are unchanged.

    python ml/train.py --freeze-backbone --splits ml/splits_combined.csv --data-root data/combined --output models/resnet18_combined.pt
    python ml/evaluate.py --checkpoint models/resnet18_combined.pt --splits ml/splits_combined.csv --data-root data/combined --confusion-matrix reports/combined_confusion_matrix.png

## Results on the unchanged 719-image test set

| Class | Baseline | Combined | Change |
| --- | ---: | ---: | ---: |
| Cardboard | 81.4% | 57.1% | −24.3 |
| Food Organics | 90.5% | 84.1% | −6.4 |
| Glass | 82.5% | 68.3% | −14.3 |
| Metal | 85.7% | 85.7% | 0 |
| Miscellaneous Trash | 57.3% | 49.3% | −8.0 |
| Paper | 81.3% | 81.3% | 0 |
| Plastic | 74.1% | 71.2% | −2.9 |
| Textile Trash | 77.6% | 63.3% | −14.3 |
| Vegetation | 98.5% | 95.5% | −3.0 |
| **Overall** | **80.4%** | **73.6%** | **−6.8** |
| **Bin accuracy** | **93.6%** | **91.0%** | **−2.6** |

Landfill bin: 84 of 124 correct (67.7%, baseline 74.2%). 26 Landfill items went to Recyclable and 14 to Compost.

## Findings

- Training on the combined data with a frozen backbone lowered accuracy on the RealWaste test set. The drop (−6.8) is far larger than run-to-run noise (~1 point).
- The biggest losses match the look of the new images: Cardboard → Paper (20), Glass → Plastic (12), Textile Trash → Miscellaneous Trash (11). The new images are mostly clean items on plain or web backgrounds, while the test set is dirty items on concrete. With only the last layer trainable, the model shifts toward the "clean" look.
- The test set is entirely RealWaste, so it cannot show whether the new data helps on webcam images, which was its purpose.

## Decision and next steps

- Keep the baseline as the current model.
- Next (Frank): train on the combined data with the backbone unfrozen, then compare on both the RealWaste test set and the webcam test set.
- If Textile Trash still drops, the next single change is removing the clothes/shoes images (product photos are least like real textile waste).