# Week 5 follow-up / 补充实验结果

All runs: layer4 + fc, no augmentation, seed 42, 5 epochs, batch 32, Adam lr 0.001. Same saved initialization; 710 validation and 719 test images. Checkpoints selected by validation accuracy. BN running statistics continue updating.

| Training data | 9-class | 3-bin | Misc recall | Landfill recall | Recyclable → Landfill |
|---|---:|---:|---:|---:|---:|
| RealWaste only | 87.20% | 96.52% | 78.67% | 118/124 (95.16%) | 13/466 (2.79%) |
| Combined | 85.95% | 95.13% | 73.33% | 100/124 (80.65%) | 8/466 (1.72%) |
| Combined +137 trash | 84.84% | 94.58% | 74.67% | 113/124 (91.13%) | 14/466 (3.00%) |

## 中文结论

RealWaste-only 在本次同种子 layer4 对照中整体最好；combined 没有显示出额外的 RealWaste 测试收益。不能把之前相对冻结模型的提升全归因于新增数据。新增137张trash让Landfill召回从100/124升至113/124，但可回收物误入Landfill从8/466增至14/466；Misc只提高1张（55/75→56/75），总体准确率下降。

## English summary

The RealWaste-only layer4 control achieved the highest overall scores in this single-seed comparison. Extra combined data did not improve this RealWaste test set once layer4 was trainable. Adding 137 TrashNet trash images improved Landfill recall from 100/124 to 113/124, but recyclable-to-Landfill errors increased from 8/466 to 14/466. Misc recall improved by only one image (55/75 to 56/75), while overall accuracy declined.

These are descriptive single-seed results; webcam generalization remains untested. Five epochs represent different numbers of optimizer updates across dataset sizes. Do not claim unfreezing alone fully explains all historical gains. No review threshold applied. Original checkpoints retained.

## Per-class recall

| Class | RealWaste only | Combined | Combined +137 |
|---|---:|---:|---:|
| Cardboard | 95.71% | 87.14% | 97.14% |
| Food Organics | 85.71% | 90.48% | 74.60% |
| Glass | 96.83% | 98.41% | 93.65% |
| Metal | 86.55% | 81.51% | 81.51% |
| Miscellaneous Trash | 78.67% | 73.33% | 74.67% |
| Paper | 88.00% | 89.33% | 81.33% |
| Plastic | 80.58% | 83.45% | 82.01% |
| Textile Trash | 89.80% | 79.59% | 93.88% |
| Vegetation | 92.42% | 96.97% | 93.94% |

## Checkpoints and artifacts

- RealWaste-only: [checkpoint](../reports/week5_realwaste_layer4_seed42/layer4/best.pt), [summary](../reports/week5_realwaste_layer4_seed42/layer4/summary.json), [predictions](../reports/week5_realwaste_layer4_seed42/layer4/predictions.csv), [configuration](../reports/week5_realwaste_layer4_seed42/config.json).
- Combined +137 trash: [checkpoint](../reports/week5_trash_layer4_seed42/layer4/best.pt), [summary](../reports/week5_trash_layer4_seed42/layer4/summary.json), [predictions](../reports/week5_trash_layer4_seed42/layer4/predictions.csv), [configuration](../reports/week5_trash_layer4_seed42/config.json).
