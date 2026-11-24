# Results

One training run, seed 42, ConvNeXt-Tiny (27.9 M parameters), 224 px, RTX 3060. This repository contains a curated summary of the reported run results; the full machine-generated evaluation files and per-image predictions are not bundled. A subset of the summary metrics is stored in `results/summary_metrics.json`, and the included PNG charts are regenerated from those documented summary values by `scripts/build_summary_charts.py`. Evaluation used horizontal-flip test-time augmentation (`evaluate --tta`).

## 1. Training

Early stopping fired at epoch 16 of 20 (no validation macro-F1 gain for 6 epochs). Total 56 min 29 s at about 306 images/s.
The checkpoint used is **epoch 10** (validation macro-F1 0.8246), chosen on validation macro-F1 only.

| Epoch | Train loss | Val loss | Val acc | Val macro-F1 | PlantDoc acc (monitor only) |
|---|---|---|---|---|---|
| 1 | 2.083 | 0.381 | 0.9006 | 0.7774 | 0.5964 |
| 2 | 1.611 | 0.349 | 0.9114 | 0.8031 | 0.6547 |
| 4 | 1.386 | 0.329 | 0.9213 | 0.8063 | 0.6637 |
| 6 | 1.264 | 0.320 | 0.9248 | 0.8190 | 0.6682 |
| 7 | 1.225 | 0.315 | 0.9259 | 0.8221 | 0.6637 |
| **10** | 1.102 | 0.305 | 0.9322 | **0.8246** | 0.6996 |
| 13 | 1.009 | 0.306 | 0.9342 | 0.8231 | 0.7265 |
| 16 | 0.944 | 0.312 | 0.9351 | 0.8192 | 0.7220 |

- Validation loss bottomed at 0.305 (epochs 10 and 11) and then rose while training loss kept falling. More epochs would not help.
- PlantDoc accuracy moved between 0.66 and 0.75 across epochs 5 to 16. With 223 photos that is mostly noise.
  Epoch 11 scored 0.7489 on PlantDoc, but the model was selected on validation macro-F1, so the reported figure is the 0.6996 of epoch 10.

## 2. Final evaluation

| Set | Photos | Accuracy | Macro-F1 | Top-3 | ROC-AUC | mAP | ECE | ECE after temperature scaling |
|---|---|---|---|---|---|---|---|---|
| validation | 11,735 | 0.9331 | 0.8266 | 0.9838 | 0.9951 | 0.8725 | 0.0649 | 0.0213 |
| **test** | 13,525 | **0.9114** | 0.8332 | 0.9749 | 0.9954 | 0.8800 | 0.0598 | 0.0229 |
| **PlantDoc (held out)** | 223 | **0.6996** | 0.7366 | 0.9283 | 0.9885 | 0.8279 | 0.0443 | 0.1150 |

Temperature T = 0.833, fitted on the field-photo part of the validation set.

- Report the **test** row. Validation was used to pick the checkpoint and to fit T.
- Macro-F1 (0.83) is far below accuracy (0.91) because rare classes score lower.
- On PlantDoc the right answer is in the top three 92.8% of the time, against 70.0% for top-1. ROC-AUC is 0.9885, so the ranking is good and the top choice is the weak point.
- **Calibration does not survive the domain shift.** Temperature scaling reduces ECE on validation and test, but on PlantDoc it raises ECE from 0.044 to 0.115.

### Accuracy by source and domain (test set)

| Source | Photos | Accuracy |
|---|---|---|
| PlantVillage (lab) | 5,452 | 0.9956 |
| Plant Pathology 2021 | 1,548 | 0.9832 |
| FieldPlant | 851 | 0.9001 |
| Cassava | 2,134 | 0.8885 |
| PlantWild | 3,540 | 0.7669 |

| Domain | Photos | Accuracy |
|---|---|---|
| lab | 5,452 | 0.9956 |
| **field** | 8,073 | **0.8546** |

Lab photos are 40% of the test set and nearly perfect, which lifts the 91.1% headline. The field figure (85.5%) is the better guide.
Plant Pathology scores 98.3% partly because it covers one crop (apple) with few classes. PlantWild is the hardest source (76.7%).

## 3. Domain gap

| Model | Classes | Train photos | Test accuracy | PlantDoc accuracy | PlantDoc macro-F1 | PlantDoc top-3 |
|---|---|---|---|---|---|---|
| Lab-only (PlantVillage) | 38 | 43,435 | 0.9983 | **0.3453** | 0.3342 | 0.5874 |
| Lab + field (main model) | 109 | 94,531 | 0.9114 | **0.6996** | 0.7366 | 0.9283 |

- Mixing field photos into training adds **+35.4 points** on the same PlantDoc photos.
- The lab-only model got *worse* on PlantDoc while it got better on its own validation set (PlantDoc 0.38 after epoch 1, 0.35 at the end; validation 0.993 to 0.998).
- The lab-only model is badly calibrated on PlantDoc: ECE 0.416, and 0.582 after temperature scaling.
- Caveats: the two models have different class lists and training sizes, so their *test* accuracies are not comparable. Only the PlantDoc comparison is like for like.
  Lab-only was trained for 8 epochs, the main model for 16. PlantDoc has 223 photos (about +/-6 points).

## 4. Robustness (1,500 test photos, 12 conditions)

| Condition | Accuracy | Change vs clean |
|---|---|---|
| clean | 0.9033 | - |
| low contrast | 0.9073 | +0.4 |
| dark (x0.4) | 0.9053 | +0.2 |
| rotate 30 degrees | 0.8893 | -1.4 |
| noise (sigma 15) | 0.8820 | -2.1 |
| bright (x1.8) | 0.8760 | -2.7 |
| blur (radius 2) | 0.8553 | -4.8 |
| JPEG quality 15 | 0.8427 | -6.1 |
| low resolution | 0.8240 | -8.0 |
| occlusion of 25% | 0.8153 | -8.8 |
| noise (sigma 35) | 0.7393 | -16.4 |
| **blur (radius 5)** | **0.4593** | **-44.4** |

The model is robust to lighting, contrast and rotation. **Strong blur is the major failure mode**, and phone photos can be blurry.

## 5. Selective prediction (the abstain feature)

The app answers only when calibrated confidence is at least **0.807** (target: 95% accuracy on answered photos).

| Set | Photos answered | Accuracy on answered | Target met |
|---|---|---|---|
| test | 88.1% | 0.9654 | yes |
| PlantDoc | 62.8% | 0.8357 | **no** |

The threshold is fitted on validation photos from the same sources as training. Real-world photos from a new source are harder,
so on PlantDoc the model abstains on 37% of photos and still gets 16% of the answered ones wrong.

## 6. Per-class results (test set)

Weakest classes by F1 (test photos in brackets):

| Class | Photos | Precision | Recall | F1 |
|---|---|---|---|---|
| Ginger: sheath blight | 14 | 0.625 | 0.357 | 0.455 |
| Cauliflower: alternaria leaf spot | 15 | 0.545 | 0.400 | 0.462 |
| Corn: mildew | 4 | 0.375 | 0.750 | 0.500 |
| Lettuce: mosaic virus | 19 | 0.643 | 0.474 | 0.545 |
| Bean: healthy | 47 | 0.519 | 0.596 | 0.554 |
| Apple: mosaic virus | 34 | 0.696 | 0.471 | 0.561 |
| Plum: healthy | 64 | 0.613 | 0.594 | 0.603 |
| Ginger: leaf spot | 15 | 0.556 | 0.667 | 0.606 |
| Broccoli: downy mildew | 13 | 0.615 | 0.615 | 0.615 |
| Bean: mosaic virus | 25 | 0.567 | 0.680 | 0.618 |

Strongest: apple frog eye leaf spot 0.987 (n=319), apple powdery mildew 0.967, apple scab 0.961, banana panama disease 0.955, apple rust 0.950.

- Weak classes are mostly small ones (4 to 64 test photos). F1 on 4 to 20 photos is very noisy, so do not over-read single classes.
- Several "healthy" classes score low, since healthy leaves are easy to confuse with mild disease.
- Viral and downy-mildew conditions with subtle signs recur in the weak list. Their ROC-AUC is still 0.93 to 0.999, so the ranking is good and the decision threshold is the weak point.

### Most frequent PlantDoc confusions

| Count | True class | Predicted class |
|---|---|---|
| 4 | corn northern leaf blight | corn gray leaf spot |
| 2 | tomato septoria leaf spot | tomato early blight |
| 2 | tomato late blight | tomato early blight |
| 2 | potato early blight | potato late blight |
| 2 | pepper bacterial spot | tomato bacterial spot |
| 2 | cherry healthy | plum healthy |
| 2 | apple scab | cherry leaf spot |
| 2 | apple healthy | apple scab |

All are plausible (similar lesions, or the same disease on a similar crop). Counts are 1 to 4 photos, so treat them as examples, not patterns.
Cross-crop errors show the model sometimes weighs lesion appearance over crop identity.

## 7. Export and speed

| | fp32 ONNX | int8 ONNX |
|---|---|---|
| File size | 107 MB | 27 MB |
| CPU latency (4 threads, 1 photo; export record) | 24.7 ms | 128.6 ms |
| Agreement with fp32 predictions | - | 97.9% |

- fp32 ONNX matches PyTorch: largest logit difference 4.29e-06, 100% identical predictions.
- PyTorch CPU latency is 27.3 ms per photo. GPU: 3.1 ms per photo at batch 1, 966.5 photos/s at batch 64.
- INT8 is about 4x smaller but slower on CPU in this export record (128.6 ms vs 24.7 ms); it remains usable for interactive single-photo inference. The quantiser reported that no pre-processing step was run, which may explain the speed.

## 8. Statistical caveats

- PlantDoc has 223 photos: one photo is 0.45 points, and a 95% interval on 70% is roughly +/-6 points (about 63% to 76%).
- Many classes have fewer than 25 test photos, so per-class F1 is noisy.
- One run and one seed: differences under about 1 point are not meaningful.
- Validation data was used for early stopping, checkpoint choice, and fitting temperature and the abstain threshold, so validation numbers are slightly optimistic. The test set is clean.
