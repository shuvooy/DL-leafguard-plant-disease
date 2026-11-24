# Methodology

## 1. Label unification

The six datasets name classes differently. `leafguard.py` converts every raw label into a key `crop__condition` with explicit rules:

1. The text is lower-cased and punctuation is removed.
2. The crop is matched from a list with synonyms (for example "bell pepper" and "pepper, bell" become `pepper`; "maize" becomes `corn`; "orange" becomes `citrus`).
3. The word "leaf" is dropped unless it is part of a disease name ("leaf spot", "leaf blight", "leaf mold", "leaf curl", "leaf scorch", "leaf rust").
4. Remaining text is mapped through crop-specific rules, for example corn "rust leaf" and "Common_rust_" both become `corn__common_rust`,
   and "Cassava Mosaic" and "Cassava Mosaic Disease (CMD)" both become `cassava__mosaic_disease`.
5. Unmatched conditions keep their normalised text.

The result is written to `label_map.csv` with the status of each raw label. Any mapping can be changed in `label_overrides.csv` and `prepare` rerun.
Different diseases are never merged: early and late blight, or the same disease on different crops, stay separate (covered by unit tests).

## 2. Dataset preparation (`prepare`)

1. **Scan** the six sources. FieldPlant boxes become one photo per leaf (box enlarged by 8%, boxes under 32 px skipped).
   Plant Pathology keeps single-label photos only.
2. **Cache** every photo as a JPEG of at most 384 px on the long side, and compute a 256-bit difference hash. Unreadable photos are dropped.
3. **Remove duplicates.** Photos with the same hash and the same label keep one copy. Photos with the same hash and different labels are removed entirely.
4. **Guard the test set.** Any training photo within Hamming distance 10 (of 256) of a PlantDoc test photo is removed.
   This catches the same web photo appearing in several sources. It cannot catch every rotated or heavily edited copy.
5. **Drop rare classes** with fewer than 40 photos.
6. **Split.** PlantWild keeps its official folders. PlantDoc's train folder is training data and its test folder is the external test set.
   All other photos use a 10-fold stratified group split (fold 0 test, fold 1 validation, the rest train).
   Groups are the source photo: leaf crops of one FieldPlant photo, and Roboflow augmented copies (`name_jpg.rf.<hash>`), always share a split.

## 3. Training (`train`)

| Setting | Value |
|---|---|
| Model | timm `convnext_tiny`, ImageNet pre-trained, new 109-way head |
| Input | 224 px (training crops from 384 px caches) |
| Augmentation | random resized crop (scale 0.35 to 1), flips, TrivialAugment, colour jitter, Gaussian blur (p 0.15), random erasing |
| Optimiser | AdamW, lr 2e-4 for the backbone and 2e-3 for the head, weight decay 0.05, no decay on biases and norms |
| Schedule | 1 warmup epoch, cosine decay to 1% |
| Regularisation | label smoothing 0.1, dropout 0.2, stochastic depth 0.1, gradient clipping at 1.0 |
| EMA | weights averaged with decay 0.999; the EMA model is evaluated and saved |
| Sampling | 60,000 photos per epoch, weight proportional to 1/sqrt(class size), field photos x1.5 |
| Precision | bf16 mixed precision, TF32 matmuls, channels-last |
| Selection | best validation macro-F1, patience 6 |
| Seed | 42 |

## 4. Evaluation (`evaluate`)

- **Metrics.** Accuracy, macro and weighted F1, macro precision and recall, top-3 and top-5, one-vs-rest macro ROC-AUC, macro mAP, NLL, Brier score, ECE (15 bins).
  Test-time augmentation averages the original and the horizontally flipped photo.
- **Calibration.** A single temperature is fitted on the field-photo part of the validation set by minimising NLL.
- **Abstain threshold.** The lowest calibrated confidence at which answered validation photos reach 95% accuracy. The app also refuses when the largest raw logit falls below the 2nd percentile of validation logits.
- **Breakdowns.** By source, domain and crop; per-class precision, recall, F1 and AUC; confusion matrices; the 40 most frequent confusions.
- **Robustness.** 1,500 test photos under 11 corruptions: blur (radius 2 and 5), dark, bright, low contrast, Gaussian noise (sigma 15 and 35), rotation 30 degrees, JPEG quality 15, low resolution, 25% occlusion.
- **Explainability.** Grad-CAM on the last feature map, for correct and incorrect predictions. The app uses an occlusion heatmap instead, which needs no PyTorch.
- **Speed.** GPU and CPU latency of the PyTorch model.

## 5. Export and app

The EMA weights are exported to ONNX (opset 17, dynamic batch) and checked against PyTorch on 48 test photos. A dynamic int8 version is also produced.
The Streamlit app preprocesses a photo exactly like evaluation (resize shorter side to 255 px, centre crop 224 px, ImageNet normalisation),
applies temperature scaling, and shows the top matches with general guidance for the predicted condition.
Photos are processed in memory and not stored.

## 6. Tests

`pytest` runs label-unification tests, metric and calibration checks, an abstain-threshold check, preprocessing, argument parsing,
the model-download helper, and an end-to-end `prepare` run on small synthetic datasets that checks duplicates are removed, near-copies of the PlantDoc
test set are removed, and grouped photos never cross splits.
