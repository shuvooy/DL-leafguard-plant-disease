# Model card: LeafGuard plant disease classifier

## Model details

- **Architecture:** ConvNeXt-Tiny (timm), ImageNet pre-trained, fine-tuned end to end. 27.9 M parameters. Input 224 x 224 px.
- **Output:** 109 classes, each a crop plus a condition (for example `tomato__late_blight`, `apple__healthy`). Probabilities are temperature-scaled (T = 0.833).
- **Deployment artifact included here:** ONNX INT8 (about 28 MB); its predictions agree with the full-precision ONNX export on 97.92% of comparison predictions. The full-precision model and training checkpoint are not included in this repository.
- **Author:** Sarnajit Paul Shuvo. **Version:** 1.0 (November 2026). **Code licence:** MIT. **Weights:** research and education use, see "Licences".

## Intended use

- Decision support for students, researchers and extension workers: photograph one leaf and get a likely diagnosis, a confidence, and general next steps.
- A learning example of leak-free evaluation, calibration and domain-shift measurement for image classifiers.

## Out of scope

- Replacing an agronomist, a lab test, or a plant-health authority. Do not base spraying, quarantine or sale decisions on this model alone.
- Commercial products (see "Licences").
- Whole fields, fruits, stems, roots, seedlings under a microscope, drone imagery, or photos with several leaves and diseases.
- Crops and diseases outside the 109 classes. The model has no "unknown" class and will name the closest class.

## Training data

Six public datasets (details and licences in [DATASETS.md](DATASETS.md)): PlantVillage (lab photos), PlantWild, PlantDoc, FieldPlant, Cassava Leaf Disease, Plant Pathology 2021.
120,014 photos after cleaning: 45% lab, 55% field. Labels were unified by rules into 109 classes (`leafguard_output/label_map.csv`, generated during preparation).
Duplicates, conflicting-label photos and look-alikes of PlantDoc test photos were removed.
Split: train 94,531, validation 11,735, test 13,525, external PlantDoc test 223.

## Evaluation

| Set | Photos | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|---|
| Internal test, all | 13,525 | 91.1% | 83.3% | 97.5% |
| Internal test, lab photos | 5,452 | 99.6% | | |
| Internal test, **field photos** | 8,073 | **85.5%** | | |
| **PlantDoc test (held out)** | 223 | **70.0%** (about +/-6) | 73.7% | 92.8% |

Robustness: 90.3% clean, 88.9% rotated 30 degrees, 84.3% JPEG quality 15, 81.5% with 25% occlusion, **45.9% under strong blur**.
Calibration: ECE 0.060 to 0.023 on the test set after temperature scaling; worse on PlantDoc (0.044 to 0.115).
Selective prediction: answers 88.1% of test photos at 96.5% accuracy; on PlantDoc 62.8% at 83.6%.
Full details: [docs/RESULTS.md](docs/RESULTS.md).

## Known limitations

- **Domain shift.** Accuracy drops from 99.6% on lab photos to 85.5% on field photos and about 70% on a different real-world source. A lab-only model scored 34.5% on the same PlantDoc photos.
- **Blur and low quality.** Strong blur halves accuracy; low resolution, compression and occlusion each cost 6 to 9 points.
- **Class imbalance and small classes.** Rare classes (under 100 photos) score lower, and several "healthy" classes are confused with mild disease.
- **Look-alike diseases** are confused (corn northern leaf blight and gray leaf spot; early and late blight) and the model sometimes confuses the crop.
- **Confidence outside the training sources is overconfident.** The abstain rule is tuned on validation photos from the training sources and misses its 95% target on PlantDoc.
- **Data coverage.** Public datasets over-represent some crops, regions, cameras and growth stages. Performance on Bangladeshi or other local conditions has not been measured.
- **Single run, single seed.** No variance estimate.
- **Treatment text** is generic and does not name products or doses.

## Ethical considerations

A wrong answer can lead to unnecessary spraying or to missing a real outbreak. The app therefore shows confidence, declines low-confidence photos,
and tells users to confirm with a local agricultural extension officer. Photos are processed in memory and not stored by the app.

## Licences

- **Code:** MIT.
- **Weights:** trained on data that includes non-commercial and no-derivatives terms (PlantVillage CC BY-NC-SA 4.0, PlantWild Attribution-NonCommercial-NoDerivatives).
  Whether those terms apply to a trained model is unsettled. Use the weights for research and education, do not sell them or use them in commercial products,
  and check each dataset licence or ask the authors before any other use. This is not legal advice.
- **Sample photos** in `deploy/samples/` are from PlantDoc (CC BY 4.0, Singh et al., CoDS-COMAD 2020).
