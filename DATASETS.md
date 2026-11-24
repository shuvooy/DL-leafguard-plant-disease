# Datasets

**The datasets are not included in this repository and should not be re-uploaded.**
Download each one from its original source, accept its terms, and point `LEAFGUARD_ROOT` at the folder that holds them.

Licence notes reflect the source pages checked when the data was downloaded. Re-check the linked terms before relying on them.

| Dataset | Role | Usable photos after cleaning | Source | Licence shown |
|---|---|---|---|---|
| PlantVillage (`color` folder) | lab photos, 38 classes | 54,280 | [Original GitHub dataset](https://github.com/spMohanty/PlantVillage-Dataset) · [2016 paper](https://doi.org/10.3389/fpls.2016.01419) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| PlantWild | field photos, 89 classes | 17,800 | [Authors' project page](https://tqwei05.github.io/PlantWild/) · [Hugging Face dataset](https://huggingface.co/datasets/uqtwei2/PlantWild) · [2024 paper](https://arxiv.org/abs/2408.03120) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| PlantDoc (cropped version) | field photos; **test split held out** | 2,368 (2,145 train, 223 test) | [Original GitHub dataset](https://github.com/pratikkayal/PlantDoc-Dataset) · [2020 paper](https://doi.org/10.1145/3371158.3371196) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| FieldPlant (YOLO format) | field photos, leaves cropped from boxes | 8,494 | [Roboflow Universe v11](https://universe.roboflow.com/plant-disease-detection/fieldplant/dataset/11) · [2023 IEEE Access paper](https://doi.org/10.1109/ACCESS.2023.3263042) | CC BY 4.0 on Roboflow, "Unknown" on Kaggle |
| Cassava Leaf Disease Classification | field photos, 5 classes | 21,397 | [Kaggle competition and dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) | [Competition rules](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/rules) |
| Plant Pathology 2021 - FGVC8 | field photos of apple leaves, single-label only | 15,675 | [Kaggle competition and dataset](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8) · [FGVC8 project page](https://sites.google.com/view/fgvc8/competitions/plantpathologychallenge2021) | [Competition rules](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/rules) |

Total: 120,014 photos in 109 classes after removing duplicates and classes with fewer than 40 photos.

## Expected folder layout

```
<LEAFGUARD_ROOT>/
  archive/plantvillage dataset/color/<class>/*.jpg     PlantVillage
  archive (1)/{train,val,test}/<class>/*.jpg           PlantWild
  archive (2)/{data.yaml,train/images,train/labels}    FieldPlant
  cassava-leaf-disease-classification/{train.csv,label_num_to_disease_map.json,train_images/}
  PlantDoc-Dataset-master/{train,test}/<class>/*.jpg
  plant-pathology-2021-fgvc8/{train.csv,train_images/}
```

These are the default Kaggle extraction names. If yours differ, edit `PATHS` near the top of `leafguard.py`.
`python leafguard.py check` reports which folders it finds.

## What was changed

- PlantVillage: only the colour images are used (the grayscale and segmented copies would duplicate every photo).
- FieldPlant: each annotated leaf box is cropped into its own classification photo.
- Plant Pathology 2021: photos with more than one disease label, and the "complex" label, are skipped (2,957 photos).
- All sources: images are resized to at most 384 px on the long side for fast training. Labels are unified into `crop__condition`.
  The training pipeline generates the full mapping in `leafguard_output/label_map.csv`; the deployed class list is in `deploy/model_config.json`.
- 229 exact duplicates and 745 photos that appeared with conflicting labels were removed.
  11 training photos that looked like PlantDoc test photos were removed.

## Sample photos in this repository

`deploy/samples/` holds ten resized photos from the PlantDoc test split, used only for the app's "Try a sample" tab.
PlantDoc is CC BY 4.0: credit Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", CoDS-COMAD 2020.

## Model licence note

The trained weights come from data that includes non-commercial and no-derivatives terms (PlantVillage, PlantWild).
Whether those terms reach a trained model is not settled. Until you have checked each licence, or asked the dataset authors,
share the weights for research and education only, and do not sell them or use them in a commercial product.
This is not legal advice.
