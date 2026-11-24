# LeafGuard deployment model

- **Run:** `convnext_tiny`
- **Architecture:** ConvNeXt-Tiny
- **Classes:** 109 crop/condition classes
- **Input:** 224 × 224 RGB image
- **Runtime:** ONNX Runtime CPU
- **Included weight file:** `model_int8.onnx` (about 28 MB)
- **Agreement with full-precision ONNX predictions:** 97.92% on the comparison set recorded during export
- **Internal held-out test:** 13,525 photos; 91.14% accuracy and 83.32% macro-F1
- **Held-out PlantDoc test:** 223 photos; 69.96% accuracy and 73.66% macro-F1

The app is a research and education decision-support tool, not a substitute for a plant pathologist or local agricultural extension officer. See [`../MODEL_CARD.md`](../MODEL_CARD.md) and [`../DATASETS.md`](../DATASETS.md) for intended use, limitations, and data licence notes.
