# `deploy/`

Runtime files used by the Streamlit app. The included deployment was exported from the `convnext_tiny` checkpoint and contains 109 classes.

- `model_config.json`: class list, preprocessing values, temperature, abstain threshold, evaluation metadata, and export metadata.
- `model_int8.onnx`: compact INT8 ONNX model (about 28 MB); included in this repository.
- `samples/`: ten sample leaf photos used by the app's sample tab.
- `SHA256SUMS`: SHA-256 checksums for the model, configuration, and sample photos. From the repository root, verify them with `sha256sum -c deploy/SHA256SUMS`.
- `requirements.txt`: lightweight app dependencies (also mirrored in the repository root).
- `MODEL_CARD.md`: deployment model summary; see the root `MODEL_CARD.md` for full details and licence notes.

The optional full-precision model, `model.onnx` (about 112 MB), is not included. The app automatically uses `model_int8.onnx` when the full-precision file is absent, so no additional model download is needed.
