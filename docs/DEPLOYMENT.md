# Deploy LeafGuard on Streamlit Community Cloud

LeafGuard is a Streamlit app whose entry point is `leafguard.py`. The repository already contains the runtime dependencies, Streamlit configuration, 109-class model configuration, compact ONNX model, and sample photos.

## Before deploying

Make sure the repository contains these files at its root:

- `leafguard.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `deploy/model_config.json`
- `deploy/model_int8.onnx`
- `deploy/samples/`

The full-precision ONNX export is optional and is not needed for deployment. The app automatically uses the included INT8 model. No API key, database, or Streamlit secret is required for the current app.

## Community Cloud settings

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Create an app and select the repository and branch containing LeafGuard.
3. Set **Main file path** to `leafguard.py`.
4. In **Advanced settings**, select **Python 3.12** to match the project environment used for validation.
5. Deploy. Community Cloud assigns the app a `*.streamlit.app` URL; you can choose an available subdomain during deployment.

Community Cloud reads the root `requirements.txt`, runs the app from the repository root, and uses `.streamlit/config.toml` for Streamlit settings. Keep the `deploy/` directory in the repository because the app loads its configuration, ONNX model, and sample photos from there.

## After the app is live

1. Open the public `*.streamlit.app` URL and test all three input modes: upload, camera, and sample photo.
2. Test the **Show where the model looked** option. It runs an occlusion heatmap and is slower than standard prediction.
3. Add the final app URL to the **Live demo** line near the top of the root `README.md`.
4. Optionally add the same URL to your GitHub repository's About section.

The app displays general educational guidance only. It is not a substitute for an agronomist or laboratory diagnosis.

## Official documentation

- [Deploy an app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Community Cloud file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
