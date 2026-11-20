# VisionLab GitHub Pages site

This folder is the static GitHub Pages site. It has no server-side component.

## Enable GitHub Pages

1. Push the repository to GitHub after reviewing the release and model licensing notes in `../MODEL_CARD.md` and `../THIRD-PARTY-NOTICES.md`.
2. Open **Settings → Pages** in the repository.
3. Under **Build and deployment**, select **Deploy from a branch**.
4. Select the default branch and the `/docs` folder, then save.
5. Wait for the Pages deployment workflow to complete. The site URL will be shown in the Pages settings.

The site uses relative asset paths, so it works both at a GitHub Pages project URL (for example `/repository/`) and when hosted at a custom domain.

## Interactive demo

- `models/best.onnx` is the exported YOLO11n model used by the demo.
- `models/classes.txt` is the 80-class label map.
- `assets/app.js` implements letterbox preprocessing, ONNX Runtime Web inference, class-aware NMS and canvas rendering.
- The WebAssembly-only ONNX Runtime Web `1.23.2` build and its WASM binary are loaded from jsDelivr. The model and uploaded images are processed in the visitor's browser; uploaded images are not sent to a project server.
- The current baseline uses the WebAssembly CPU execution provider. WebGPU is intentionally not enabled until operator coverage and browser compatibility are verified.

## Run locally

From the repository root, start a static HTTP server (do not open `index.html` as a `file://` URL):

```powershell
py -m http.server 8080 --directory docs
```

Then open `http://localhost:8080` in a modern browser. The ONNX model is approximately 10.7 MB; the ONNX Runtime Web JavaScript and WebAssembly runtime files are downloaded separately from jsDelivr.

## Important release notes

- Do not add COCO photographs, labels or annotation payloads to this site unless their individual redistribution rights have been checked. See `../DATASETS.md`.
- The selected base checkpoint's original download event and upstream checksum were not retained. The provenance and local SHA-256 are recorded in `../MODEL_CARD.md`.
- Review all third-party notices and AGPL Corresponding Source obligations before publishing the repository or serving the model to visitors.
