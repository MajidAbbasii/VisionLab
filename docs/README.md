# VisionLab GitHub Pages demo

**Persian documentation:** [README.fa.md](README.fa.md) · **Model card:** [model-card.html](model-card.html)

This folder contains the static GitHub Pages application. It has no project-side inference server: image, video-frame and webcam inference run locally in the visitor's browser.

## Interactive browser features

- **Image detection:** JPG, PNG and WEBP upload, drag-and-drop, bounding boxes, class labels and confidence scores.
- **Video detection:** MP4, WEBM and OGG input (up to 100 MB and 5 minutes), frame-by-frame inference, annotated canvas preview, stop control and WEBM recording/download when `MediaRecorder` and canvas capture are supported. The generated recording contains video only; input audio is not muxed.
- **Live webcam:** browser camera permission via `getUserMedia`, live frame inference and explicit camera stop/release. HTTPS or localhost is required.
- **Analytics:** latest-frame object count, unique class count, average confidence, inference latency, observed processing FPS, frame count and class distribution.
- **Threshold controls:** confidence and class-aware NMS IoU.

All modes use the same exported YOLO11n ONNX model and the 80-class label map. The current browser execution provider is WebAssembly CPU; WebGPU is not enabled. Video processing can fall behind the source frame rate, so not every source frame is guaranteed to be analyzed. Webcam/video performance depends on the visitor's device and browser.

## Privacy and runtime

- Uploaded images, decoded video frames and webcam frames are processed in the browser and are not sent to a VisionLab inference server.
- `models/best.onnx` is approximately 10.7 MB; `models/classes.txt` contains 80 labels.
- ONNX Runtime Web `1.23.2` and its WebAssembly runtime files are loaded from jsDelivr.
- Downloaded processed video is assembled locally with the browser's MediaRecorder API where supported.

## Run locally

From the repository root, start a static HTTP server (do not open `index.html` as a `file://` URL):

```powershell
py -m http.server 8080 --directory docs
```

Open `http://localhost:8080`. Camera access is available on localhost or an HTTPS origin, subject to browser permission.

## Enable GitHub Pages

1. Open **Settings → Pages** in the repository.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Select the publishing branch and the `/docs` folder, then save.
4. Wait for the Pages deployment workflow to complete. The site URL is shown in Pages settings.

The site uses relative asset paths and supports GitHub Pages project URLs as well as a custom domain.

## Model and project notes

The model is an experimental YOLO11n detector for 80 COCO categories, with a 640×640 input and ONNX opset 17. It is intended for learning and experimentation, not safety-critical decisions. Training was stopped after 10 epochs at the user's request; the selected best expanded-dataset checkpoint is from epoch 1. See [model-card.html](model-card.html) for evaluation, provenance and limitations.

Do not add COCO photographs, full labels or annotation payloads unless their individual redistribution rights have been checked. See [DATASETS.md](DATASETS.md), [COCO-ATTRIBUTION.md](COCO-ATTRIBUTION.md) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Review AGPL Corresponding Source obligations before publishing or redistributing model artifacts.
