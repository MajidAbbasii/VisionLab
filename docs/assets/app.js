(() => {
  'use strict';

  const MODEL_URL = './models/best.onnx';
  const LABELS_URL = './models/classes.txt';
  const IMAGE_LIMIT = 12 * 1024 * 1024;
  const VIDEO_LIMIT = 100 * 1024 * 1024;
  const INPUT_SIZE = 640;
  const palette = ['#72e6c2', '#9b8cff', '#ffbd69', '#6bbcff', '#ff7e9d', '#c4e86b', '#e6a8ff', '#62d6dc'];

  const $ = (id) => document.getElementById(id);
  const elements = {
    input: $('image-input'), drop: $('drop-zone'), prompt: $('upload-prompt'), preview: $('preview-wrap'),
    canvas: $('result-canvas'), browse: $('browse-button'), remove: $('remove-image'),
    load: $('load-model'), run: $('run-detection'), confidence: $('confidence'), confidenceValue: $('confidence-value'),
    iou: $('iou'), iouValue: $('iou-value'), status: $('status-message'), runtime: $('runtime-label'),
    indicator: $('model-indicator'), summary: $('result-summary'), results: $('results-content'),
    videoInput: $('video-input'), browseVideo: $('browse-video'), video: $('video-source'), videoCanvas: $('video-canvas'),
    videoPlaceholder: $('video-placeholder'), processVideo: $('process-video'), stopVideo: $('stop-video'), downloadVideo: $('download-video'), videoStatus: $('video-status'),
    camera: $('camera-source'), cameraCanvas: $('camera-canvas'), cameraPlaceholder: $('camera-placeholder'),
    startCamera: $('start-camera'), stopCamera: $('stop-camera'), cameraStatus: $('camera-status'),
    metricCount: $('metric-count'), metricClasses: $('metric-classes'), metricConfidence: $('metric-confidence'),
    metricLatency: $('metric-latency'), metricFps: $('metric-fps'), metricFrames: $('metric-frames'),
    analyticsSource: $('analytics-source'), classDistribution: $('class-distribution')
  };

  let session = null;
  let labels = [];
  let sourceImage = null;
  let detections = [];
  let busy = false;
  let cameraStream = null;
  let cameraFrame = 0;
  let mediaBusy = false;
  let videoProcessing = false;
  let cameraStarting = false;
  let videoResultUrl = null;
  let videoRecorder = null;
  let videoStopRequested = false;

  function setStatus(message, kind = '') {
    elements.status.textContent = message;
    elements.status.className = `status-message ${kind}`.trim();
  }

  function setBusy(value) {
    busy = value;
    elements.load.disabled = value || Boolean(session);
    elements.run.disabled = value || mediaBusy || videoProcessing || cameraStarting || Boolean(cameraStream) || !session || !sourceImage;
    elements.remove.disabled = value || mediaBusy || videoProcessing || cameraStarting || Boolean(cameraStream);
    elements.input.disabled = value;
    elements.browse.disabled = value;
    elements.processVideo.disabled = value || !session || !elements.video.src || Boolean(cameraStream) || cameraStarting || videoProcessing;
    elements.browseVideo.disabled = value || videoProcessing;
    elements.startCamera.disabled = value || !session || Boolean(cameraStream) || cameraStarting || videoProcessing;
  }

  function roundToEven(value) {
    const floor = Math.floor(value);
    const fraction = value - floor;
    if (Math.abs(fraction - 0.5) < 1e-8) return floor % 2 === 0 ? floor : floor + 1;
    return Math.round(value);
  }

  async function loadModel() {
    if (session || busy) return;
    if (!window.ort) {
      setStatus('The browser runtime could not be loaded. Check your connection and refresh the page.', 'error');
      return;
    }
    setBusy(true);
    elements.runtime.textContent = 'Browser runtime · loading';
    setStatus('Downloading the ONNX model and initializing WebAssembly. This may take a moment…');
    try {
      window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
      window.ort.env.wasm.numThreads = 1;
      const labelsResponse = await fetch(LABELS_URL);
      if (!labelsResponse.ok) throw new Error(`Could not load class labels (${labelsResponse.status}).`);
      labels = (await labelsResponse.text()).split(/\r?\n/).map(x => x.trim()).filter(Boolean);
      if (labels.length !== 80) throw new Error(`Expected 80 class labels, found ${labels.length}.`);
      session = await window.ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      const input = session.inputNames[0];
      const output = session.outputNames[0];
      if (!input || !output) throw new Error('The ONNX model does not expose the expected input/output tensors.');
      elements.indicator.classList.add('ready');
      elements.indicator.title = 'Model ready';
      elements.runtime.textContent = 'Browser runtime · WASM ready';
      elements.processVideo.disabled = !elements.video.src;
      elements.startCamera.disabled = false;
      setMediaStatus(elements.cameraStatus, 'Model ready. Click Start camera and allow camera permission in your browser.');
      setStatus('Model loaded. Choose an image and run detection.', 'success');
    } catch (error) {
      session = null;
      elements.runtime.textContent = 'Browser runtime · not loaded';
      setStatus(error.message || 'Model initialization failed.', 'error');
    } finally {
      setBusy(false);
    }
  }

  function loadImage(file) {
    if (!file || busy || mediaBusy || cameraStarting || cameraStream) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setStatus('Please choose a JPG, PNG or WEBP image.', 'error');
      return;
    }
    if (file.size > IMAGE_LIMIT) {
      setStatus('This image is larger than 12 MB. Choose a smaller file.', 'error');
      return;
    }
    const url = URL.createObjectURL(file);
    elements.videoInput.value = '';
    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth * image.naturalHeight > 20000000) {
        URL.revokeObjectURL(url);
        setStatus('This image is too large to process safely in the browser. Choose an image under 20 megapixels.', 'error');
        return;
      }
      if (sourceImage?.url) URL.revokeObjectURL(sourceImage.url);
      sourceImage = { image, url, name: file.name };
      detections = [];
      elements.prompt.hidden = true;
      elements.preview.hidden = false;
      drawImage([]);
      renderResults([], false);
      setStatus(session ? 'Image ready. Adjust thresholds if needed, then run detection.' : 'Image ready. Load the model to start detection.');
      elements.run.disabled = busy || !session;
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus('The selected image could not be opened.', 'error');
    };
    image.src = url;
  }

  function drawImage(items) {
    if (!sourceImage) return;
    const image = sourceImage.image;
    const canvas = elements.canvas;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    context.drawImage(image, 0, 0);
    const scale = Math.max(1, Math.min(image.naturalWidth, image.naturalHeight) / 520);
    context.lineWidth = Math.max(2, 2.2 * scale);
    context.font = `600 ${Math.max(12, 12 * scale)}px Manrope, sans-serif`;
    items.forEach((item) => {
      const color = palette[item.classId % palette.length];
      context.strokeStyle = color;
      context.fillStyle = color;
      context.strokeRect(item.x, item.y, item.width, item.height);
      const text = `${item.label} ${(item.score * 100).toFixed(0)}%`;
      const padding = 5 * scale;
      const textWidth = context.measureText(text).width;
      const tagHeight = Math.max(19, 19 * scale);
      const tagY = Math.max(0, item.y - tagHeight);
      context.fillRect(item.x, tagY, textWidth + padding * 2, tagHeight);
      context.fillStyle = '#07131b';
      context.fillText(text, item.x + padding, tagY + tagHeight * 0.72);
    });
  }

  function renderResults(items, completed = true) {
    elements.summary.textContent = items.length ? `${items.length} object${items.length === 1 ? '' : 's'} detected` : completed ? '0 objects detected' : 'No detections yet';
    if (!items.length) {
      elements.results.innerHTML = `<div class="empty-results">${completed ? 'No objects matched the current confidence threshold.' : 'Run detection to see object labels and confidence scores.'}</div>`;
      return;
    }
    elements.results.innerHTML = `<div class="result-list">${items.slice().sort((a, b) => b.score - a.score).map(item => {
      const color = palette[item.classId % palette.length];
      return `<div class="result-item"><span class="result-swatch" style="background:${color}"></span><span class="result-info"><strong>${escapeHtml(item.label)}</strong><small>${item.width} × ${item.height} px</small></span><span class="result-score">${(item.score * 100).toFixed(1)}%</span></div>`;
    }).join('')}</div>`;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  }

  function preprocess(image) {
    const dimensions = getMediaSize(image);
    const scale = Math.min(INPUT_SIZE / dimensions.width, INPUT_SIZE / dimensions.height);
    const width = Math.max(1, roundToEven(dimensions.width * scale));
    const height = Math.max(1, roundToEven(dimensions.height * scale));
    const left = Math.floor((INPUT_SIZE - width) / 2);
    const top = Math.floor((INPUT_SIZE - height) / 2);
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = 'rgb(114,114,114)';
    context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(image, left, top, width, height);
    const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const plane = INPUT_SIZE * INPUT_SIZE;
    const tensorData = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const offset = i * 4;
      tensorData[i] = pixels[offset] / 255;
      tensorData[plane + i] = pixels[offset + 1] / 255;
      tensorData[2 * plane + i] = pixels[offset + 2] / 255;
    }
    return { tensorData, scale, left, top, imageWidth: dimensions.width, imageHeight: dimensions.height };
  }

  function getMediaSize(media) {
    return {
      width: media.naturalWidth || media.videoWidth || media.width,
      height: media.naturalHeight || media.videoHeight || media.height
    };
  }

  function intersectionOverUnion(a, b) {
    const left = Math.max(a.x1, b.x1);
    const top = Math.max(a.y1, b.y1);
    const right = Math.min(a.x2, b.x2);
    const bottom = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    return intersection / Math.max(1e-8, areaA + areaB - intersection);
  }

  function postprocess(output, transform, confidence, iou) {
    const dims = output.dims;
    if (dims.length !== 3) throw new Error(`Expected rank-3 YOLO output, received [${dims.join(', ')}].`);
    const channelFirst = dims[1] <= dims[2] && dims[1] >= 5;
    const featureCount = channelFirst ? dims[1] : dims[2];
    const candidateCount = channelFirst ? dims[2] : dims[1];
    const classCount = featureCount - 4;
    if (classCount !== labels.length) throw new Error(`Model has ${classCount} classes, but ${labels.length} labels were loaded.`);
    const read = (candidate, feature) => channelFirst
      ? output.data[feature * candidateCount + candidate]
      : output.data[candidate * featureCount + feature];
    const byClass = new Map();
    for (let i = 0; i < candidateCount; i++) {
      let classId = 0;
      let score = 0;
      for (let c = 0; c < classCount; c++) {
        const candidateScore = read(i, c + 4);
        if (candidateScore > score) { score = candidateScore; classId = c; }
      }
      if (score < confidence) continue;
      const cx = (read(i, 0) - transform.left) / transform.scale;
      const cy = (read(i, 1) - transform.top) / transform.scale;
      const width = read(i, 2) / transform.scale;
      const height = read(i, 3) / transform.scale;
      const imageWidth = transform.imageWidth;
      const imageHeight = transform.imageHeight;
      const x1 = Math.max(0, Math.min(imageWidth - 1, Math.trunc(cx - width / 2)));
      const y1 = Math.max(0, Math.min(imageHeight - 1, Math.trunc(cy - height / 2)));
      const x2 = Math.max(x1 + 1, Math.min(imageWidth, Math.trunc(cx + width / 2)));
      const y2 = Math.max(y1 + 1, Math.min(imageHeight, Math.trunc(cy + height / 2)));
      const box = { classId, score, x1, y1, x2, y2 };
      if (!byClass.has(classId)) byClass.set(classId, []);
      byClass.get(classId).push(box);
    }
    const kept = [];
    for (const candidates of byClass.values()) {
      candidates.sort((a, b) => b.score - a.score);
      while (candidates.length) {
        const best = candidates.shift();
        kept.push({
          classId: best.classId,
          label: labels[best.classId],
          score: best.score,
          x: best.x1,
          y: best.y1,
          width: best.x2 - best.x1,
          height: best.y2 - best.y1
        });
        for (let i = candidates.length - 1; i >= 0; i--) {
          if (intersectionOverUnion(best, candidates[i]) > iou) candidates.splice(i, 1);
        }
      }
    }
    return kept.sort((a, b) => b.score - a.score).slice(0, 300);
  }

  async function runDetection() {
    if (!session || !sourceImage || busy || mediaBusy || videoProcessing || cameraStarting || cameraStream) return;
    setBusy(true);
    setStatus('Running inference in your browser…');
    const start = performance.now();
    try {
      detections = await infer(sourceImage.image);
      drawImage(detections);
      renderResults(detections);
      const elapsed = Math.round(performance.now() - start);
      updateAnalytics(detections, elapsed, 0, 1, 'Single image');
      setStatus(`Completed in ${elapsed} ms on WebAssembly CPU.`, 'success');
    } catch (error) {
      setStatus(error.message || 'Inference failed.', 'error');
    } finally {
      setBusy(false);
    }
  }

  async function infer(media) {
    const transform = preprocess(media);
    const tensor = new window.ort.Tensor('float32', transform.tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    return postprocess(outputs[session.outputNames[0]], transform, Number(elements.confidence.value), Number(elements.iou.value));
  }

  function drawMedia(canvas, media, items) {
    const size = getMediaSize(media);
    if (!size.width || !size.height) return;
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d');
    context.drawImage(media, 0, 0, size.width, size.height);
    const scale = Math.max(1, Math.min(size.width, size.height) / 520);
    context.lineWidth = Math.max(2, 2.2 * scale);
    context.font = `600 ${Math.max(12, 12 * scale)}px Manrope, sans-serif`;
    items.forEach(item => {
      const color = palette[item.classId % palette.length];
      context.strokeStyle = color;
      context.fillStyle = color;
      context.strokeRect(item.x, item.y, item.width, item.height);
      const text = `${item.label} ${(item.score * 100).toFixed(0)}%`;
      const padding = 5 * scale;
      const tagHeight = Math.max(19, 19 * scale);
      const tagY = Math.max(0, item.y - tagHeight);
      context.fillRect(item.x, tagY, context.measureText(text).width + padding * 2, tagHeight);
      context.fillStyle = '#07131b';
      context.fillText(text, item.x + padding, tagY + tagHeight * 0.72);
    });
  }

  function updateAnalytics(items, elapsed, fps, frames, source) {
    const classes = new Set(items.map(item => item.classId));
    const average = items.length ? items.reduce((sum, item) => sum + item.score, 0) / items.length : 0;
    elements.metricCount.textContent = String(items.length);
    elements.metricClasses.textContent = String(classes.size);
    elements.metricConfidence.textContent = items.length ? `${(average * 100).toFixed(1)}%` : '—';
    elements.metricLatency.textContent = `${Math.round(elapsed)} ms`;
    elements.metricFps.textContent = fps > 0 ? fps.toFixed(1) : '—';
    elements.metricFrames.textContent = `Frames processed: ${frames}`;
    elements.analyticsSource.textContent = source;
    const counts = new Map();
    items.forEach(item => counts.set(item.label, (counts.get(item.label) || 0) + 1));
    elements.classDistribution.innerHTML = counts.size
      ? [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([label, count]) => `<span class="class-chip"><strong>${escapeHtml(label)}</strong><small>${count}</small></span>`).join('')
      : '<span class="empty-results">No objects detected in the latest frame.</span>';
  }

  function setMediaStatus(element, message, kind = '') {
    element.textContent = message;
    element.className = `status-message ${kind}`.trim();
  }

  function loadVideo(file) {
    if (!file || mediaBusy) return;
    if (!['video/mp4', 'video/webm', 'video/ogg'].includes(file.type)) {
      setMediaStatus(elements.videoStatus, 'Choose an MP4, WEBM or OGG video.', 'error');
      return;
    }
    if (file.size > VIDEO_LIMIT) {
      setMediaStatus(elements.videoStatus, 'This video is larger than 100 MB. Choose a smaller file.', 'error');
      return;
    }
    const url = URL.createObjectURL(file);
    elements.video.pause();
    if (elements.video.src.startsWith('blob:')) URL.revokeObjectURL(elements.video.src);
    elements.video.src = url;
    elements.video.hidden = false;
    elements.videoCanvas.hidden = true;
    elements.videoPlaceholder.hidden = true;
    elements.processVideo.disabled = !session;
    elements.video.onloadedmetadata = () => {
      if (elements.video.duration > 300) {
        elements.video.pause();
        elements.video.removeAttribute('src');
        elements.video.load();
        URL.revokeObjectURL(url);
        elements.video.hidden = true;
        elements.videoPlaceholder.hidden = false;
        elements.processVideo.disabled = true;
        setMediaStatus(elements.videoStatus, 'Videos longer than 5 minutes are not supported in this browser demo.', 'error');
        return;
      }
      elements.videoCanvas.width = elements.video.videoWidth;
      elements.videoCanvas.height = elements.video.videoHeight;
      const ctx = elements.videoCanvas.getContext('2d');
      ctx.clearRect(0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
      setMediaStatus(elements.videoStatus, `Ready: ${file.name} · ${elements.video.videoWidth}×${elements.video.videoHeight} · ${elements.video.duration.toFixed(1)} sec`, 'success');
      elements.downloadVideo.disabled = true;
    };
    elements.video.onerror = () => {
      URL.revokeObjectURL(url);
      setMediaStatus(elements.videoStatus, 'This browser could not decode the selected video format.', 'error');
    };
  }

  function waitForRecorder(recorder) {
    return new Promise(resolve => recorder.addEventListener('stop', resolve, { once: true }));
  }

  async function processVideo() {
    if (!session || !elements.video.src || mediaBusy || cameraStarting || cameraStream || videoProcessing) return;
    mediaBusy = true;
    videoProcessing = true;
    elements.processVideo.disabled = true;
    elements.stopVideo.disabled = false;
    elements.browseVideo.disabled = true;
    elements.run.disabled = true;
    elements.startCamera.disabled = true;
    elements.input.disabled = true;
    elements.browse.disabled = true;
    elements.remove.disabled = true;
    elements.downloadVideo.disabled = true;
    videoStopRequested = false;
    let recorderDone = null;
    let stream = null;
    let frameCount = 0;
    let totalInference = 0;
    let processingStarted = 0;
    const classTotals = new Map();
    const canvas = elements.videoCanvas;
    try {
      elements.video.hidden = true;
      canvas.hidden = false;
      if (canvas.captureStream && window.MediaRecorder) {
        stream = canvas.captureStream(0);
        if (!stream.getVideoTracks()[0]?.requestFrame) {
          stream.getTracks().forEach(track => track.stop());
          stream = canvas.captureStream(24);
        }
        const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(type => MediaRecorder.isTypeSupported(type));
        videoRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        const chunks = [];
        videoRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
        recorderDone = waitForRecorder(videoRecorder).then(() => {
          if (videoResultUrl) URL.revokeObjectURL(videoResultUrl);
          videoResultUrl = URL.createObjectURL(new Blob(chunks, { type: videoRecorder.mimeType || 'video/webm' }));
          elements.downloadVideo.disabled = false;
        });
        videoRecorder.start(250);
      }
      elements.video.currentTime = 0;
      await elements.video.play();
      processingStarted = performance.now();
      setMediaStatus(elements.videoStatus, 'Processing video frames locally…');
      while (!elements.video.ended && !videoStopRequested) {
        await new Promise(resolve => requestAnimationFrame(resolve));
        if (elements.video.paused || !elements.video.videoWidth) continue;
        const start = performance.now();
        const items = await infer(elements.video);
        const elapsed = performance.now() - start;
        drawMedia(canvas, elements.video, items);
        if (stream?.getVideoTracks()[0]?.requestFrame) stream.getVideoTracks()[0].requestFrame();
        frameCount++;
        totalInference += elapsed;
        items.forEach(item => classTotals.set(item.label, (classTotals.get(item.label) || 0) + 1));
        updateAnalytics(items, elapsed, processingStarted ? frameCount * 1000 / Math.max(1, performance.now() - processingStarted) : 0, frameCount, 'Video file');
      }
      elements.video.pause();
      if (videoRecorder?.state === 'recording') videoRecorder.stop();
      if (recorderDone) await recorderDone;
      const distribution = [...classTotals.entries()].sort((a, b) => b[1] - a[1]).map(([label, count]) => `<span class="class-chip"><strong>${escapeHtml(label)}</strong><small>${count}</small></span>`).join('');
      if (distribution) elements.classDistribution.innerHTML = distribution;
      elements.metricFrames.textContent = `Frames processed: ${frameCount}`;
      setMediaStatus(elements.videoStatus, `Completed · ${frameCount} frames · average inference ${(totalInference / Math.max(1, frameCount)).toFixed(1)} ms/frame${videoResultUrl ? ' · processed WEBM is ready to download' : ' · recording is not supported by this browser'}`, 'success');
    } catch (error) {
      if (videoRecorder?.state === 'recording') videoRecorder.stop();
      if (recorderDone) await recorderDone.catch(() => {});
      setMediaStatus(elements.videoStatus, error.message || 'Video processing failed.', 'error');
    } finally {
      stream?.getTracks().forEach(track => track.stop());
      videoRecorder = null;
      mediaBusy = false;
      videoProcessing = false;
      elements.processVideo.disabled = !session || !elements.video.src;
      elements.stopVideo.disabled = true;
      elements.browseVideo.disabled = false;
      elements.run.disabled = !session || !sourceImage;
      elements.startCamera.disabled = !session;
      elements.input.disabled = false;
      elements.browse.disabled = false;
      elements.remove.disabled = false;
    }
  }

  async function startCamera() {
    if (!session || cameraStream || cameraStarting || mediaBusy || videoProcessing) return;
    cameraStarting = true;
    elements.startCamera.disabled = true;
    elements.processVideo.disabled = true;
    elements.browseVideo.disabled = true;
    elements.run.disabled = true;
    elements.remove.disabled = true;
    elements.input.disabled = true;
    elements.browse.disabled = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraStarting = false;
      elements.startCamera.disabled = !session;
      elements.processVideo.disabled = !elements.video.src;
      elements.browseVideo.disabled = false;
      elements.run.disabled = !session || !sourceImage;
      elements.remove.disabled = false;
      elements.input.disabled = false;
      elements.browse.disabled = false;
      setMediaStatus(elements.cameraStatus, location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1'
        ? 'Camera is blocked because this page is not secure. Open the HTTPS GitHub Pages address or use localhost.'
        : 'Camera API is unavailable in this browser or browsing context. Try an up-to-date Chrome or Edge browser.', 'error');
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
      cameraStarting = false;
      elements.camera.srcObject = cameraStream;
      await elements.camera.play();
      elements.camera.hidden = false;
      elements.cameraPlaceholder.hidden = true;
      elements.startCamera.disabled = true;
      elements.stopCamera.disabled = false;
      elements.run.disabled = true;
      elements.processVideo.disabled = true;
      elements.browseVideo.disabled = true;
      elements.input.disabled = true;
      elements.browse.disabled = true;
      elements.remove.disabled = true;
      setMediaStatus(elements.cameraStatus, 'Camera is running. Frames are processed locally.', 'success');
      const cameraStarted = performance.now();
      const tick = async () => {
        if (!cameraStream) return;
        if (!mediaBusy && elements.camera.videoWidth) {
          mediaBusy = true;
          const start = performance.now();
          try {
            const items = await infer(elements.camera);
            const elapsed = performance.now() - start;
            drawMedia(elements.cameraCanvas, elements.camera, items);
            cameraFrame++;
            updateAnalytics(items, elapsed, cameraFrame * 1000 / Math.max(1, performance.now() - cameraStarted), cameraFrame, 'Live webcam');
          } catch (error) {
            setMediaStatus(elements.cameraStatus, error.message || 'Camera inference failed.', 'error');
            stopCamera();
          } finally {
            mediaBusy = false;
          }
        }
        if (cameraStream) window.requestAnimationFrame(tick);
      };
      cameraFrame = 0;
      window.requestAnimationFrame(tick);
    } catch (error) {
      stopCamera();
      const messages = {
        NotAllowedError: 'Camera permission was denied or blocked. Allow camera access for this site in the browser address-bar settings, then try again.',
        NotFoundError: 'No camera was found. Connect or enable a webcam and try again.',
        NotReadableError: 'The camera is busy or unavailable. Close other apps or tabs using it, then try again.',
        SecurityError: 'The browser blocked camera access for security reasons. Use HTTPS or localhost.',
        AbortError: 'The camera could not start. Try again or restart the browser.',
        OverconstrainedError: 'The requested camera is unavailable. Check that your camera is connected.'
      };
      setMediaStatus(elements.cameraStatus, messages[error.name] || `Could not start the camera (${error.name || 'unknown error'}): ${error.message || 'No additional details'}`, 'error');
    }
  }

  function stopCamera() {
    cameraStarting = false;
    cameraStream?.getTracks().forEach(track => track.stop());
    cameraStream = null;
    elements.camera.pause();
    elements.camera.srcObject = null;
    elements.camera.hidden = true;
    elements.cameraPlaceholder.hidden = false;
    elements.startCamera.disabled = !session;
    elements.stopCamera.disabled = true;
    elements.run.disabled = !session || !sourceImage;
    elements.processVideo.disabled = !session || !elements.video.src;
    elements.browseVideo.disabled = false;
    elements.input.disabled = false;
    elements.browse.disabled = false;
    elements.remove.disabled = false;
    elements.remove.disabled = false;
    setMediaStatus(elements.cameraStatus, 'Camera stopped.');
  }

  function clearImage() {
    if (busy || mediaBusy || videoProcessing || cameraStarting || cameraStream) return;
    if (sourceImage?.url) URL.revokeObjectURL(sourceImage.url);
    sourceImage = null;
    detections = [];
    elements.input.value = '';
    elements.preview.hidden = true;
    elements.prompt.hidden = false;
    elements.summary.textContent = 'Waiting for an image';
    elements.results.innerHTML = '<div class="empty-results">Detected objects and confidence scores will appear here.</div>';
    setStatus(session ? 'Model loaded. Choose an image to continue.' : 'Load the model to get started.');
    elements.run.disabled = busy || !session;
  }

  elements.browse.addEventListener('click', () => elements.input.click());
  elements.input.addEventListener('change', event => loadImage(event.target.files?.[0]));
  elements.remove.addEventListener('click', clearImage);
  elements.load.addEventListener('click', loadModel);
  elements.run.addEventListener('click', runDetection);
  elements.confidence.addEventListener('input', () => { elements.confidenceValue.value = Number(elements.confidence.value).toFixed(2); });
  elements.iou.addEventListener('input', () => { elements.iouValue.value = Number(elements.iou.value).toFixed(2); });
  ['dragenter', 'dragover'].forEach(name => elements.drop.addEventListener(name, event => { event.preventDefault(); elements.drop.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(name => elements.drop.addEventListener(name, event => { event.preventDefault(); elements.drop.classList.remove('drag-over'); }));
  elements.drop.addEventListener('drop', event => loadImage(event.dataTransfer?.files?.[0]));
  elements.browseVideo.addEventListener('click', () => elements.videoInput.click());
  elements.videoInput.addEventListener('change', event => loadVideo(event.target.files?.[0]));
  elements.processVideo.addEventListener('click', processVideo);
  elements.stopVideo.addEventListener('click', () => { videoStopRequested = true; elements.video.pause(); });
  elements.downloadVideo.addEventListener('click', () => {
    if (!videoResultUrl) return;
    const link = document.createElement('a');
    link.href = videoResultUrl;
    link.download = 'visionlab-detection.webm';
    link.click();
  });
  elements.startCamera.addEventListener('click', startCamera);
  elements.stopCamera.addEventListener('click', stopCamera);
  window.addEventListener('beforeunload', () => {
    stopCamera();
    if (sourceImage?.url) URL.revokeObjectURL(sourceImage.url);
    if (videoResultUrl) URL.revokeObjectURL(videoResultUrl);
    if (elements.video.src.startsWith('blob:')) URL.revokeObjectURL(elements.video.src);
  });
  elements.run.disabled = true;
  elements.processVideo.disabled = true;
  elements.startCamera.disabled = true;
  elements.stopCamera.disabled = true;
  elements.stopVideo.disabled = true;
})();
