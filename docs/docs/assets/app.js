(() => {
  'use strict';

  const MODEL_URL = './models/best.onnx';
  const LABELS_URL = './models/classes.txt';
  const IMAGE_LIMIT = 12 * 1024 * 1024;
  const INPUT_SIZE = 640;
  const palette = ['#72e6c2', '#9b8cff', '#ffbd69', '#6bbcff', '#ff7e9d', '#c4e86b', '#e6a8ff', '#62d6dc'];

  const $ = (id) => document.getElementById(id);
  const elements = {
    input: $('image-input'), drop: $('drop-zone'), prompt: $('upload-prompt'), preview: $('preview-wrap'),
    canvas: $('result-canvas'), browse: $('browse-button'), remove: $('remove-image'),
    load: $('load-model'), run: $('run-detection'), confidence: $('confidence'), confidenceValue: $('confidence-value'),
    iou: $('iou'), iouValue: $('iou-value'), status: $('status-message'), runtime: $('runtime-label'),
    indicator: $('model-indicator'), summary: $('result-summary'), results: $('results-content')
  };

  let session = null;
  let labels = [];
  let sourceImage = null;
  let detections = [];
  let busy = false;

  function setStatus(message, kind = '') {
    elements.status.textContent = message;
    elements.status.className = `status-message ${kind}`.trim();
  }

  function setBusy(value) {
    busy = value;
    elements.load.disabled = value || Boolean(session);
    elements.run.disabled = value || !session || !sourceImage;
    elements.remove.disabled = value;
    elements.input.disabled = value;
    elements.browse.disabled = value;
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
    if (!file || busy) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setStatus('Please choose a JPG, PNG or WEBP image.', 'error');
      return;
    }
    if (file.size > IMAGE_LIMIT) {
      setStatus('This image is larger than 12 MB. Choose a smaller file.', 'error');
      return;
    }
    const url = URL.createObjectURL(file);
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
    const scale = Math.min(INPUT_SIZE / image.naturalWidth, INPUT_SIZE / image.naturalHeight);
    const width = Math.max(1, roundToEven(image.naturalWidth * scale));
    const height = Math.max(1, roundToEven(image.naturalHeight * scale));
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
    return { tensorData, scale, left, top };
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
      const imageWidth = sourceImage.image.naturalWidth;
      const imageHeight = sourceImage.image.naturalHeight;
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
    if (!session || !sourceImage || busy) return;
    setBusy(true);
    setStatus('Running inference in your browser…');
    const start = performance.now();
    try {
      const transform = preprocess(sourceImage.image);
      const tensor = new window.ort.Tensor('float32', transform.tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const feeds = { [session.inputNames[0]]: tensor };
      const outputs = await session.run(feeds);
      const output = outputs[session.outputNames[0]];
      detections = postprocess(output, transform, Number(elements.confidence.value), Number(elements.iou.value));
      drawImage(detections);
      renderResults(detections);
      const elapsed = Math.round(performance.now() - start);
      setStatus(`Completed in ${elapsed} ms on WebAssembly CPU.`, 'success');
    } catch (error) {
      setStatus(error.message || 'Inference failed.', 'error');
    } finally {
      setBusy(false);
    }
  }

  function clearImage() {
    if (busy) return;
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
  elements.run.disabled = true;
})();
