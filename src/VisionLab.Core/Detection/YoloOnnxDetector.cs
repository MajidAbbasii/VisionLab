using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace VisionLab.Core.Detection;

/// <summary>
/// YOLOv8/YOLO11 ONNX inference for the common raw detection output [1, 4+classes, candidates]
/// (or its transposed equivalent). Export with nms=false and a fixed 640x640 input.
/// </summary>
public sealed class YoloOnnxDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string[] _labels;
    private readonly int _inputWidth;
    private readonly int _inputHeight;

    public YoloOnnxDetector(string modelPath, IReadOnlyList<string> labels,
        int inputWidth = 640, int inputHeight = 640, bool useCuda = false)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("ONNX model not found", modelPath);
        _labels = labels.ToArray();
        _inputWidth = inputWidth;
        _inputHeight = inputHeight;
        using var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (useCuda) options.AppendExecutionProvider_CUDA();
        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public IReadOnlyList<Detection> Predict(Mat image, float confidence = 0.25f, float iou = 0.45f)
    {
        if (image.Empty()) throw new ArgumentException("Input image is empty", nameof(image));
        if (!float.IsFinite(confidence) || confidence is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(confidence), "Confidence must be between 0 and 1.");
        if (!float.IsFinite(iou) || iou is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(iou), "IoU must be between 0 and 1.");
        // Letterbox instead of stretching: preserve aspect ratio like Ultralytics.
        var scale = Math.Min((float)_inputWidth / image.Width, (float)_inputHeight / image.Height);
        var resizedWidth = Math.Max(1, (int)Math.Round(image.Width * scale));
        var resizedHeight = Math.Max(1, (int)Math.Round(image.Height * scale));
        var padLeft = (_inputWidth - resizedWidth) / 2;
        var padTop = (_inputHeight - resizedHeight) / 2;
        using var resized = new Mat();
        Cv2.Resize(image, resized, new Size(resizedWidth, resizedHeight));
        using var letterboxed = new Mat(_inputHeight, _inputWidth, MatType.CV_8UC3, new Scalar(114, 114, 114));
        using (var region = new Mat(letterboxed, new Rect(padLeft, padTop, resizedWidth, resizedHeight)))
            resized.CopyTo(region);
        using var rgb = new Mat();
        Cv2.CvtColor(letterboxed, rgb, ColorConversionCodes.BGR2RGB);
        using var floatImage = new Mat();
        rgb.ConvertTo(floatImage, MatType.CV_32FC3, 1.0 / 255.0);

        // ONNX tensor layout is NCHW, RGB, normalized to [0,1].
        var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });
        for (var y = 0; y < _inputHeight; y++)
        for (var x = 0; x < _inputWidth; x++)
        {
            var pixel = floatImage.At<Vec3f>(y, x);
            tensor[0, 0, y, x] = pixel.Item0;
            tensor[0, 1, y, x] = pixel.Item1;
            tensor[0, 2, y, x] = pixel.Item2;
        }

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
        var output = results.First().AsTensor<float>();
        var dims = output.Dimensions.ToArray();
        if (dims.Length != 3) throw new InvalidOperationException($"Expected rank-3 YOLO output, got [{string.Join(',', dims)}]");
        var channelFirst = dims[1] <= dims[2] && dims[1] >= 5;
        var featureCount = channelFirst ? dims[1] : dims[2];
        var candidateCount = channelFirst ? dims[2] : dims[1];
        var classCount = featureCount - 4;
        if (classCount <= 0) throw new InvalidOperationException("Unexpected YOLO output feature dimension.");

        float Read(int candidate, int feature) => channelFirst
            ? output[0, feature, candidate] : output[0, candidate, feature];

        var proposals = new List<(Rect box, int classId, float score)>();
        for (var candidate = 0; candidate < candidateCount; candidate++)
        {
            var bestClass = 0;
            var bestScore = 0f;
            for (var c = 0; c < classCount; c++)
            {
                var score = Read(candidate, c + 4);
                if (score > bestScore) { bestScore = score; bestClass = c; }
            }
            if (bestScore < confidence) continue;
            var cx = (Read(candidate, 0) - padLeft) / scale;
            var cy = (Read(candidate, 1) - padTop) / scale;
            var width = Read(candidate, 2) / scale;
            var height = Read(candidate, 3) / scale;
            var left = Math.Clamp((int)(cx - width / 2), 0, image.Width - 1);
            var top = Math.Clamp((int)(cy - height / 2), 0, image.Height - 1);
            var right = Math.Clamp((int)(cx + width / 2), left + 1, image.Width);
            var bottom = Math.Clamp((int)(cy + height / 2), top + 1, image.Height);
            proposals.Add((new Rect(left, top, right - left, bottom - top), bestClass, bestScore));
        }

        // Class-aware non-maximum suppression.
        var kept = new List<Detection>();
        foreach (var group in proposals.GroupBy(p => p.classId))
        {
            var boxes = group.Select(p => p.box).ToArray();
            var scores = group.Select(p => p.score).ToArray();
            CvDnn.NMSBoxes(boxes, scores, confidence, iou, out int[] indices);
            foreach (var index in indices)
            {
                var item = group.ElementAt(index);
                var label = item.classId < _labels.Length ? _labels[item.classId] : $"class_{item.classId}";
                kept.Add(new Detection(item.classId, label, item.score, item.box.X, item.box.Y,
                    item.box.Width, item.box.Height));
            }
        }
        return kept;
    }

    public void Dispose() => _session.Dispose();
}



