using System.Diagnostics;
using System.Text.Json;
using OpenCvSharp;
using VisionLab.Core.Detection;
using VisionLab.Core.Video;

static void Usage()
{
    Console.WriteLine("VisionLab .NET 8 ONNX inference");
    Console.WriteLine("  image <image-path|folder> [--model model.onnx] [--labels labels.txt]");
    Console.WriteLine("        [--confidence 0.25] [--iou 0.45] [--output path] [--json path] [--recursive]");
    Console.WriteLine("  video <video-path|camera-index> [--model model.onnx] [--labels labels.txt]");
    Console.WriteLine("        [--confidence 0.25] [--iou 0.45] [--output output.mp4] [--max-frames count] [--no-window]");
}

if (args.Length == 0 || args[0] is "--help" or "-h" or "help") { Usage(); return; }

var command = args[0].ToLowerInvariant();
if (command is not ("image" or "video"))
{
    Usage();
    return;
}

string Option(string name, string fallback)
{
    var index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
}

float Threshold(string name, float fallback)
{
    var raw = Option(name, fallback.ToString(System.Globalization.CultureInfo.InvariantCulture));
    if (!float.TryParse(raw, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var value) || value is < 0 or > 1)
        throw new ArgumentException($"{name} must be a number between 0 and 1.");
    return value;
}

var modelPath = Option("--model", "models/exported/best.onnx");
var labelsPath = Option("--labels", "datasets/classes.txt");
var labels = File.Exists(labelsPath) ? File.ReadAllLines(labelsPath) : Array.Empty<string>();
var confidence = Threshold("--confidence", 0.25f);
var iou = Threshold("--iou", 0.45f);

using var detector = new YoloOnnxDetector(modelPath, labels);

if (command == "image")
{
    if (args.Length < 2) { Usage(); return; }
    var input = Path.GetFullPath(args[1]);
    var recursive = args.Contains("--recursive");
    if (File.Exists(input))
    {
        var output = Option("--output", Path.Combine("results", "predictions", $"{Path.GetFileNameWithoutExtension(input)}-detected.jpg"));
        var json = Option("--json", Path.ChangeExtension(output, ".json"));
        ProcessImage(input, output, json, detector, confidence, iou);
    }
    else if (Directory.Exists(input))
    {
        var outputDirectory = Option("--output", Path.Combine("results", "predictions", Path.GetFileName(input)));
        var search = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var files = Directory.EnumerateFiles(input, "*", search)
            .Where(IsSupportedImage).OrderBy(path => path, StringComparer.OrdinalIgnoreCase).ToArray();
        if (files.Length == 0) throw new InvalidOperationException($"No supported images found in: {input}");

        var jsonDirectory = Option("--json", Path.Combine(outputDirectory, "json"));
        Directory.CreateDirectory(outputDirectory);
        Directory.CreateDirectory(jsonDirectory);
        var succeeded = 0;
        foreach (var file in files)
        {
            var relative = Path.GetRelativePath(input, file);
            var relativeOutput = Path.ChangeExtension(relative, ".jpg");
            var imageOutput = Path.Combine(outputDirectory, relativeOutput);
            var jsonOutput = Path.Combine(jsonDirectory, Path.ChangeExtension(relative, ".json"));
            if (ProcessImage(file, imageOutput, jsonOutput, detector, confidence, iou)) succeeded++;
        }
        Console.WriteLine($"Completed: {succeeded}/{files.Length} images ({files.Length - succeeded} skipped)");
    }
    else throw new FileNotFoundException("Input image or directory does not exist", input);
}
else if (command == "video")
{
    if (args.Length < 2) { Usage(); return; }
    var output = Option("--output", "results/predictions/dotnet-video.mp4");
    var noWindow = args.Contains("--no-window");
    int? maxFrames = null;
    var maxFramesOption = Option("--max-frames", "");
    if (!string.IsNullOrWhiteSpace(maxFramesOption))
    {
        if (!int.TryParse(maxFramesOption, out var parsedMaxFrames) || parsedMaxFrames <= 0)
            throw new ArgumentException("--max-frames must be a positive integer.");
        maxFrames = parsedMaxFrames;
    }
    new VideoProcessor(detector).Run(args[1], output, !noWindow, confidence, iou, maxFrames);
    Console.WriteLine($"Saved: {output}");
}
else Usage();

static bool IsSupportedImage(string path) =>
    new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }
        .Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

static bool ProcessImage(string input, string output, string jsonPath,
    YoloOnnxDetector detector, float confidence, float iou)
{
    using var image = Cv2.ImRead(input);
    if (image.Empty())
    {
        Console.Error.WriteLine($"Skipped unreadable image: {input}");
        return false;
    }

    var timer = Stopwatch.StartNew();
    var detections = detector.Predict(image, confidence, iou);
    timer.Stop();
    foreach (var item in detections)
    {
        var rect = new Rect((int)item.X, (int)item.Y, (int)item.Width, (int)item.Height);
        Cv2.Rectangle(image, rect, Scalar.LimeGreen, 2);
        Cv2.PutText(image, $"{item.Label} {item.Confidence:P0}",
            new Point(rect.X, Math.Max(20, rect.Y - 6)), HersheyFonts.HersheySimplex,
            0.55, Scalar.LimeGreen, 2);
    }

    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(jsonPath))!);
    if (!Cv2.ImWrite(output, image)) throw new IOException($"Could not write annotated image: {output}");

    var report = new
    {
        source = Path.GetFullPath(input),
        output = Path.GetFullPath(output),
        imageWidth = image.Width,
        imageHeight = image.Height,
        confidenceThreshold = confidence,
        iouThreshold = iou,
        inferenceMilliseconds = Math.Round(timer.Elapsed.TotalMilliseconds, 3),
        detectionCount = detections.Count,
        detections
    };
    File.WriteAllText(jsonPath, JsonSerializer.Serialize(report,
        new JsonSerializerOptions { WriteIndented = true }));

    Console.WriteLine($"{Path.GetFileName(input)}: {detections.Count} detections, {timer.Elapsed.TotalMilliseconds:F1} ms");
    Console.WriteLine($"  Image: {output}");
    Console.WriteLine($"  JSON:  {jsonPath}");
    return true;
}
