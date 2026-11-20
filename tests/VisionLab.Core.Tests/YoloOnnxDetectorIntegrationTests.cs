using OpenCvSharp;
using Xunit;
using VisionLab.Core.Detection;

namespace VisionLab.Core.Tests;

public sealed class YoloOnnxDetectorIntegrationTests
{
    // Locate the checked-in model by walking upward from the test output folder;
    // this keeps the integration tests independent of the runner's working directory.
    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var model = Path.Combine(directory.FullName, "models", "exported", "best.onnx");
            if (File.Exists(model)) return directory.FullName;
            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate models/exported/best.onnx from the test output directory.");
    }

    [Theory]
    // Cover portrait, landscape, square, and very small inputs to exercise
    // letterboxing and verify that returned boxes stay inside source dimensions.
    [InlineData(32, 32)]
    [InlineData(320, 240)]
    [InlineData(1920, 480)]
    [InlineData(480, 1920)]
    [InlineData(2560, 1440)]
    [InlineData(2048, 2048)]
    public void Predict_HandlesDifferentImageShapes_AndReturnsValidBoxes(int width, int height)
    {
        var modelPath = Path.Combine(FindRepositoryRoot(), "models", "exported", "best.onnx");
        using var detector = new YoloOnnxDetector(modelPath, Array.Empty<string>());
        using var image = new Mat(height, width, MatType.CV_8UC3, new Scalar(114, 114, 114));

        var detections = detector.Predict(image);

        Assert.NotNull(detections);
        foreach (var detection in detections)
        {
            Assert.InRange(detection.Confidence, 0f, 1f);
            Assert.True(float.IsFinite(detection.X));
            Assert.True(float.IsFinite(detection.Y));
            Assert.True(detection.Width > 0);
            Assert.True(detection.Height > 0);
            Assert.True(detection.X >= 0 && detection.Y >= 0);
            Assert.True(detection.X + detection.Width <= width);
            Assert.True(detection.Y + detection.Height <= height);
        }
    }

    [Fact]
    public void Predict_RejectsConfidenceOutsideZeroToOne()
    {
        var modelPath = Path.Combine(FindRepositoryRoot(), "models", "exported", "best.onnx");
        using var detector = new YoloOnnxDetector(modelPath, Array.Empty<string>());
        using var image = new Mat(64, 64, MatType.CV_8UC3, Scalar.Black);

        Assert.Throws<ArgumentOutOfRangeException>(() => detector.Predict(image, confidence: 1.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => detector.Predict(image, confidence: float.NaN));
    }

    [Fact]
    public void Predict_RejectsIouOutsideZeroToOne()
    {
        var modelPath = Path.Combine(FindRepositoryRoot(), "models", "exported", "best.onnx");
        using var detector = new YoloOnnxDetector(modelPath, Array.Empty<string>());
        using var image = new Mat(64, 64, MatType.CV_8UC3, Scalar.Black);

        Assert.Throws<ArgumentOutOfRangeException>(() => detector.Predict(image, iou: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => detector.Predict(image, iou: float.NaN));
    }

    [Fact]
    public void Predict_BlankImage_ReturnsNoDetections()
    {
        var modelPath = Path.Combine(FindRepositoryRoot(), "models", "exported", "best.onnx");
        using var detector = new YoloOnnxDetector(modelPath, Array.Empty<string>());
        using var image = new Mat(640, 640, MatType.CV_8UC3, Scalar.Black);

        var detections = detector.Predict(image);

        Assert.Empty(detections);
    }
}
