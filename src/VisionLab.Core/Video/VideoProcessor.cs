using OpenCvSharp;
using VisionLab.Core.Detection;

namespace VisionLab.Core.Video;

/// <summary>Processes a video file or camera and optionally writes annotated frames.</summary>
public sealed class VideoProcessor(YoloOnnxDetector detector)
{
    public void Run(string source, string? outputPath = null, bool showWindow = true,
        float confidence = 0.25f, float iou = 0.45f, int? maxFrames = null)
    {
        var camera = int.TryParse(source, out var cameraIndex);
        using var capture = camera ? new VideoCapture(cameraIndex) : new VideoCapture(source);
        if (!capture.IsOpened()) throw new InvalidOperationException($"Cannot open video source: {source}");
        VideoWriter? writer = null;
        using var frame = new Mat();
        var stopRequested = false;
        var processedFrames = 0;
        ConsoleCancelEventHandler cancelHandler = (_, eventArgs) =>
        {
            // Let the loop exit normally so the video writer can finalize the MP4 container.
            eventArgs.Cancel = true;
            stopRequested = true;
        };
        Console.CancelKeyPress += cancelHandler;
        try
        {
            while (!stopRequested && capture.Read(frame))
            {
                if (frame.Empty()) break;
                foreach (var item in detector.Predict(frame, confidence, iou))
                {
                    var rect = new Rect((int)item.X, (int)item.Y, (int)item.Width, (int)item.Height);
                    Cv2.Rectangle(frame, rect, Scalar.LimeGreen, 2);
                    Cv2.PutText(frame, $"{item.Label} {item.Confidence:P0}",
                        new Point(rect.X, Math.Max(20, rect.Y - 6)), HersheyFonts.HersheySimplex,
                        0.55, Scalar.LimeGreen, 2);
                }
                if (outputPath is not null && writer is null)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
                    var fps = capture.Get(VideoCaptureProperties.Fps);
                    if (fps <= 0 || double.IsNaN(fps)) fps = 30;
                    writer = new VideoWriter(outputPath, FourCC.MP4V, fps,
                        new Size(frame.Width, frame.Height));
                    if (!writer.IsOpened()) throw new InvalidOperationException($"Cannot create output: {outputPath}");
                }
                writer?.Write(frame);
                processedFrames++;
                if (maxFrames is > 0 && processedFrames >= maxFrames.Value) break;
                if (showWindow)
                {
                    Cv2.ImShow("VisionLab - press Q to quit", frame);
                    if (Cv2.WaitKey(1) is 'q' or 'Q') break;
                }
            }
        }
        finally
        {
            writer?.Release();
            writer?.Dispose();
            Console.CancelKeyPress -= cancelHandler;
            if (showWindow) Cv2.DestroyAllWindows();
        }
    }
}

