using OpenCvSharp;
using VisionLab.Core.Detection;

namespace VisionLab.Core.Video;

/// <summary>Processes a video file or camera and optionally writes annotated frames.</summary>
public sealed class VideoProcessor(YoloOnnxDetector detector)
{
    /// <summary>
    /// Reads frames from a video file or camera, annotates each frame, and optionally
    /// displays or writes the processed stream. A numeric source selects a camera index.
    /// </summary>
    /// <param name="source">Video file path or camera index expressed as text.</param>
    /// <param name="outputPath">Optional path for an MP4 output file; null disables writing.</param>
    /// <param name="showWindow">Whether to display frames and allow quitting with Q.</param>
    /// <param name="confidence">Minimum detection confidence.</param>
    /// <param name="iou">IoU threshold used during non-maximum suppression.</param>
    /// <param name="maxFrames">Optional positive frame limit; null processes until the source ends.</param>
    public void Run(string source, string? outputPath = null, bool showWindow = true,
        float confidence = 0.25f, float iou = 0.45f, int? maxFrames = null)
    {
        // Accept a numeric source such as "0" as a camera index; all other values
        // are treated as file paths and passed directly to OpenCV.
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
                // Create the writer lazily so its dimensions come from the first
                // decoded frame rather than assuming a particular source resolution.
                if (outputPath is not null && writer is null)
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
                    // Some cameras/codecs report zero or an invalid FPS; use a
                    // conservative fallback so the output writer can still initialize.
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
            // Release native video resources even when inference, writing, or UI
            // handling throws. Detach the Ctrl+C handler to avoid retaining this run.
            writer?.Release();
            writer?.Dispose();
            Console.CancelKeyPress -= cancelHandler;
            if (showWindow) Cv2.DestroyAllWindows();
        }
    }
}

