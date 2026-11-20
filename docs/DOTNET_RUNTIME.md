# VisionLab .NET 8 Runtime Package

This folder is the framework-dependent Windows release output. Install the **.NET 8 Runtime** (`Microsoft.NETCore.App`) to run it. The .NET 8 SDK is needed only to build from source.

## Folder layout

```text
dotnet-app/
  VisionLab.App.exe
  VisionLab.App.dll
  VisionLab.Core.dll
  models/exported/best.onnx
  datasets/classes.txt
```

Keep the working directory at the `dotnet-app` folder so the default relative model and class-name paths resolve.

## Run

```powershell
.\VisionLab.App.exe --help
.\VisionLab.App.exe image C:\path\to\image.jpg
.\VisionLab.App.exe image C:\path\to\images --output C:\path\to\output
.\VisionLab.App.exe video C:\path\to\video.mp4 --output C:\path\to\annotated.mp4 --no-window
.\VisionLab.App.exe video 0 --max-frames 120 --output C:\path\to\webcam-test.mp4
```

The current package uses CPU ONNX Runtime. Webcam use requires an accessible camera and OS permission. Video preview requires a graphical desktop; use `--no-window` on headless systems.

The OpenCvSharp native runtime included in this candidate is Windows-specific. Linux/macOS publishing is not covered by this package.
