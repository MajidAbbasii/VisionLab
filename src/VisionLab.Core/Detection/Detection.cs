namespace VisionLab.Core.Detection;

/// <summary>A detected object in source-image pixel coordinates.</summary>
public sealed record Detection(int ClassId, string Label, float Confidence,
    float X, float Y, float Width, float Height);

