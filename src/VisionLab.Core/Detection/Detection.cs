namespace VisionLab.Core.Detection;

/// <summary>Represents one detected object using pixel coordinates in the source image.</summary>
/// <param name="ClassId">Zero-based class index produced by the model.</param>
/// <param name="Label">Human-readable class name associated with <paramref name="ClassId"/>.</param>
/// <param name="Confidence">Model confidence score in the range [0, 1].</param>
/// <param name="X">Left coordinate of the bounding box.</param>
/// <param name="Y">Top coordinate of the bounding box.</param>
/// <param name="Width">Bounding-box width in pixels.</param>
/// <param name="Height">Bounding-box height in pixels.</param>
public sealed record Detection(int ClassId, string Label, float Confidence,
    float X, float Y, float Width, float Height);

