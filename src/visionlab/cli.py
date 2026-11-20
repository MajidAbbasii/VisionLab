"""Command-line entry point for detection and experiment workflows."""
from __future__ import annotations
import argparse
from .detector import detect_image, detect_video

def main() -> None:
    """Dispatch image or video/webcam inference from command-line arguments."""
    parser = argparse.ArgumentParser(prog="visionlab")
    sub = parser.add_subparsers(dest="command", required=True)
    image = sub.add_parser("image", help="Detect objects in an image")
    image.add_argument("source"); image.add_argument("--model"); image.add_argument("--conf", type=float, default=.25)
    video = sub.add_parser("video", help="Detect objects in a video file or webcam")
    video.add_argument("source", help="Video path or camera index, e.g. 0")
    video.add_argument("--model"); video.add_argument("--conf", type=float, default=.25)
    video.add_argument("--show", action="store_true"); video.add_argument("--output", default="results/predictions/video.mp4")
    video.add_argument("--max-frames", type=int, help="Stop after this many frames (useful for bounded webcam captures)")
    args = parser.parse_args()
    if args.command == "image": detect_image(args.source, args.model, args.conf)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        detect_video(source, args.model, args.conf, args.show, args.output, args.max_frames)

if __name__ == "__main__": main()

