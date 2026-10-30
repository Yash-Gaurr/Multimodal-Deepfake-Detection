"""
CLI Inference Script for testing arbitrary video/audio files.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import DeepfakeDetectionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Deepfake Detection Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--audio", type=str, default=None, help="Optional separate audio file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "visual_only", "audio_only", "late_fusion"],
        help="Inference mode",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50, help="Fake decision threshold"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found at: {args.video}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Inference] Loading pipeline and analyzing: {args.video}")
    pipeline = DeepfakeDetectionPipeline(
        checkpoint_path=args.checkpoint,
        fake_threshold=args.threshold,
    )

    result = pipeline.predict(
        video_path=args.video, audio_path=args.audio, mode=args.mode
    )

    report = pipeline.generate_report(result)
    print("\n" + report + "\n")


if __name__ == "__main__":
    main()
