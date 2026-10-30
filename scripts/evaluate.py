"""
CLI Evaluation and Ablation Benchmark Script.
Evaluates Video-only, Audio-only, Late-fusion, and Multimodal cross-attention models.
Outputs paper-ready comparison tables and ROC figures.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multimodal_fusion import MultimodalDeepfakeDetector
from src.training.dataset import MultimodalDeepfakeDataset
from src.evaluation.comparator import ModelComparator
from src.utils.config import get_config
from scripts.generate_sample_data import generate_sample_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark Deepfake Detectors")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="data/sample_data/dataset_index.csv",
        help="Path to CSV dataset index",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="reports",
        help="Directory to save comparison table and ROC curves",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    # Ensure dataset exists
    if not os.path.exists(args.dataset_csv):
        print(f"[Eval] Dataset CSV not found at {args.dataset_csv}. Generating sample dataset...")
        generate_sample_dataset("data/sample_data")

    df = pd.read_csv(args.dataset_csv)
    samples = df.to_dict(orient="records")

    print(f"\n[Eval] Loaded {len(samples)} test samples from {args.dataset_csv}")

    test_dataset = MultimodalDeepfakeDataset(
        samples=samples,
        cache_dir="data/processed/cache",
        is_training=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Initialize model
    model = MultimodalDeepfakeDetector(
        embedding_dim=cfg.models.multimodal_fusion.hidden_dim,
        hidden_dim=cfg.models.multimodal_fusion.hidden_dim // 2,
        num_classes=2,
        num_heads=cfg.models.multimodal_fusion.num_heads,
        fusion_type=cfg.models.multimodal_fusion.fusion_type,
        pretrained_visual=True,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[Eval] Loaded checkpoint: {args.checkpoint}")

    comparator = ModelComparator(model=model, test_loader=test_loader, device=args.device)
    comparator.run_ablation_study(
        modes=["visual_only", "audio_only", "late_fusion", "multimodal"],
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
