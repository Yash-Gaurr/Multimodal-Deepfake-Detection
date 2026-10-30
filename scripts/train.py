"""
CLI Training Script for Multimodal Deepfake Detection.
Supports training visual-only, audio-only, late-fusion, or full multimodal cross-attention models.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multimodal_fusion import MultimodalDeepfakeDetector
from src.training.dataset import MultimodalDeepfakeDataset
from src.training.trainer import DeepfakeTrainer
from src.training.losses import MultimodalDeepfakeLoss
from src.utils.config import get_config
from scripts.generate_sample_data import generate_sample_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Deepfake Detection Model")
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "visual_only", "audio_only", "late_fusion"],
        help="Ablation mode to train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (tuned for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default="data/sample_data/dataset_index.csv",
        help="Path to CSV dataset index",
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints", help="Directory for saved models"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    # Generate sample dataset if it does not exist yet
    if not os.path.exists(args.dataset_csv):
        print(f"[Train] Dataset CSV not found at {args.dataset_csv}. Generating sample dataset...")
        generate_sample_dataset("data/sample_data")

    # Read dataset
    df = pd.read_csv(args.dataset_csv)
    samples = df.to_dict(orient="records")

    print(f"\n[Train] Total loaded samples: {len(samples)}")
    print(f"[Train] Selected training mode: {args.mode.upper()}")

    # Build dataset
    full_dataset = MultimodalDeepfakeDataset(
        samples=samples,
        cache_dir="data/processed/cache",
        is_training=True,
    )

    # Train / Val Split (80% / 20%)
    val_size = max(1, int(len(full_dataset) * 0.25))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
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

    criterion = MultimodalDeepfakeLoss(
        weight_visual=0.3,
        weight_audio=0.3,
        weight_sync=0.2,
        label_smoothing=cfg.training.label_smoothing,
    )

    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=args.lr,
        device=args.device,
        checkpoints_dir=args.checkpoints_dir,
        mode=args.mode,
    )

    trainer.fit(
        epochs=args.epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        save_name=f"best_{args.mode}_model.pth",
    )


if __name__ == "__main__":
    main()
