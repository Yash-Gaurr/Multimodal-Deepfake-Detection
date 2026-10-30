"""
Training Engine for Multimodal Deepfake Detection.
Features Mixed Precision (AMP), Cosine Learning Rate Scheduling,
Early Stopping, and Metric Tracking.
"""

import os
import time
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.multimodal_fusion import MultimodalDeepfakeDetector
from src.training.losses import MultimodalDeepfakeLoss
from src.evaluation.metrics import evaluate_predictions


class DeepfakeTrainer:
    """
    Executes training and evaluation loops with GPU acceleration and metric logging.
    """

    def __init__(
        self,
        model: MultimodalDeepfakeDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        checkpoints_dir: str = "checkpoints",
        mixed_precision: bool = True,
        mode: str = "multimodal",
    ):
        """
        Args:
            model: The detection network instance.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation set.
            criterion: Loss function module.
            learning_rate: Initial AdamW optimizer learning rate.
            weight_decay: L2 regularization penalty.
            device: 'cuda' or 'cpu'.
            checkpoints_dir: Save path for model weights.
            mixed_precision: Whether to use FP16 AMP.
            mode: 'multimodal', 'visual_only', 'audio_only', or 'late_fusion'.
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or MultimodalDeepfakeLoss()
        self.checkpoints_dir = checkpoints_dir
        self.mode = mode
        self.mixed_precision = mixed_precision and self.device.type == "cuda"

        os.makedirs(checkpoints_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auc": [],
        }

    def train_epoch(self, epoch_idx: int) -> float:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch["frames"].to(self.device)
            mel = batch["mel_spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(frames=frames, mel_spectrogram=mel, mode=self.mode)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict["loss_total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Runs evaluation over the validation loader."""
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            mel = batch["mel_spectrogram"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(frames=frames, mel_spectrogram=mel, mode=self.mode)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict["loss_total"]

            total_loss += loss.item()

            probs = outputs["probabilities"][:, 1].cpu().numpy()
            preds = torch.argmax(outputs["probabilities"], dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

        val_loss = total_loss / max(1, len(self.val_loader))
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        eval_res = evaluate_predictions(all_targets, all_probs)

        return {
            "loss": val_loss,
            "accuracy": eval_res["accuracy"],
            "auc": eval_res["auc"],
            "f1": eval_res["f1"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
        }

    def fit(
        self,
        epochs: int = 15,
        early_stopping_patience: int = 4,
        save_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Executes complete training routine with early stopping and best checkpoint persistence.
        """
        if save_name is None:
            save_name = f"best_model_{self.mode}.pth"

        best_auc = -1.0
        best_loss = float("inf")
        patience_counter = 0
        best_save_path = os.path.join(self.checkpoints_dir, save_name)

        print(f"\n[Trainer] Training started on device: {self.device} (Mode: {self.mode})")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            elapsed = time.time() - t0

            val_loss = metrics["loss"]
            val_acc = metrics["accuracy"]
            val_auc = metrics["auc"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Val AUC: {val_auc:.4f}"
            )

            # Checkpoint save on best AUC or improved loss at top AUC
            improved = False
            if val_auc > best_auc:
                best_auc = val_auc
                best_loss = val_loss
                improved = True
            elif val_auc == best_auc and val_loss < best_loss:
                best_loss = val_loss
                improved = True

            if improved:
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": val_auc,
                        "val_accuracy": val_acc,
                        "val_loss": val_loss,
                        "mode": self.mode,
                    },
                    best_save_path,
                )
                print(f"  --> Saved new best checkpoint to: {best_save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Trainer] Early stopping triggered after {epoch} epochs.")
                    break

        print(f"[Trainer] Finished. Best Validation AUC: {best_auc:.4f}\n")
        return self.history
