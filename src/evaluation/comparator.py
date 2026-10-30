"""
Comparative Model Benchmark for Ablation Study.
Directly compares Video-Only, Audio-Only, Late-Fusion, and Multimodal Cross-Attention architectures.
"""

import os
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.multimodal_fusion import MultimodalDeepfakeDetector
from src.evaluation.metrics import evaluate_predictions, plot_roc_curve


class ModelComparator:
    """
    Executes head-to-head empirical comparison across unimodal and multimodal models.
    """

    def __init__(
        self,
        model: MultimodalDeepfakeDetector,
        test_loader: DataLoader,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.test_loader = test_loader

    @torch.no_grad()
    def evaluate_mode(self, mode: str) -> Dict[str, np.ndarray]:
        """Runs test inference for a specific ablation mode."""
        all_targets = []
        all_scores = []
        latencies = []

        for batch in self.test_loader:
            frames = batch["frames"].to(self.device)
            mel = batch["mel_spectrogram"].to(self.device)
            labels = batch["label"].cpu().numpy()

            t0 = time.perf_counter()
            outputs = self.model(frames=frames, mel_spectrogram=mel, mode=mode)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(labels)
            latencies.append(elapsed_ms)

            fake_probs = outputs["fake_prob"].cpu().numpy()
            all_scores.extend(fake_probs.tolist())
            all_targets.extend(labels.tolist())

        y_true = np.array(all_targets)
        y_score = np.array(all_scores)
        metrics = evaluate_predictions(y_true, y_score)
        metrics["latency_ms"] = float(np.mean(latencies))

        return metrics

    def run_ablation_study(
        self,
        modes: Optional[List[str]] = None,
        save_dir: str = "reports",
    ) -> pd.DataFrame:
        """
        Runs complete benchmark across all ablation modes and exports tables and plots.
        """
        if modes is None:
            modes = ["visual_only", "audio_only", "late_fusion", "multimodal"]

        os.makedirs(save_dir, exist_ok=True)
        results = {}
        rows = []

        name_map = {
            "visual_only": "1. Visual-Only (Video Baseline)",
            "audio_only": "2. Audio-Only (Audio Baseline)",
            "late_fusion": "3. Late Fusion (Score Concat)",
            "multimodal": "4. Multimodal Cross-Attention (Proposed)",
        }

        print("\n" + "=" * 70)
        print("MULTIMODAL DEEPFAKE DETECTION — ABLATION BENCHMARK")
        print("=" * 70)

        for mode in modes:
            display_name = name_map.get(mode, mode)
            print(f"--> Evaluating: {display_name}...")
            metrics = self.evaluate_mode(mode)
            results[display_name] = metrics

            rows.append({
                "Architecture / Mode": display_name,
                "Accuracy (%)": f"{metrics['accuracy'] * 100:.2f}%",
                "AUC-ROC": f"{metrics['auc']:.4f}",
                "EER (%)": f"{metrics['eer'] * 100:.2f}%",
                "F1-Score": f"{metrics['f1']:.4f}",
                "Latency (ms/sample)": f"{metrics['latency_ms']:.1f}",
            })

        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))
        print("=" * 70 + "\n")

        # Save Markdown table
        md_path = os.path.join(save_dir, "ablation_benchmark_results.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Multimodal Deepfake Detection - Comparative Ablation Results\n\n")
            headers = list(df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, r in df.iterrows():
                f.write("| " + " | ".join(str(r[c]) for c in headers) + " |\n")
            f.write("\n")

        # Save ROC comparison plot
        roc_path = os.path.join(save_dir, "roc_comparison_curve.png")
        plot_roc_curve(results, save_path=roc_path)
        print(f"[Comparator] Saved benchmark table to: {md_path}")
        print(f"[Comparator] Saved comparative ROC curve to: {roc_path}\n")

        return df
