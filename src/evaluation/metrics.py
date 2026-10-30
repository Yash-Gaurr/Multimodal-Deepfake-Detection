"""
Pure NumPy / PyTorch Evaluation Metrics and Visualization for Deepfake Detection.
Computes AUC-ROC, Equal Error Rate (EER), F1-Score, Brier Score, and Confusion Matrix
with zero external dependencies.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt


def compute_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes False Positive Rates (FPR), True Positive Rates (TPR), and thresholds.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Sort scores in descending order
    desc_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    # Find distinct threshold indices
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    # True positives and False positives cumulative counts
    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    total_pos = max(1, tps[-1])
    total_neg = max(1, fps[-1])

    tpr = np.r_[0.0, tps / total_pos]
    fpr = np.r_[0.0, fps / total_neg]
    thresholds = np.r_[y_score_sorted[0] + 1e-3, y_score_sorted[threshold_idxs]]

    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Computes Area Under Curve using trapezoidal rule (compatible with all NumPy versions).
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    elif hasattr(np, "trapz"):
        return float(np.trapz(tpr, fpr))
    else:
        # Fallback manual trapezoidal integration
        d_fpr = np.diff(fpr)
        avg_tpr = (tpr[:-1] + tpr[1:]) / 2.0
        return float(np.sum(d_fpr * avg_tpr))


def compute_equal_error_rate(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) where False Positive Rate == False Negative Rate.
    """
    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    best_threshold = float(thresholds[min(eer_index, len(thresholds) - 1)])
    return eer, best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes full suite of classification and forensic evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    acc = float(np.mean(y_true == y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    fpr, tpr, threshs = compute_roc_curve(y_true, y_score)
    roc_auc = compute_auc(fpr, tpr)
    eer, eer_thresh = compute_equal_error_rate(fpr, tpr, threshs)
    brier = float(np.mean((y_score - y_true) ** 2))
    cm = np.array([[tn, fp], [fn, tp]])

    return {
        "accuracy": acc,
        "auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc_curve(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots comparative ROC curves for multiple models (e.g. Visual, Audio, Multimodal).
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    for i, (model_name, res) in enumerate(results_dict.items()):
        fpr = res["fpr"]
        tpr = res["tpr"]
        auc_val = res["auc"]
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{model_name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Multimodal Deepfake Detection - ROC Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
