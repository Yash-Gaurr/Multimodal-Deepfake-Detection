"""
Loss Functions for Multimodal Deepfake Detection.
Includes multi-task cross-entropy, label smoothing, and audio-visual inconsistency regularization.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalDeepfakeLoss(nn.Module):
    """
    Combined multi-task loss for joint audio-visual deepfake detection.
    
    L_total = L_multimodal + lambda_v * L_visual + lambda_a * L_audio + lambda_sync * L_sync
    """

    def __init__(
        self,
        weight_visual: float = 0.3,
        weight_audio: float = 0.3,
        weight_sync: float = 0.2,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.weight_visual = weight_visual
        self.weight_audio = weight_audio
        self.weight_sync = weight_sync
        self.label_smoothing = label_smoothing
        self.bce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dictionary returned by MultimodalDeepfakeDetector forward pass.
            targets: LongTensor of shape (B,) containing class labels (0=Real, 1=Fake).
            
        Returns:
            Dictionary with total loss and individual loss components.
        """
        # Primary Multimodal Loss
        total_loss = self.bce(outputs["logits"], targets)
        loss_dict = {"loss_multimodal": total_loss}

        # Visual Branch Auxiliary Loss (if present)
        if "visual_prob" in outputs and outputs["visual_prob"] is not None:
            vis_logits = torch.stack(
                [1.0 - outputs["visual_prob"], outputs["visual_prob"]], dim=-1
            )
            # Safe log for cross entropy
            vis_loss = F.nll_loss(torch.log(vis_logits + 1e-7), targets)
            total_loss = total_loss + self.weight_visual * vis_loss
            loss_dict["loss_visual"] = vis_loss

        # Audio Branch Auxiliary Loss (if present)
        if "audio_prob" in outputs and outputs["audio_prob"] is not None:
            aud_logits = torch.stack(
                [1.0 - outputs["audio_prob"], outputs["audio_prob"]], dim=-1
            )
            aud_loss = F.nll_loss(torch.log(aud_logits + 1e-7), targets)
            total_loss = total_loss + self.weight_audio * aud_loss
            loss_dict["loss_audio"] = aud_loss

        # Audio-Visual Inconsistency Regularization Loss
        # Real videos (target=0) should have low cross-modal inconsistency (~0.0)
        # Fake videos (target=1) should correlate with higher inconsistency
        if "mean_inconsistency" in outputs and outputs["mean_inconsistency"] is not None:
            target_float = targets.float()
            sync_loss = F.mse_loss(outputs["mean_inconsistency"], target_float * 0.8)
            total_loss = total_loss + self.weight_sync * sync_loss
            loss_dict["loss_sync"] = sync_loss

        loss_dict["loss_total"] = total_loss
        return loss_dict
