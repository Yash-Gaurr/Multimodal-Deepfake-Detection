"""
Multimodal Audio-Visual Fusion Network for Deepfake Detection.
Integrates visual and audio streams using cross-modal attention, temporal alignment,
and cross-modal inconsistency modeling. Supports ablation testing (Video-only, Audio-only, Fused).
"""

from typing import Dict, Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.visual_detector import VisualDetector
from src.models.audio_detector import AudioDetector
from src.models.cross_attention import CrossModalAttentionBlock


class MultimodalDeepfakeDetector(nn.Module):
    """
    Unified Multimodal Deepfake Detector.
    
    Can operate in four ablation modes:
      1. 'multimodal': Full cross-attention audio-visual architecture (Recommended)
      2. 'late_fusion': Baseline concatenation of separate visual and audio embeddings
      3. 'visual_only': Uses visual stream only
      4. 'audio_only': Uses audio stream only
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        fusion_type: Literal["cross_attention", "late_fusion"] = "cross_attention",
        pretrained_visual: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # 1. Unimodal Visual Branch
        self.visual_net = VisualDetector(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained_visual,
        )

        # 2. Unimodal Audio Branch
        self.audio_net = AudioDetector(
            n_mels=80,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # 3. Cross-Modal Attention Blocks (Phoneme-Viseme Interaction)
        self.cross_attention = CrossModalAttentionBlock(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=embedding_dim * 2,
            dropout=dropout,
        )

        # 4. Joint Multimodal Fusion Head
        # Concatenates: [pooled_v, pooled_a, pooled_cross_v, pooled_cross_a, mean_inconsistency]
        joint_input_dim = (embedding_dim * 4) + 1

        self.multimodal_classifier = nn.Sequential(
            nn.Linear(joint_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # 5. Baseline Late Fusion Head (for ablation comparisons)
        self.late_fusion_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        mel_spectrogram: Optional[torch.Tensor] = None,
        mode: Optional[Literal["multimodal", "visual_only", "audio_only", "late_fusion"]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.
        
        Args:
            frames: Video frames tensor (B, T, 3, H, W) or (B, T, H, W, 3)
            mel_spectrogram: Audio Log-Mel Spectrogram (B, n_mels, T)
            mode: Override execution mode (defaults to self.fusion_type or input presence)
            
        Returns:
            Dictionary containing:
                - 'logits': Primary classification logits (B, num_classes)
                - 'probabilities': Softmax probabilities [P(Real), P(Fake)]
                - 'visual_prob': Unimodal visual fake probability
                - 'audio_prob': Unimodal audio fake probability
                - 'inconsistency_timeline': Temporal phoneme-viseme mismatch curve (B, T)
                - 'mean_inconsistency': Mean cross-modal mismatch score (B,)
        """
        if mode is None:
            if frames is not None and mel_spectrogram is not None:
                mode = self.fusion_type
            elif frames is not None:
                mode = "visual_only"
            elif mel_spectrogram is not None:
                mode = "audio_only"
            else:
                raise ValueError("At least one modality (frames or mel_spectrogram) must be provided.")

        # --- ABLATION 1: Visual-Only Mode ---
        if mode == "visual_only":
            vis_res = self.visual_net(frames)
            probs = F.softmax(vis_res["logits"], dim=-1)
            return {
                "logits": vis_res["logits"],
                "probabilities": probs,
                "fake_prob": probs[:, 1],
                "visual_prob": probs[:, 1],
                "audio_prob": None,
                "inconsistency_timeline": None,
                "mean_inconsistency": None,
                "frame_scores": vis_res["frame_scores"],
            }

        # --- ABLATION 2: Audio-Only Mode ---
        if mode == "audio_only":
            aud_res = self.audio_net(mel_spectrogram)
            probs = F.softmax(aud_res["logits"], dim=-1)
            return {
                "logits": aud_res["logits"],
                "probabilities": probs,
                "fake_prob": probs[:, 1],
                "visual_prob": None,
                "audio_prob": probs[:, 1],
                "inconsistency_timeline": None,
                "mean_inconsistency": None,
                "time_scores": aud_res["time_scores"],
            }

        # Both modalities available
        vis_res = self.visual_net(frames)
        aud_res = self.audio_net(mel_spectrogram)

        vis_probs = F.softmax(vis_res["logits"], dim=-1)
        aud_probs = F.softmax(aud_res["logits"], dim=-1)

        v_feat = vis_res["features"]  # (B, T_v, embed_dim)
        a_feat = aud_res["features"]  # (B, T_a, embed_dim)

        # --- ABLATION 3: Late Fusion Baseline ---
        if mode == "late_fusion":
            pooled_v = torch.mean(v_feat, dim=1)
            pooled_a = torch.mean(a_feat, dim=1)
            concat_feat = torch.cat([pooled_v, pooled_a], dim=-1)
            logits = self.late_fusion_classifier(concat_feat)
            probs = F.softmax(logits, dim=-1)
            return {
                "logits": logits,
                "probabilities": probs,
                "fake_prob": probs[:, 1],
                "visual_prob": vis_probs[:, 1],
                "audio_prob": aud_probs[:, 1],
                "inconsistency_timeline": None,
                "mean_inconsistency": None,
            }

        # --- CORE INNOVATION: Cross-Modal Attention & Inconsistency Fusion ---
        cross_res = self.cross_attention(v_feat, a_feat)
        v_cross = cross_res["visual_out"]           # (B, T_v, embed_dim)
        a_cross = cross_res["audio_out"]            # (B, T_a, embed_dim)
        inconsistency = cross_res["inconsistency_scores"]  # (B, T)
        mean_inconsistency = torch.mean(inconsistency, dim=-1, keepdim=True)  # (B, 1)

        # Temporal pooling of native and cross-attended representations
        p_v = torch.mean(v_feat, dim=1)
        p_a = torch.mean(a_feat, dim=1)
        p_cross_v = torch.mean(v_cross, dim=1)
        p_cross_a = torch.mean(a_cross, dim=1)

        # Form composite multimodal representation
        joint_feat = torch.cat(
            [p_v, p_a, p_cross_v, p_cross_a, mean_inconsistency], dim=-1
        )

        logits = self.multimodal_classifier(joint_feat)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "fake_prob": probs[:, 1],
            "visual_prob": vis_probs[:, 1],
            "audio_prob": aud_probs[:, 1],
            "inconsistency_timeline": inconsistency,
            "mean_inconsistency": mean_inconsistency.squeeze(-1),
            "frame_scores": vis_res["frame_scores"],
            "time_scores": aud_res["time_scores"],
            "attn_v2a": cross_res["attn_v2a"],
            "attn_a2v": cross_res["attn_a2v"],
        }
