"""
Cross-Modal Multi-Head Attention and Inconsistency Modeling.
Captures phoneme-viseme desynchronization between speech audio and lip/face motion.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention Layer between Visual and Audio temporal features.
    
    1. Audio Query attends to Visual Keys/Values (speech-conditioned visual expectation)
    2. Visual Query attends to Audio Keys/Values (visual-conditioned acoustic expectation)
    3. Calculates explicit cross-modal discrepancy scores across the temporal axis.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # V -> A attention: Visual attends to Audio
        self.mha_v2a = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_v1 = nn.LayerNorm(embed_dim)
        self.norm_v2 = nn.LayerNorm(embed_dim)
        self.ffn_v = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # A -> V attention: Audio attends to Visual
        self.mha_a2v = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, visual_seq: torch.Tensor, audio_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_seq: (B, T_v, embed_dim)
            audio_seq:  (B, T_a, embed_dim)
        Returns:
            Dictionary containing:
                - 'visual_out': (B, T_v, embed_dim)
                - 'audio_out': (B, T_a, embed_dim)
                - 'attn_v2a': (B, T_v, T_a) attention map
                - 'attn_a2v': (B, T_a, T_v) attention map
                - 'inconsistency_scores': (B, min(T_v, T_a)) cross-modal divergence
        """
        # Visual queries attend to Audio keys & values
        v_attended, attn_v2a = self.mha_v2a(
            query=visual_seq, key=audio_seq, value=audio_seq
        )
        v_res = self.norm_v1(visual_seq + v_attended)
        v_out = self.norm_v2(v_res + self.ffn_v(v_res))

        # Audio queries attend to Visual keys & values
        a_attended, attn_a2v = self.mha_a2v(
            query=audio_seq, key=visual_seq, value=visual_seq
        )
        a_res = self.norm_a1(audio_seq + a_attended)
        a_out = self.norm_a2(a_res + self.ffn_a(a_res))

        # Temporal Phoneme-Viseme Inconsistency Metric
        # Cosine distance between co-occurring cross-attended representations
        min_len = min(v_out.shape[1], a_out.shape[1])
        v_sub = F.normalize(v_out[:, :min_len, :], p=2, dim=-1)
        a_sub = F.normalize(a_out[:, :min_len, :], p=2, dim=-1)

        # 1 - cos_sim: range [0.0 (perfect sync), 2.0 (maximum desync)]
        cos_sim = torch.sum(v_sub * a_sub, dim=-1)  # (B, min_len)
        inconsistency = (1.0 - cos_sim) / 2.0        # Normalized to [0.0, 1.0]

        return {
            "visual_out": v_out,
            "audio_out": a_out,
            "attn_v2a": attn_v2a,
            "attn_a2v": attn_a2v,
            "inconsistency_scores": inconsistency,
        }
