"""
Visual Deepfake Detection Network.
Extracts spatial face/lip artifacts and models temporal frame consistency.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    """
    Extracts spatial 2D embeddings from individual face and lip frames.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Use lightweight ResNet-18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        # Remove final FC classification layer
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Project 512 backbone output to target embedding dim
        self.proj = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames of shape (B*T, 3, H, W)
        Returns:
            Embeddings of shape (B*T, embedding_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


class VisualDetector(nn.Module):
    """
    Visual-Only Deepfake Detector analyzing temporal facial and lip dynamics.
    Outputs video-level prediction and frame-level anomaly scores.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.spatial_extractor = VisualFeatureExtractor(
            embedding_dim=embedding_dim, pretrained=pretrained
        )

        # Temporal sequence modeling (BiGRU)
        self.temporal_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        temporal_dim = hidden_dim * 2  # Bidirectional

        # Frame-level anomaly projection (for explainability)
        self.frame_classifier = nn.Linear(temporal_dim, 1)

        # Video-level classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: Tensor of shape (B, T, 3, H, W) or (B, T, H, W, 3)
        Returns:
            Dictionary with:
                - 'logits': (B, num_classes) classification logits
                - 'features': (B, T, embedding_dim) temporal visual feature sequence
                - 'frame_scores': (B, T) frame-level manipulation probability scores
        """
        # Ensure channels-first format (B, T, 3, H, W)
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 1, 4, 2, 3)

        B, T, C, H, W = frames.shape

        # Fold temporal dimension into batch: (B*T, C, H, W)
        flat_frames = frames.reshape(B * T, C, H, W)
        spatial_feats = self.spatial_extractor(flat_frames)  # (B*T, embedding_dim)

        # Reshape back to temporal sequence: (B, T, embedding_dim)
        visual_seq = spatial_feats.reshape(B, T, self.embedding_dim)

        # Temporal recurrent modeling
        temporal_feats, _ = self.temporal_gru(visual_seq)  # (B, T, temporal_dim)

        # Frame-level anomaly scores
        frame_logits = self.frame_classifier(temporal_feats).squeeze(-1)  # (B, T)
        frame_scores = torch.sigmoid(frame_logits)

        # Global temporal pooling (attention or mean pooling)
        pooled = torch.mean(temporal_feats, dim=1)  # (B, temporal_dim)

        # Classification logits
        logits = self.classifier(pooled)  # (B, num_classes)

        return {
            "logits": logits,
            "features": visual_seq,
            "temporal_features": temporal_feats,
            "frame_scores": frame_scores,
        }
