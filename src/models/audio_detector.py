"""
Audio Deepfake Detection Network.
Extracts spectral time-frequency artifacts from Log-Mel Spectrograms to detect
voice cloning, synthetic speech synthesis, and acoustic manipulation.
"""

from typing import Dict
import torch
import torch.nn as nn


class AudioSpectrogramEncoder(nn.Module):
    """
    2D Convolutional encoder processing Log-Mel Spectrograms.
    Captures local acoustic cues, harmonic phase discontinuities, and high-frequency roll-off.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 256, n_mels: int = 80):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Pool frequency (80 -> 40), preserve time

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Pool frequency (40 -> 20), preserve time

            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Pool frequency (20 -> 10), preserve time

            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),   # Pool remaining frequency to 1, keep exact time
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Tensor of shape (B, 1, n_mels, T) or (B, n_mels, T)
        Returns:
            Embeddings of shape (B, T, embedding_dim)
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # (B, 1, n_mels, T)

        x = self.conv_blocks(mel)   # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        x = self.proj(x)            # (B, T, embedding_dim)
        return x


class AudioDetector(nn.Module):
    """
    Audio-Only Deepfake Detector analyzing synthetic voice artifacts.
    Outputs audio-level prediction and time-step anomaly scores.
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = AudioSpectrogramEncoder(
            in_channels=1, embedding_dim=embedding_dim, n_mels=n_mels
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

        temporal_dim = hidden_dim * 2

        # Step-level anomaly classifier
        self.step_classifier = nn.Linear(temporal_dim, 1)

        # Global classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, mel_spectrogram: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel_spectrogram: Tensor of shape (B, n_mels, T) or (B, 1, n_mels, T)
        Returns:
            Dictionary with:
                - 'logits': (B, num_classes)
                - 'features': (B, T, embedding_dim)
                - 'temporal_features': (B, T, temporal_dim)
                - 'time_scores': (B, T) step-level manipulation probabilities
        """
        # Feature extraction
        audio_seq = self.encoder(mel_spectrogram)  # (B, T, embedding_dim)

        # Temporal modeling
        temporal_feats, _ = self.temporal_gru(audio_seq)  # (B, T, temporal_dim)

        # Time-step level scores
        step_logits = self.step_classifier(temporal_feats).squeeze(-1)
        time_scores = torch.sigmoid(step_logits)

        # Global pooling
        pooled = torch.mean(temporal_feats, dim=1)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "features": audio_seq,
            "temporal_features": temporal_feats,
            "time_scores": time_scores,
        }
