"""
Audio-Visual Temporal Synchronization and Alignment Module.
Aligns visual frame sequences with corresponding audio spectrogram time slices
using high-performance PyTorch 1D interpolation.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F


class SyncAligner:
    """
    Aligns video and audio temporal representations to ensure 1-to-1 or segment-to-segment
    cross-modal correspondence for attention and fusion networks.
    """

    def __init__(self, target_num_steps: int = 32):
        """
        Args:
            target_num_steps: Fixed number of temporal steps for joint sequence modeling.
        """
        self.target_num_steps = target_num_steps

    def align_spectrogram_to_frames(
        self,
        mel_spectrogram: np.ndarray,
        target_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resamples the temporal dimension of a Mel-Spectrogram to match target frame count
        using PyTorch linear interpolation.

        Args:
            mel_spectrogram: Array of shape (n_mels, T_audio_frames)
            target_steps: Target time length. If None, uses self.target_num_steps.

        Returns:
            Aligned Mel-Spectrogram of shape (n_mels, target_steps)
        """
        if target_steps is None:
            target_steps = self.target_num_steps

        n_mels, current_time_steps = mel_spectrogram.shape

        if current_time_steps == target_steps:
            return mel_spectrogram.astype(np.float32)

        # Use PyTorch 1D linear interpolation: shape (1, n_mels, T)
        mel_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).float()
        aligned_tensor = F.interpolate(
            mel_tensor, size=target_steps, mode="linear", align_corners=False
        )

        return aligned_tensor.squeeze(0).numpy().astype(np.float32)

    def prepare_multimodal_sample(
        self,
        video_features: Dict[str, np.ndarray],
        audio_features: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Combines and synchronizes extracted visual and acoustic features into a unified bundle.

        Returns:
            Dictionary containing:
                - 'faces': Aligned face sequence of shape (T, H, W, 3)
                - 'lips': Aligned lip sequence of shape (T, H, W, 3)
                - 'mel_spectrogram': Aligned spectrogram of shape (n_mels, T)
                - 'raw_spectrogram': Original unscaled spectrogram (n_mels, T_raw)
                - 'timestamps': Video timestamps (T,)
        """
        faces = video_features["faces"]      # (T, H, W, 3)
        lips = video_features["lips"]        # (T, H, W, 3)
        timestamps = video_features["timestamps"]

        mel = audio_features["mel_spectrogram"]  # (n_mels, T_raw)
        num_frames = faces.shape[0]

        # Resample audio spectrogram time axis to match visual frames
        aligned_mel = self.align_spectrogram_to_frames(mel, target_steps=num_frames)

        return {
            "faces": faces,
            "lips": lips,
            "mel_spectrogram": aligned_mel,
            "raw_spectrogram": mel,
            "timestamps": timestamps,
        }
