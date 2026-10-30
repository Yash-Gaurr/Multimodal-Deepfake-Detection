"""
PyTorch Dataset and Augmentations for Multimodal Deepfake Detection.
Supports raw video/audio loading, cached feature arrays, and cross-modal augmentations.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.sync_aligner import SyncAligner


class MultimodalDeepfakeDataset(Dataset):
    """
    Dataset loader for audio-visual deepfake samples.
    """

    def __init__(
        self,
        samples: List[Dict[str, Union[str, int]]],
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        sync_aligner: Optional[SyncAligner] = None,
        cache_dir: Optional[str] = None,
        is_training: bool = True,
    ):
        """
        Args:
            samples: List of dicts, each with keys:
                - 'video_path': path to video file (or None)
                - 'audio_path': optional path to audio file if separated
                - 'label': int (0 for Real, 1 for Fake)
                - 'category': optional str (e.g. 'faceswap', 'voiceswap', 'lipsync', 'real')
            video_processor: Configured VideoProcessor instance.
            audio_processor: Configured AudioProcessor instance.
            sync_aligner: Configured SyncAligner instance.
            cache_dir: Directory to cache extracted features for rapid epoch iteration.
            is_training: Whether to apply data augmentations.
        """
        self.samples = samples
        self.video_processor = video_processor or VideoProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        self.sync_aligner = sync_aligner or SyncAligner()
        self.cache_dir = cache_dir
        self.is_training = is_training

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """
        Applies SpecAugment (Time and Frequency masking) to Mel-Spectrogram.
        """
        mel_aug = mel.copy()
        n_mels, t_steps = mel_aug.shape

        # Frequency masking
        f_mask_len = np.random.randint(1, max(2, n_mels // 8))
        f_start = np.random.randint(0, n_mels - f_mask_len)
        mel_aug[f_start : f_start + f_mask_len, :] = 0.0

        # Time masking
        t_mask_len = np.random.randint(1, max(2, t_steps // 8))
        t_start = np.random.randint(0, t_steps - t_mask_len)
        mel_aug[:, t_start : t_start + t_mask_len] = 0.0

        return mel_aug

    def _apply_visual_augment(self, frames: np.ndarray) -> np.ndarray:
        """
        Applies video-level augmentations: random horizontal flip across all frames.
        """
        # Horizontal flip with 50% probability
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()
        return frames

    def _load_sample_features(self, item: Dict[str, Union[str, int]]) -> Dict[str, np.ndarray]:
        """Loads or computes preprocessed multimodal features for a sample."""
        video_path = item.get("video_path")
        audio_path = item.get("audio_path") or video_path

        # If cache exists, load from disk
        cache_file = None
        if self.cache_dir and video_path:
            safe_name = Path(video_path).stem + ".npz"
            cache_file = os.path.join(self.cache_dir, safe_name)
            if os.path.exists(cache_file):
                try:
                    data = np.load(cache_file)
                    return {
                        "faces": data["faces"],
                        "lips": data["lips"],
                        "mel_spectrogram": data["mel_spectrogram"],
                    }
                except Exception:
                    pass

        # Extract on the fly
        if video_path and os.path.exists(video_path):
            v_feats = self.video_processor.extract_frames(video_path)
            a_feats = self.audio_processor.extract_features(audio_path)
            bundle = self.sync_aligner.prepare_multimodal_sample(v_feats, a_feats)
        else:
            # Fallback synthetic tensor for graceful execution if file is missing
            t_frames = self.video_processor.num_frames
            h_face = self.video_processor.face_crop_size
            h_lip = self.video_processor.lip_crop_size
            n_mels = self.audio_processor.n_mels

            bundle = {
                "faces": np.zeros((t_frames, h_face, h_face, 3), dtype=np.float32),
                "lips": np.zeros((t_frames, h_lip, h_lip, 3), dtype=np.float32),
                "mel_spectrogram": np.zeros((n_mels, t_frames), dtype=np.float32),
            }

        # Save to cache if enabled
        if cache_file:
            np.savez_compressed(
                cache_file,
                faces=bundle["faces"],
                lips=bundle["lips"],
                mel_spectrogram=bundle["mel_spectrogram"],
            )

        return bundle

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        label = int(item["label"])

        features = self._load_sample_features(item)
        faces = features["faces"]           # (T, H, W, 3)
        mel = features["mel_spectrogram"]   # (n_mels, T)

        # Augmentation during training
        if self.is_training:
            faces = self._apply_visual_augment(faces)
            if np.random.rand() > 0.5:
                mel = self._apply_spec_augment(mel)

        # Convert to PyTorch Tensors
        # Visual: (T, 3, H, W)
        face_tensor = torch.from_numpy(faces).permute(0, 3, 1, 2).float()
        # Audio: (n_mels, T)
        mel_tensor = torch.from_numpy(mel).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "frames": face_tensor,
            "mel_spectrogram": mel_tensor,
            "label": label_tensor,
        }
