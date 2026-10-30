"""
Unit and Integration Tests for Multimodal Deepfake Detection.
Validates tensor shapes, audio-visual alignment, and forward passes across all ablation modes.
"""

import os
import unittest
import numpy as np
import torch

from src.preprocessing.video_processor import VideoProcessor
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.sync_aligner import SyncAligner
from src.models.visual_detector import VisualDetector
from src.models.audio_detector import AudioDetector
from src.models.cross_attention import CrossModalAttentionBlock
from src.models.multimodal_fusion import MultimodalDeepfakeDetector
from src.training.losses import MultimodalDeepfakeLoss


class TestMultimodalPipeline(unittest.TestCase):
    """Test suite covering the complete pipeline."""

    def setUp(self):
        self.batch_size = 2
        self.num_frames = 16
        self.face_size = 224
        self.lip_size = 96
        self.n_mels = 80
        self.embed_dim = 128
        self.hidden_dim = 64

    def test_video_processor_crop(self):
        """Validates face and lip cropping logic."""
        vp = VideoProcessor(
            num_frames=self.num_frames,
            face_crop_size=self.face_size,
            lip_crop_size=self.lip_size,
        )
        fake_frame = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
        face_rgb, lip_rgb = vp.crop_face(fake_frame)

        self.assertEqual(face_rgb.shape, (self.face_size, self.face_size, 3))
        self.assertEqual(lip_rgb.shape, (self.lip_size, self.lip_size, 3))

    def test_audio_processor_melspec(self):
        """Validates STFT and Mel filterbank calculations."""
        ap = AudioProcessor(sample_rate=16000, duration_seconds=1.6, n_mels=self.n_mels)
        fake_wave = np.random.uniform(-1.0, 1.0, 16000 * 2).astype(np.float32)
        mel = ap.compute_log_mel_spectrogram(fake_wave)

        self.assertEqual(mel.shape[0], self.n_mels)
        self.assertGreater(mel.shape[1], 0)

    def test_sync_aligner(self):
        """Validates temporal synchronization between audio and video frames."""
        aligner = SyncAligner(target_num_steps=self.num_frames)
        dummy_mel = np.random.randn(self.n_mels, 95).astype(np.float32)
        aligned_mel = aligner.align_spectrogram_to_frames(dummy_mel, self.num_frames)

        self.assertEqual(aligned_mel.shape, (self.n_mels, self.num_frames))

    def test_cross_attention_block(self):
        """Validates bidirectional cross-modal attention shapes and discrepancy scores."""
        block = CrossModalAttentionBlock(
            embed_dim=self.embed_dim, num_heads=2, ff_dim=self.embed_dim * 2
        )
        v = torch.randn(self.batch_size, self.num_frames, self.embed_dim)
        a = torch.randn(self.batch_size, self.num_frames, self.embed_dim)

        out = block(v, a)
        self.assertEqual(out["visual_out"].shape, (self.batch_size, self.num_frames, self.embed_dim))
        self.assertEqual(out["audio_out"].shape, (self.batch_size, self.num_frames, self.embed_dim))
        self.assertEqual(out["inconsistency_scores"].shape, (self.batch_size, self.num_frames))

    def test_multimodal_ablation_modes(self):
        """Validates forward passes across all 4 ablation configurations."""
        model = MultimodalDeepfakeDetector(
            embedding_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_classes=2,
            num_heads=2,
            pretrained_visual=False,
        )

        frames = torch.randn(self.batch_size, self.num_frames, 3, self.face_size, self.face_size)
        mel = torch.randn(self.batch_size, self.n_mels, self.num_frames)

        # Mode 1: Multimodal
        out_mm = model(frames=frames, mel_spectrogram=mel, mode="multimodal")
        self.assertEqual(out_mm["logits"].shape, (self.batch_size, 2))
        self.assertIsNotNone(out_mm["inconsistency_timeline"])

        # Mode 2: Visual Only
        out_vo = model(frames=frames, mode="visual_only")
        self.assertEqual(out_vo["logits"].shape, (self.batch_size, 2))

        # Mode 3: Audio Only
        out_ao = model(mel_spectrogram=mel, mode="audio_only")
        self.assertEqual(out_ao["logits"].shape, (self.batch_size, 2))

        # Mode 4: Late Fusion
        out_lf = model(frames=frames, mel_spectrogram=mel, mode="late_fusion")
        self.assertEqual(out_lf["logits"].shape, (self.batch_size, 2))

    def test_multitask_loss(self):
        """Validates multi-task loss computation and gradient backpropagation."""
        criterion = MultimodalDeepfakeLoss()
        outputs = {
            "logits": torch.randn(self.batch_size, 2, requires_grad=True),
            "visual_prob": torch.sigmoid(torch.randn(self.batch_size)),
            "audio_prob": torch.sigmoid(torch.randn(self.batch_size)),
            "mean_inconsistency": torch.sigmoid(torch.randn(self.batch_size)),
        }
        targets = torch.tensor([0, 1], dtype=torch.long)

        loss_dict = criterion(outputs, targets)
        self.assertIn("loss_total", loss_dict)
        loss_dict["loss_total"].backward()
        self.assertIsNotNone(outputs["logits"].grad)


if __name__ == "__main__":
    unittest.main()
