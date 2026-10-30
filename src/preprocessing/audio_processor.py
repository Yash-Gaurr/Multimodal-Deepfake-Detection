"""
Audio Preprocessing Module for Multimodal Deepfake Detection.
Handles audio demuxing, 16kHz resampling, and Log-Mel Spectrogram extraction
using GPU-accelerated PyTorch / TorchAudio with native wave fallback.
"""

import os
import subprocess
import tempfile
import wave
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioProcessor:
    """
    Extracts and transforms audio signals into Log-Mel Spectrograms and acoustic features
    leveraging GPU-accelerated TorchAudio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration_seconds: float = 3.2,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        """
        Args:
            sample_rate: Standard sampling rate (Hz) for acoustic analysis (default 16kHz).
            duration_seconds: Target temporal segment duration.
            n_fft: FFT window size.
            hop_length: Stride between successive FFT frames.
            n_mels: Number of Mel filterbank bins.
            f_min: Minimum frequency for Mel filterbank.
            f_max: Maximum frequency for Mel filterbank.
        """
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.target_length = int(sample_rate * duration_seconds)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # TorchAudio native MelSpectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=2.0,
            normalized=False,
            center=True,
        )

    def extract_audio_from_video(self, video_path: str, output_wav_path: Optional[str] = None) -> str:
        """
        Demuxes audio track from video to a 16kHz mono WAV file using imageio-ffmpeg or ffmpeg.
        """
        if output_wav_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_wav_path = tmp.name
            tmp.close()

        # Try imageio_ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                output_wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_wav_path
        except Exception:
            pass

        # Try system ffmpeg
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                output_wav_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_wav_path
        except Exception:
            # Fallback silence WAV using standard python wave module
            with wave.open(output_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                silence_bytes = b"\x00" * (self.target_length * 2)
                wf.writeframes(silence_bytes)
            return output_wav_path

    def load_audio(
        self,
        audio_path_or_video: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Loads and normalizes raw audio waveform to float32 in [-1.0, 1.0].
        """
        if duration is None:
            duration = self.duration_seconds

        is_video = audio_path_or_video.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        temp_wav = None

        if is_video:
            wav_path = self.extract_audio_from_video(audio_path_or_video)
            temp_wav = wav_path
        else:
            wav_path = audio_path_or_video

        try:
            waveform = None
            sr = self.sample_rate

            # Use torchaudio if file exists and is valid
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                try:
                    tensor_wave, sr = torchaudio.load(wav_path)
                    # Convert to mono
                    if tensor_wave.shape[0] > 1:
                        tensor_wave = torch.mean(tensor_wave, dim=0, keepdim=True)
                    # Resample if needed
                    if sr != self.sample_rate:
                        resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                        tensor_wave = resampler(tensor_wave)
                    waveform = tensor_wave.squeeze(0).numpy().astype(np.float32)
                except Exception:
                    pass

            # Fallback to standard library wave module if torchaudio load had an issue
            if waveform is None:
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                    try:
                        with wave.open(wav_path, "rb") as wf:
                            sr = wf.getframerate()
                            n_channels = wf.getnchannels()
                            sampwidth = wf.getsampwidth()
                            n_frames = wf.getnframes()
                            raw_bytes = wf.readframes(n_frames)
                            if sampwidth == 2:
                                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            else:
                                data = np.frombuffer(raw_bytes, dtype=np.int8).astype(np.float32) / 128.0
                            if n_channels > 1:
                                data = data.reshape(-1, n_channels).mean(axis=1)
                            waveform = data
                    except Exception:
                        pass

            if waveform is None:
                waveform = np.zeros(self.target_length, dtype=np.float32)

            # Slice time window
            start_sample = int(start_time * self.sample_rate)
            expected_samples = int(duration * self.sample_rate)
            end_sample = start_sample + expected_samples

            if start_sample < len(waveform):
                waveform = waveform[start_sample:end_sample]
            else:
                waveform = np.zeros(expected_samples, dtype=np.float32)

            # Pad or truncate
            if len(waveform) < expected_samples:
                pad_width = expected_samples - len(waveform)
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
            elif len(waveform) > expected_samples:
                waveform = waveform[:expected_samples]

            return waveform

        finally:
            if temp_wav is not None and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def compute_log_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Computes standardized Log-Mel Spectrogram using TorchAudio.
        Returns 2D numpy array of shape (n_mels, T_frames) normalized to mean=0, std=1.
        """
        if len(waveform) == 0:
            return np.zeros((self.n_mels, 1), dtype=np.float32)

        wave_tensor = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, N)

        # TorchAudio Mel-Spectrogram
        mel_spec = self.mel_transform(wave_tensor).squeeze(0)  # (n_mels, T)

        # Log compression
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))

        # Standardization
        mean = torch.mean(log_mel)
        std = torch.std(log_mel) + 1e-7
        norm_log_mel = (log_mel - mean) / std

        return norm_log_mel.numpy().astype(np.float32)

    def extract_features(
        self,
        audio_or_video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts both raw waveform and standardized Mel-Spectrogram.
        """
        waveform = self.load_audio(audio_or_video_path, start_time, duration)
        mel_spec = self.compute_log_mel_spectrogram(waveform)
        return {
            "waveform": waveform,
            "mel_spectrogram": mel_spec,
        }
