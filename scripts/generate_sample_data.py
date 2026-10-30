"""
Generates synthetic sample audio-visual media files for testing, training,
and running the ablation study immediately without requiring a 50GB dataset download.
Uses Pillow, ImageIO, and standard library wave module for zero-dependency execution.
"""

import os
import wave
import numpy as np
from PIL import Image, ImageDraw
import imageio


def create_synthetic_face_frame(
    width: int = 320,
    height: int = 320,
    mouth_open: float = 0.5,
    artifact_noise: float = 0.0,
    face_hue: int = 0,
    t_step: int = 0,
) -> np.ndarray:
    """
    Renders an avatar face with dynamic mouth opening and optional manipulation artifacts
    using Pillow.
    """
    img = Image.new("RGB", (width, height), color=(235, 238, 242))
    draw = ImageDraw.Draw(img)

    # Face Oval with variation
    cx, cy = width // 2, height // 2
    rx = int(width * (0.30 + (face_hue % 5) * 0.01))
    ry = int(height * (0.38 + (face_hue % 4) * 0.01))

    # Base skin tone variation
    r_tone = min(255, 235 + (face_hue % 7) * 2)
    g_tone = max(180, 215 - (face_hue % 5) * 4)
    b_tone = max(160, 195 - (face_hue % 6) * 5)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(r_tone, g_tone, b_tone), outline=(180, 160, 140), width=2)

    # Eyes
    eye_y = int(height * 0.42)
    left_x = int(width * 0.39)
    right_x = int(width * 0.61)
    r = int(width * 0.04)

    draw.ellipse([left_x - r, eye_y - r, left_x + r, eye_y + r], fill=(255, 255, 255), outline=(100, 100, 100))
    draw.ellipse([right_x - r, eye_y - r, right_x + r, eye_y + r], fill=(255, 255, 255), outline=(100, 100, 100))
    draw.ellipse([left_x - r//2, eye_y - r//2, left_x + r//2, eye_y + r//2], fill=(30, 40, 60))
    draw.ellipse([right_x - r//2, eye_y - r//2, right_x + r//2, eye_y + r//2], fill=(30, 40, 60))

    # Dynamic Mouth (Viseme articulation)
    m_cx, m_cy = cx, int(height * 0.70)
    m_rx = int(width * 0.14)
    m_ry = int(max(4, m_rx * mouth_open))
    draw.ellipse([m_cx - m_rx, m_cy - m_ry, m_cx + m_rx, m_cy + m_ry], fill=(175, 55, 55), outline=(130, 35, 35))

    frame = np.array(img, dtype=np.uint8)

    # Inject visual deepfake boundary artifacts if specified
    if artifact_noise > 0.0:
        # Sharp high-frequency boundary distortion & block artifacts
        noise = (np.random.normal(0, artifact_noise * 45, (height, width, 3))).astype(np.int16)
        
        # Color banding / checkerboard artifact typical of neural generators
        block_size = 16
        checker = np.indices((height // block_size, width // block_size)).sum(axis=0) % 2
        checker_full = np.repeat(np.repeat(checker, block_size, axis=0), block_size, axis=1)
        checker_artifact = (checker_full[:, :, None] * (artifact_noise * 30)).astype(np.int16)

        frame_int = frame.astype(np.int16) + noise + checker_artifact
        frame = np.clip(frame_int, 0, 255).astype(np.uint8)

    return frame


def generate_audio_wav(
    output_wav_path: str,
    duration: float = 3.2,
    sample_rate: int = 16000,
    carrier_freq: float = 220.0,
    amplitude_mod_freq: float = 2.0,
    add_synthetic_jitter: bool = False,
):
    """
    Generates a speech-like harmonic carrier with amplitude modulation using standard wave module.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Harmonics
    signal = (
        0.5 * np.sin(2 * np.pi * carrier_freq * t)
        + 0.3 * np.sin(2 * np.pi * carrier_freq * 2.0 * t)
        + 0.15 * np.sin(2 * np.pi * carrier_freq * 3.0 * t)
        + 0.08 * np.sin(2 * np.pi * carrier_freq * 4.0 * t)
    )

    # Phoneme amplitude envelope
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * amplitude_mod_freq * t))
    audio = signal * envelope

    if add_synthetic_jitter:
        # Unnatural robotic pitch jump and phase discontinuity
        noise = np.random.normal(0, 0.12, num_samples)
        glitch = np.sign(np.sin(2 * np.pi * 880.0 * t)) * 0.15
        audio = audio * 0.7 + noise + glitch

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def build_sample_media(
    video_path: str,
    audio_path: str,
    num_frames: int = 32,
    fps: int = 10,
    is_sync: bool = True,
    is_visual_fake: bool = False,
    is_audio_fake: bool = False,
    sample_seed: int = 0,
):
    """
    Synthesizes and writes a video and audio file pair using imageio and wave.
    """
    duration = num_frames / fps
    mod_freq = 1.8 + (sample_seed % 5) * 0.3
    carrier_freq = 180.0 + (sample_seed % 7) * 20.0

    if is_audio_fake:
        carrier_freq += 200.0  # Unnatural robotic frequency offset

    # Generate Audio
    generate_audio_wav(
        output_wav_path=audio_path,
        duration=duration,
        sample_rate=16000,
        carrier_freq=carrier_freq,
        amplitude_mod_freq=mod_freq,
        add_synthetic_jitter=is_audio_fake,
    )

    # Generate Video frames
    phase_shift = 0.0 if is_sync else (np.pi * 0.8 + (sample_seed % 3) * 0.2)
    frames = []

    for i in range(num_frames):
        t_sec = i / fps
        mouth_opening = 0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t_sec + phase_shift))
        artifact = 0.85 if is_visual_fake else 0.0

        frame = create_synthetic_face_frame(
            width=320,
            height=320,
            mouth_open=mouth_opening,
            artifact_noise=artifact,
            face_hue=sample_seed,
            t_step=i,
        )
        frames.append(frame)

    # Write video using imageio
    try:
        imageio.mimwrite(video_path, frames, fps=fps)
    except Exception:
        gif_path = video_path.replace(".mp4", ".gif")
        imageio.mimwrite(gif_path, frames, fps=fps)
        if not os.path.exists(video_path):
            import shutil
            shutil.copyfile(gif_path, video_path)


def generate_sample_dataset(output_dir: str = "data/sample_data", count_per_category: int = 6) -> str:
    """
    Generates a balanced dataset covering:
      1. Real Video + Real Audio (AUTHENTIC)
      2. Fake Video + Real Audio (FaceSwap / Visual Deepfake)
      3. Real Video + Fake Audio (Voice Clone / Acoustic Deepfake)
      4. Fake Video + Fake Audio (LipSync Desynchronized Deepfake)
      5. Multimodal Combined Fake
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    categories = [
        # (category_name, is_sync, is_vis_fake, is_aud_fake, label)
        ("authentic_real", True, False, False, 0),
        ("faceswap_visual", True, True, False, 1),
        ("voiceclone_audio", True, False, True, 1),
        ("lipsync_mismatch", False, False, False, 1),
        ("multimodal_combined", False, True, True, 1),
    ]

    print(f"\n[Sample Generator] Generating robust dataset in: {output_dir}")
    sample_idx = 1

    for cat_name, is_sync, is_vis_fake, is_aud_fake, label in categories:
        for k in range(1, count_per_category + 1):
            name = f"sample_{sample_idx:02d}_{cat_name}_{k}"
            v_path = os.path.join(output_dir, f"{name}.mp4")
            a_path = os.path.join(output_dir, f"{name}.wav")

            build_sample_media(
                video_path=v_path,
                audio_path=a_path,
                num_frames=32,
                fps=10,
                is_sync=is_sync,
                is_visual_fake=is_vis_fake,
                is_audio_fake=is_aud_fake,
                sample_seed=sample_idx,
            )

            rows.append(f"{v_path},{a_path},{label},{cat_name}")
            sample_idx += 1

    index_csv = os.path.join(output_dir, "dataset_index.csv")
    with open(index_csv, "w", encoding="utf-8") as f:
        f.write("video_path,audio_path,label,category\n")
        f.write("\n".join(rows) + "\n")

    print(f"[Sample Generator] Created {len(rows)} samples. Index saved to: {index_csv}\n")
    return index_csv


if __name__ == "__main__":
    generate_sample_dataset()
