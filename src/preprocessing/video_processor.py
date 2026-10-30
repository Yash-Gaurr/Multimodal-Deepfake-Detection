"""
Video Preprocessing Module for Multimodal Deepfake Detection.
Handles frame extraction, temporal subsampling, face cropping, lip ROI extraction,
and visual tensor normalization supporting OpenCV, ImageIO, and Pillow backends.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Check OpenCV availability
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Check ImageIO availability
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


class VideoProcessor:
    """
    Extracts face frames and lip sequences from video streams for deepfake analysis.
    Dual-backend architecture supporting OpenCV and Pillow/ImageIO.
    """

    def __init__(
        self,
        target_fps: int = 25,
        num_frames: int = 32,
        face_crop_size: int = 224,
        lip_crop_size: int = 96,
        face_margin: float = 0.2,
    ):
        """
        Args:
            target_fps: Frame rate to normalize video sequences.
            num_frames: Number of temporal frames to sample per segment.
            face_crop_size: Output spatial resolution (H=W) for cropped face images.
            lip_crop_size: Output spatial resolution for lip ROI images.
            face_margin: Padding ratio around detected face bounding box.
        """
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.face_crop_size = face_crop_size
        self.lip_crop_size = lip_crop_size
        self.face_margin = face_margin

        self.face_cascade = None
        if HAS_CV2:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def read_video_metadata(self, video_path: str) -> Dict[str, Union[int, float]]:
        """Reads basic properties of a video file."""
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0.0
                cap.release()
                return {
                    "fps": fps,
                    "total_frames": total_frames,
                    "width": width,
                    "height": height,
                    "duration": duration,
                }

        if HAS_IMAGEIO:
            try:
                reader = imageio.get_reader(video_path)
                meta = reader.get_meta_data()
                fps = meta.get("fps", self.target_fps)
                total_frames = meta.get("nframes", self.num_frames)
                size = meta.get("size", (320, 320))
                reader.close()
                return {
                    "fps": fps,
                    "total_frames": total_frames if total_frames != float("inf") else self.num_frames,
                    "width": size[0],
                    "height": size[1],
                    "duration": self.num_frames / fps,
                }
            except Exception:
                pass

        # Fallback default metadata
        return {
            "fps": self.target_fps,
            "total_frames": self.num_frames,
            "width": 320,
            "height": 320,
            "duration": self.num_frames / self.target_fps,
        }

    def detect_face_bbox(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face bounding box (x, y, w, h) in an RGB frame.
        """
        h_img, w_img = frame_rgb.shape[:2]

        if HAS_CV2 and self.face_cascade is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                largest_face = max(faces, key=lambda b: b[2] * b[3])
                return tuple(map(int, largest_face))

        # Default face bounding box: center 65% crop
        side = int(min(h_img, w_img) * 0.65)
        x = (w_img - side) // 2
        y = int((h_img - side) * 0.4)
        return (x, y, side, side)

    def crop_face(
        self,
        frame_rgb: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crops face region and lip ROI from an RGB frame.
        Returns (face_rgb, lip_rgb) normalized arrays.
        """
        h_img, w_img = frame_rgb.shape[:2]

        if bbox is None:
            bbox = self.detect_face_bbox(frame_rgb)

        x, y, w, h = bbox
        mx = int(w * self.face_margin)
        my = int(h * self.face_margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w_img, x + w + mx)
        y2 = min(h_img, y + h + my)

        face_crop = frame_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = frame_rgb

        # Lip ROI: lower 35% of face, middle 50% width
        fh, fw = face_crop.shape[:2]
        lip_y1 = int(fh * 0.65)
        lip_y2 = int(fh * 0.95)
        lip_x1 = int(fw * 0.25)
        lip_x2 = int(fw * 0.75)
        lip_crop = face_crop[lip_y1:lip_y2, lip_x1:lip_x2]

        if lip_crop.size == 0:
            lip_crop = face_crop

        # Resize using PIL (always available)
        pil_face = Image.fromarray(face_crop).resize(
            (self.face_crop_size, self.face_crop_size), Image.Resampling.BILINEAR
        )
        pil_lip = Image.fromarray(lip_crop).resize(
            (self.lip_crop_size, self.lip_crop_size), Image.Resampling.BILINEAR
        )

        face_arr = np.array(pil_face, dtype=np.uint8)
        lip_arr = np.array(pil_lip, dtype=np.uint8)

        return face_arr, lip_arr

    def extract_frames(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts synchronized, normalized face and lip sequences.
        """
        meta = self.read_video_metadata(video_path)
        fps = meta["fps"]

        raw_frames = []

        # Try OpenCV reader
        if HAS_CV2:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                while len(raw_frames) < self.num_frames * 4:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # BGR -> RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_frames.append(rgb)
                cap.release()

        # Try ImageIO reader if OpenCV failed or is absent
        if len(raw_frames) == 0 and HAS_IMAGEIO and os.path.exists(video_path):
            try:
                reader = imageio.get_reader(video_path)
                for idx, frame in enumerate(reader):
                    raw_frames.append(np.array(frame))
                    if len(raw_frames) >= self.num_frames * 4:
                        break
                reader.close()
            except Exception:
                pass

        # If file is empty or cannot be opened, generate synthetic placeholder frames
        if len(raw_frames) == 0:
            for _ in range(self.num_frames):
                blank = np.ones((self.face_crop_size, self.face_crop_size, 3), dtype=np.uint8) * 220
                raw_frames.append(blank)

        # Uniformly sample num_frames
        indices = np.linspace(0, len(raw_frames) - 1, self.num_frames, dtype=int)

        sampled_faces = []
        sampled_lips = []
        timestamps = []
        bboxes = []

        for step_idx, frame_idx in enumerate(indices):
            frame = raw_frames[frame_idx]
            bbox = self.detect_face_bbox(frame)
            face_arr, lip_arr = self.crop_face(frame, bbox)

            sampled_faces.append(face_arr)
            sampled_lips.append(lip_arr)
            timestamps.append(step_idx / fps if fps > 0 else step_idx / self.target_fps)
            bboxes.append(bbox)

        faces_tensor = np.array(sampled_faces, dtype=np.float32) / 255.0
        lips_tensor = np.array(sampled_lips, dtype=np.float32) / 255.0
        time_arr = np.array(timestamps, dtype=np.float32)

        return {
            "faces": faces_tensor,       # (T, H, W, 3)
            "lips": lips_tensor,         # (T, H, W, 3)
            "timestamps": time_arr,     # (T,)
            "face_bboxes": bboxes,
        }
