import os
import cv2
import logging
from dataclasses import dataclass
from typing import Generator, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Dataclass to hold video metadata."""
    video_path: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    total_frames: int
    estimated_processing_time_seconds: float
    title: str = "Lecture Video"

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def duration_str(self) -> str:
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class FrameExtractor:
    """Service to load, validate, and extract frames from an MP4 video."""

    def __init__(self, video_path: str, extraction_interval: Optional[float] = None, duration_override: Optional[float] = None):
        """
        Initialize the FrameExtractor.

        :param video_path: Path to the local video file or streaming URL.
        :param extraction_interval: Interval in seconds to extract candidate frames. If None, uses adaptive sampling.
        :param duration_override: Pre-resolved duration in seconds (useful for YouTube streams).
        """
        self.video_path = video_path
        self.extraction_interval = extraction_interval
        self.duration_override = duration_override
        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None

    def validate_video(self) -> None:
        """
        Validate that the video file exists, is in a readable format, and is not corrupted.
        Raises ValueError or FileNotFoundError if invalid.
        """
        is_stream = self.video_path.startswith("http://") or self.video_path.startswith("https://")
        
        if not is_stream:
            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video file not found at path: {self.video_path}")

            if not os.path.isfile(self.video_path):
                raise ValueError(f"Path is not a file: {self.video_path}")

            # Check file extension
            _, ext = os.path.splitext(self.video_path.lower())
            supported_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            if ext not in supported_extensions:
                raise ValueError(
                     f"Unsupported video format '{ext}'. Supported formats: {', '.join(supported_extensions)}"
                )

        # Attempt to open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file (possibly corrupted or missing codecs): {self.video_path}")

        # Test read first frame to check corruption
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or frame.size == 0:
            raise ValueError(f"Video file is corrupted or empty: cannot read any frames.")

        logger.info(f"Successfully validated video: {self.video_path}")

    def get_metadata(self) -> VideoMetadata:
        """
        Extract video metadata and estimate processing time.
        """
        if self.metadata is not None:
            return self.metadata

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video to read metadata.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps is None or fps <= 0:
            fps = 29.97  # Fallback
            logger.warning(f"Invalid FPS read from video. Falling back to default: {fps}")

        fps_val = fps if (fps is not None and fps > 0) else 29.97
        if self.duration_override is not None:
            duration_seconds = self.duration_override
        else:
            total_frames_val = total_frames if total_frames is not None else 0
            duration_seconds = total_frames_val / fps_val

        if total_frames is None or total_frames <= 0:
            total_frames = int(duration_seconds * fps_val)

        # Resolve extraction interval dynamically if not specified or adaptive
        if self.extraction_interval is None:
            if duration_seconds is None:
                duration_seconds = 0.0
            if duration_seconds <= 600:
                self.extraction_interval = 2.0
            elif duration_seconds <= 1800:
                self.extraction_interval = 3.0
            elif duration_seconds <= 3600:
                self.extraction_interval = 4.0
            else:
                self.extraction_interval = 5.0
            logger.info(f"Adaptive frame sampling configured interval: {self.extraction_interval}s based on duration {duration_seconds:.1f}s")
        
        # Estimate processing speed at 150 frames per second as a base baseline
        estimated_speed_fps = 150.0
        
        fps_val = fps if (fps is not None and fps > 0) else 29.97
        total_frames_val = total_frames if total_frames is not None else 0
        estimated_speed_fps_val = estimated_speed_fps if (estimated_speed_fps is not None and estimated_speed_fps > 0) else 150.0
            
        estimated_processing_time_seconds = (total_frames_val / fps_val) / (estimated_speed_fps_val / fps_val)
        
        if self.extraction_interval is None or self.extraction_interval <= 0:
            self.extraction_interval = 2.0
            
        duration_seconds_val = duration_seconds if duration_seconds is not None else 0.0
        interval_val = self.extraction_interval if (self.extraction_interval is not None and self.extraction_interval > 0) else 2.0
        candidate_count = max(1, int(duration_seconds_val / interval_val))
        estimated_processing_time_seconds = candidate_count / 10.0

        # Try to extract a clean title from filename
        try:
            title = os.path.splitext(os.path.basename(self.video_path))[0]
            # Replace underscores and hyphens with spaces, capitalize
            title = title.replace("_", " ").replace("-", " ").title()
        except Exception:
            title = "Lecture Video"

        self.metadata = VideoMetadata(
            video_path=self.video_path,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            estimated_processing_time_seconds=estimated_processing_time_seconds,
            title=title
        )
        return self.metadata

    def extract_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Streams candidate frames every extraction_interval seconds.
        Yields (frame_index, timestamp_seconds, frame_image)
        """
        self.validate_video()
        meta = self.get_metadata()

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise ValueError("Failed to open video capture streaming stream.")

        fps = meta.fps
        if fps is None or fps <= 0:
            fps = 30.0
            
        if self.extraction_interval is None or self.extraction_interval <= 0:
            self.extraction_interval = 2.0

        frame_step = max(1, int(round(fps * self.extraction_interval)))
        total_frames = meta.total_frames

        frame_idx = 0
        try:
            while True:
                # Set reader to specific frame index
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    break

                fps_val = fps if (fps is not None and fps > 0) else 30.0
                timestamp_seconds = frame_idx / fps_val
                yield frame_idx, timestamp_seconds, frame

                # Increment index
                frame_idx += frame_step
                if frame_idx >= total_frames:
                    break
        finally:
            if self._cap is not None:
                self._cap.release()
                logger.info("Video capture released successfully.")
