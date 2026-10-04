import queue
import threading
import logging
import cv2
import numpy as np
from PIL import Image
import imagehash
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Dict, Any, Tuple, Optional

logger = logging.getLogger("parallel_preprocessor")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    import os
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "parallel_preprocessor.log"), mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class ParallelFramePreprocessor:
    """
    Performs frame pre-processing (grayscale conversion, face detection, presenter checking,
    dHash generation, edge masking) in parallel threads.
    """

    def __init__(self, frame_generator: Generator[Tuple[int, float, np.ndarray], None, None], detector: Any, max_workers: int = 4, queue_size: int = 16):
        """
        Initialize preprocessor.
        
        :param frame_generator: Generator yielding (frame_idx, timestamp_seconds, frame_image)
        :param detector: SlideDetector instance
        :param max_workers: Number of concurrent worker threads
        :param queue_size: Bounded queue size to prevent out-of-memory errors
        """
        self.frame_generator = frame_generator
        self.detector = detector
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.producer_thread = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        """Starts the background producer thread to queue frame processing tasks."""
        self.producer_thread = threading.Thread(target=self._produce, daemon=True)
        self.producer_thread.start()
        logger.info(f"Started ParallelFramePreprocessor producer thread (workers={self.max_workers}, queue_size={self.queue_size})")

    def _preprocess_frame(self, frame_idx: int, timestamp_seconds: float, frame: np.ndarray) -> Dict[str, Any]:
        """Preprocesses a single frame. Runs on the thread pool."""
        try:
            # Grayscale conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            face_boxes = self.detector.detect_faces(frame)

            # Presenter only check
            is_presenter = self.detector.is_presenter_only_scene(frame, face_boxes)

            # dHash calculation
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dhash = imagehash.dhash(pil_img)

            # Edge detection mask
            edge_mask = self.detector.compute_text_edges(gray)

            return {
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "frame": frame,
                "gray": gray,
                "face_boxes": face_boxes,
                "is_presenter": is_presenter,
                "dhash": dhash,
                "edge_mask": edge_mask,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error preprocessing frame {frame_idx}: {e}", exc_info=True)
            return {
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "frame": frame,
                "error": e
            }

    def _produce(self) -> None:
        """Background thread logic: pulls frames from generator and submits tasks to executor."""
        try:
            for frame_idx, timestamp_seconds, frame in self.frame_generator:
                if self.stop_event.is_set():
                    break
                
                # Copy frame array so modifications do not corrupt subsequent frames
                frame_copy = frame.copy()
                
                # Submit processing task to executor
                future = self.executor.submit(self._preprocess_frame, frame_idx, timestamp_seconds, frame_copy)
                
                # Put future in task_queue (blocks if queue is full)
                self.task_queue.put(future)
                
            # Signal completion
            self.task_queue.put(None)
        except Exception as e:
            logger.error(f"Exception in producer loop: {e}", exc_info=True)
            self.task_queue.put(None)

    def get_preprocessed_frames(self) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding preprocessed frame metadata dicts from the bounded queue."""
        self.start()
        try:
            while True:
                future = self.task_queue.get()
                if future is None:
                    break
                
                result = future.result()
                if result.get("error") is not None:
                    logger.warning(f"Skipping frame {result['frame_idx']} due to error: {result['error']}")
                    continue
                    
                yield result
        finally:
            self.stop_event.set()
            self.executor.shutdown(wait=False)
            logger.info("Shutdown ParallelFramePreprocessor.")
