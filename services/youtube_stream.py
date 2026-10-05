import os
import re
import logging
import subprocess
import tempfile
import shutil
import threading
from typing import Tuple, Generator, Callable, Optional
import yt_dlp
import numpy as np
try:
    from services.frame_extractor import VideoMetadata
except ModuleNotFoundError:
    from frame_extractor import VideoMetadata

logger = logging.getLogger("youtube_stream")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "youtube_stream.log"), mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class YouTubeExtractionError(Exception):
    """Custom exception for YouTube extraction failures with application-level error codes."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class YouTubeStreamResolver:
    """Helper service to extract direct stream URLs and stream video frames directly via FFmpeg pipes."""

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Determines if a given URL is a YouTube address."""
        if not url:
            return False
        url_lower = url.lower().strip()
        domains = ["youtube.com", "youtu.be", "youtube-nocookie.com"]
        return any(d in url_lower for d in domains)

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extracts the YouTube 11-character video ID from a URL."""
        match = re.search(r"(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/|\/live\/|^)([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else "unknown"

    @staticmethod
    def get_video_metadata(youtube_url: str) -> dict:
        """
        Extracts video metadata including title, duration, dimensions, stream URL, etc.
        Tries multiple player clients and cookie fallbacks (with cookies -> without cookies).
        """
        if not YouTubeStreamResolver.is_youtube_url(youtube_url):
            raise ValueError("Invalid YouTube URL. Please check the address and try again.")

        logger.info(f"Extracting YouTube metadata for: {youtube_url}")

        cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE")
        has_cookie_file = cookie_file and os.path.exists(cookie_file)

        # Order of attempts: with cookies first (if exists), then without cookies
        cookie_strategies = [True, False] if has_cookie_file else [False]

        # Player client configurations to try in sequence (android/mweb first for datacenter IPs)
        player_clients = [
            ['android'],
            ['mweb'],
            ['ios'],
            ['tv_embedded'],
            ['default']
        ]

        last_error = None
        last_error_code = "EXTRACTION_FAILED"

        for use_cookies in cookie_strategies:
            temp_dir = None
            temp_cookie_path = None
            if use_cookies:
                logger.info("Attempting extraction WITH configured cookies")
                temp_dir = tempfile.mkdtemp(prefix="ytextract_")
                temp_cookie_path = os.path.join(temp_dir, "youtube_cookies.txt")
                shutil.copy2(cookie_file, temp_cookie_path)
            else:
                if has_cookie_file:
                    logger.info("Attempting extraction WITHOUT cookies as fallback")
                else:
                    logger.info("Attempting extraction WITHOUT cookies (no cookie file configured)")

            for client_cfg in player_clients:
                ydl_opts = {
                    'format': 'best[height<=480]/best[height<=720]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                    'skip_download': True,
                    'extractor_args': {
                        'youtube': {'player_client': client_cfg},
                        'youtubepot-bgutilhttp': {'base_url': ['http://pot_provider:4416']}
                    }
                }
                if temp_cookie_path:
                    ydl_opts["cookiefile"] = temp_cookie_path

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                        
                        video_id = info.get('id') or YouTubeStreamResolver.extract_video_id(youtube_url)
                        title = info.get('title', 'YouTube Video')
                        duration = info.get('duration', 0.0)
                        thumbnail = info.get('thumbnail', '')
                        stream_url = info.get('url')
                        
                        width = info.get('width')
                        height = info.get('height')
                        fps = info.get('fps')
                        
                        if not width or not height:
                            width = 854
                            height = 480
                        if fps is None or fps <= 0:
                            fps = 30.0
                            
                        if not stream_url:
                            # Format url missing, continue trying next client
                            continue
                        
                        duration_seconds = float(duration) if duration is not None else 0.0
                        logger.info(f"Successfully extracted metadata using client={client_cfg} cookies={use_cookies}")
                        
                        return {
                            'video_id': video_id,
                            'title': title,
                            'duration': duration_seconds,
                            'thumbnail': thumbnail,
                            'stream_url': stream_url,
                            'width': int(width),
                            'height': int(height),
                            'fps': float(fps),
                            'http_headers': info.get('http_headers', {}),
                            'used_cookies': use_cookies,
                            'used_client': client_cfg[0] if client_cfg else 'android',
                            'mode': "FFmpeg Stream Processing"
                        }
                except yt_dlp.utils.DownloadError as e:
                    err_msg = str(e)
                    clean_msg = err_msg.split(";")[-1] if ";" in err_msg else err_msg
                    logger.warning(f"yt-dlp attempt failed (client={client_cfg}, cookies={use_cookies}): {clean_msg}")
                    
                    err_lower = err_msg.lower()
                    if "private video" in err_lower:
                        last_error_code = "VIDEO_PRIVATE"
                        last_error = "This YouTube video is private."
                        break # Private video won't work with other clients without auth
                    elif "bot verification" in err_lower or "captcha" in err_lower or "not a bot" in err_lower:
                        last_error_code = "BOT_VERIFICATION"
                        last_error = "YouTube blocked access requiring bot verification."
                    elif "confirm your age" in err_lower or "age-gated" in err_lower or "sign in" in err_lower or "login required" in err_lower:
                        last_error_code = "AUTH_REQUIRED"
                        last_error = "This YouTube video requires authenticated access. Update youtube_cookies.txt with valid YouTube cookies."
                        if use_cookies:
                            break # Go to next cookie strategy (without cookies or vice versa)
                    elif "unavailable" in err_lower or "not available" in err_lower or "removed" in err_lower or "does not exist" in err_lower:
                        last_error_code = "VIDEO_UNAVAILABLE"
                        last_error = "This YouTube video is unavailable or has been removed."
                    elif "geo-restricted" in err_lower or "not available in your country" in err_lower:
                        last_error_code = "GEO_RESTRICTED"
                        last_error = "This video is geo-restricted."
                        break
                    elif "members-only" in err_lower:
                        last_error_code = "MEMBERS_ONLY"
                        last_error = "This is a members-only video."
                        break
                    elif "429" in err_lower or "rate limit" in err_lower:
                        last_error_code = "RATE_LIMITED"
                        last_error = "YouTube rate limit exceeded."
                    elif "403" in err_lower or "forbidden" in err_lower:
                        last_error_code = "HTTP_403"
                        last_error = "YouTube returned HTTP 403 Forbidden."
                    else:
                        last_error_code = "EXTRACTION_FAILED"
                        last_error = f"Extraction failed: {clean_msg}"
                except Exception as e:
                    logger.warning(f"Unexpected error in attempt (client={client_cfg}, cookies={use_cookies}): {e}")
                    last_error = str(e)
                finally:
                    if temp_dir and os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")

        # If all attempts fail, raise custom exception with best error code
        raise YouTubeExtractionError(
            last_error_code,
            last_error or "Could not extract video stream from YouTube across all client configurations and cookie options."
        )

    @staticmethod
    def ffmpeg_stream_processor(
        stream_url: str,
        width: int,
        height: int,
        fps: float,
        extraction_interval: float,
        original_url: str = "",
        used_cookies: bool = True,
        used_client: str = "android"
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Decodes video stream directly using FFmpeg with HTTP reconnect support.
        """
        logger.info(f"Initializing FFmpeg stream processor for stream URL: {stream_url[:60]}...")
        
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            raise RuntimeError("FFmpeg is not installed or not found in system PATH. Streaming mode requires FFmpeg.") from e

        if extraction_interval is None or extraction_interval <= 0:
            extraction_interval = 2.0

        frame_size = width * height * 3

        cmd = [
            'ffmpeg',
            '-y',
            '-loglevel', 'error',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-i', stream_url,
            '-vf', f"fps=1/{extraction_interval}",
            '-s', f"{width}x{height}",
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

        frame_idx = 0
        try:
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) < frame_size:
                    break
                    
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
                timestamp_seconds = frame_idx * extraction_interval
                yield frame_idx, timestamp_seconds, frame
                frame_idx += 1
        except Exception as e:
            logger.error(f"Error reading frames from FFmpeg stream: {e}", exc_info=True)
            raise ValueError(f"FFmpeg stream error: {e}") from e
        finally:
            if process.stdout:
                process.stdout.close()
            process.terminate()
            process.wait()
            logger.info("FFmpeg stream processor terminated.")

    @staticmethod
    def download_and_extract_frames(
        youtube_url: str,
        interval: Optional[float] = None
    ) -> Tuple[VideoMetadata, Generator[Tuple[int, float, np.ndarray], None, None], Callable[[], None]]:
        """
        Downloads low-res mp4 to tempdir via yt-dlp and yields frames via OpenCV VideoCapture.
        Bulletproof strategy for cloud hosts (Streamlit Cloud, Colab, etc).
        """
        temp_dir = tempfile.mkdtemp(prefix="yt_lecture_")
        video_path = os.path.join(temp_dir, "lecture.mp4")
        
        cookie_file = os.environ.get("YOUTUBE_COOKIES_FILE")
        
        ydl_opts = {
            'format': 'best[height<=360]/bestvideo[height<=360]+bestaudio/best',
            'outtmpl': video_path,
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {
                'youtube': {'player_client': ['android', 'mweb', 'ios', 'default']}
            }
        }
        if cookie_file and os.path.exists(cookie_file):
            ydl_opts['cookiefile'] = cookie_file

        logger.info(f"Downloading YouTube video to {video_path}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'YouTube Video')
            duration = info.get('duration', 0.0) or 0.0
            fps = info.get('fps', 30.0) or 30.0

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(duration * fps)

        if interval is None or interval <= 0:
            if duration <= 600:
                interval = 2.0
            elif duration <= 1800:
                interval = 3.0
            elif duration <= 3600:
                interval = 4.0
            else:
                interval = 5.0

        metadata = VideoMetadata(
            video_path=video_path,
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            estimated_processing_time_seconds=duration / 20.0,
            title=title
        )

        def frame_generator():
            try:
                frame_step = max(1, int(round(fps * interval)))
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    timestamp_seconds = frame_idx / fps
                    yield frame_idx, timestamp_seconds, frame
                    
                    if frame_step > 1:
                        for _ in range(frame_step - 1):
                            if not cap.grab():
                                break
                        frame_idx += frame_step
                    else:
                        frame_idx += 1
            finally:
                cap.release()

        def cleanup_fn():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

        return metadata, frame_generator(), cleanup_fn

    @staticmethod
    def process_youtube_video(youtube_url: str, interval: Optional[float] = None) -> Tuple[VideoMetadata, Generator[Tuple[int, float, np.ndarray], None, None], Callable[[], None]]:
        """
        Orchestrates YouTube processing: downloads low-res video for zero-failure frame extraction.
        """
        try:
            return YouTubeStreamResolver.download_and_extract_frames(youtube_url, interval)
        except Exception as e:
            logger.warning(f"Download & extract failed, falling back to stream resolver: {e}")
            meta_dict = YouTubeStreamResolver.get_video_metadata(youtube_url)
            duration = meta_dict['duration']
            stream_url = meta_dict['stream_url']
            width = meta_dict['width']
            height = meta_dict['height']
            fps = meta_dict['fps']
            title = meta_dict.get('title', 'YouTube Video')
            used_cookies = meta_dict.get('used_cookies', True)
            used_client = meta_dict.get('used_client', 'android')
            
            if duration is None:
                duration = 0.0
            if fps is None or fps <= 0:
                fps = 30.0

            if interval is None or interval <= 0:
                if duration <= 600:
                    interval = 2.0
                elif duration <= 1800:
                    interval = 3.0
                elif duration <= 3600:
                    interval = 4.0
                else:
                    interval = 5.0

            duration_val = duration if duration is not None else 0.0
            fps_val = fps if (fps is not None and fps > 0) else 30.0
            interval_val = interval if (interval is not None and interval > 0) else 2.0
            total_frames = int(duration_val * fps_val)
            candidate_count = max(1, int(duration_val / interval_val))
            estimated_processing_time = candidate_count / 10.0
            
            metadata = VideoMetadata(
                video_path=youtube_url,
                duration_seconds=duration,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                estimated_processing_time_seconds=estimated_processing_time,
                title=title
            )
            
            frame_gen = YouTubeStreamResolver.ffmpeg_stream_processor(
                stream_url=stream_url,
                width=width,
                height=height,
                fps=fps,
                extraction_interval=interval,
                original_url=youtube_url,
                used_cookies=used_cookies,
                used_client=used_client
            )
            
            cleanup_fn = lambda: None
            return metadata, frame_gen, cleanup_fn
        
        cleanup_fn = lambda: None
        return metadata, frame_gen, cleanup_fn

