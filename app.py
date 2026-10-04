import os
import sys
import json
import time
import logging
import shutil
import cv2
import numpy as np
import streamlit as st

# Append current directory so services subfolder is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from services.frame_extractor import FrameExtractor, VideoMetadata
    from services.slide_detector import SlideDetector
    from services.change_classifier import ChangeClassifier
    from services.ocr_service import OCRService
    from services.ppt_generator import PPTGenerator
    from services.pdf_generator import PDFGenerator
    from services.youtube_stream import YouTubeStreamResolver
    from services.parallel_preprocessor import ParallelFramePreprocessor
except ModuleNotFoundError:
    from frame_extractor import FrameExtractor, VideoMetadata
    from slide_detector import SlideDetector
    from change_classifier import ChangeClassifier
    from ocr_service import OCRService
    from ppt_generator import PPTGenerator
    from pdf_generator import PDFGenerator
    from youtube_stream import YouTubeStreamResolver
    from parallel_preprocessor import ParallelFramePreprocessor

logger = logging.getLogger("streamlit_prototype")

# Define standalone output paths
OUTPUT_DIR = os.path.abspath("outputs")
SLIDES_DIR = os.path.join(OUTPUT_DIR, "slides")
METADATA_PATH = os.path.join(OUTPUT_DIR, "slide_metadata.json")
OCR_RESULTS_PATH = os.path.join(OUTPUT_DIR, "ocr_results.json")
OCR_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "ocr_summary.json")
VIDEO_METADATA_PATH = os.path.join(OUTPUT_DIR, "video_metadata.json")
PPTX_PATH = os.path.join(OUTPUT_DIR, "lecture_presentation.pptx")
PDF_PATH = os.path.join(OUTPUT_DIR, "lecture_presentation.pdf")

# Streamlit Page Configuration
st.set_page_config(
    page_title="Lecture Content Extractor",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    .step-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .step-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-size: 15px;
    }
    .step-completed { color: #10b981; font-weight: 600; }
    .step-active { color: #3b82f6; font-weight: 600; }
    .step-pending { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


def format_seconds(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def check_existing_outputs():
    required_files = [METADATA_PATH, OCR_RESULTS_PATH, PPTX_PATH]
    return all(os.path.exists(f) for f in required_files)


if "processed" not in st.session_state:
    st.session_state.processed = check_existing_outputs()
if "processing" not in st.session_state:
    st.session_state.processing = False


def render_status(current_step, slide_count=0, speed=0.0, eta="--:--:--"):
    steps = [
        ("Slide Detection & Content Filtering (Visual Only)", ""),
        ("Post-Hoc Parallel OCR & PPT Generation", ""),
        ("Exporting Presentation Files (PPTX / PDF)", ""),
    ]
    
    html = "<div class='step-card'>"
    html += "<h3 style='margin-top:0;'>Processing Status</h3>"
    
    for idx, (name, icon) in enumerate(steps):
        step_num = idx + 1
        if step_num < current_step:
            html += f"<div class='step-item step-completed'>✅ Step {step_num}/3: {name} (Completed)</div>"
        elif step_num == current_step:
            html += f"<div class='step-item step-active'>⏳ Step {step_num}/3: {name} (In Progress...)</div>"
        else:
            html += f"<div class='step-item step-pending'>⚪ Step {step_num}/3: {name} (Pending...)</div>"
            
    if current_step == 1:
        html += f"<div style='margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(128, 128, 128, 0.15); font-size: 14px; color: gray;'>"
        html += f"Slides Captured: <strong>{slide_count}</strong> | Speed: <strong>{speed:.1f} frames/s</strong> | ETA: <strong>{eta}</strong>"
        html += f"</div>"
        
    html += "</div>"
    return html


# App Header
st.title("📚 Lecture Content Extractor Prototype")
st.markdown("Convert YouTube lectures into clean PowerPoint and PDF presentations without Docker or Redis.")

# Video Input
youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Extraction Trigger
if st.button("🚀 Start Extraction", type="primary", use_container_width=True):
    if not youtube_url:
        st.warning("Please enter a valid YouTube URL.")
    elif not YouTubeStreamResolver.is_youtube_url(youtube_url):
        st.warning("Please enter a valid YouTube address (youtube.com or youtu.be).")
    else:
        st.session_state.processing = True
        st.session_state.processed = False
        st.session_state.youtube_url = youtube_url
        st.rerun()


# Pipeline Processing Loop
if st.session_state.processing:
    status_slot = st.empty()
    status_slot.markdown(render_status(1), unsafe_allow_html=True)
    
    try:
        url = st.session_state.youtube_url
        
        # YouTube Metadata & Stream Setup
        metadata, frame_gen, cleanup_fn = YouTubeStreamResolver.process_youtube_video(url, interval=None)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(VIDEO_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "title": getattr(metadata, "title", "Lecture Video"),
                "video_path": metadata.video_path,
                "duration": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "total_frames": metadata.total_frames
            }, f, indent=4)

        detector = SlideDetector(sensitivity="medium")
        classifier = ChangeClassifier()
        ocr_service = OCRService()
        
        if os.path.exists(SLIDES_DIR):
            shutil.rmtree(SLIDES_DIR, ignore_errors=True)
        os.makedirs(SLIDES_DIR, exist_ok=True)
        
        slide_metadata_list = []
        slide_id = 0
        
        prev_accepted_frame = None
        prev_accepted_gray = None
        prev_accepted_hash = None
        prev_accepted_edge = None
        
        duration = metadata.duration_seconds or 0.0
        fps = metadata.fps if (metadata.fps and metadata.fps > 0) else 30.0
        
        if duration <= 600:
            resolved_interval = 2.0
        elif duration <= 1800:
            resolved_interval = 3.0
        elif duration <= 3600:
            resolved_interval = 4.0
        else:
            resolved_interval = 5.0
            
        candidate_step = max(1, int(round(fps * resolved_interval)))
        total_frames = metadata.total_frames or int(duration * fps)
        total_candidates = max(1, int(total_frames / candidate_step))
        processed_candidates = 0
        
        loop_start_time = time.time()
        
        candidate_frame = None
        candidate_gray = None
        candidate_hash = None
        candidate_edge = None
        candidate_timestamp = None
        stable_count = 0
        persistence_threshold = 1

        # Step 1: Slide Detection & Filtering
        preprocessor = ParallelFramePreprocessor(frame_gen, detector)
        preprocessed_gen = preprocessor.get_preprocessed_frames()

        for precomputed in preprocessed_gen:
            frame_idx = precomputed["frame_idx"]
            timestamp_seconds = precomputed["timestamp_seconds"]
            current_frame = precomputed["frame"]
            gray_curr = precomputed["gray"]
            face_boxes = precomputed["face_boxes"]
            is_presenter = precomputed["is_presenter"]
            dhash_curr = precomputed["dhash"]
            edge_curr = precomputed["edge_mask"]

            processed_candidates += 1
            timestamp_str = format_seconds(timestamp_seconds)
            
            elapsed_time = max(0.001, time.time() - loop_start_time)
            avg_speed = processed_candidates / elapsed_time
            remaining_candidates = max(0, total_candidates - processed_candidates)
            eta_seconds = remaining_candidates / avg_speed if avg_speed > 0 else 0
            eta_str = format_seconds(eta_seconds) if eta_seconds > 0 else "--:--:--"
            
            status_slot.markdown(render_status(1, slide_id, avg_speed, eta_str), unsafe_allow_html=True)
            
            if is_presenter:
                candidate_frame = None
                candidate_gray = None
                candidate_hash = None
                candidate_edge = None
                stable_count = 0
                continue
                
            if prev_accepted_frame is None:
                slide_id += 1
                slide_filename = f"slide_{slide_id:04d}.jpg"
                slide_filepath = os.path.join(SLIDES_DIR, slide_filename)
                cv2.imwrite(slide_filepath, current_frame)
                
                slide_metadata_list.append({
                    "slide_id": slide_id,
                    "timestamp": timestamp_str,
                    "change_type": "new_slide",
                    "file": f"slides/{slide_filename}"
                })
                
                prev_accepted_frame = current_frame.copy()
                prev_accepted_gray = gray_curr
                prev_accepted_hash = dhash_curr
                prev_accepted_edge = edge_curr
                continue
                
            is_sig, score, details = detector.analyze_change_precomputed(
                gray_curr, dhash_curr, edge_curr, face_boxes,
                prev_accepted_gray, prev_accepted_hash, prev_accepted_edge,
                sensitive=True
            )
            
            if is_sig:
                if candidate_frame is None:
                    candidate_frame = current_frame.copy()
                    candidate_gray = gray_curr
                    candidate_hash = dhash_curr
                    candidate_edge = edge_curr
                    candidate_timestamp = timestamp_str
                    stable_count = 0
                else:
                    is_cand_diff, cand_score, cand_details = detector.analyze_change_precomputed(
                        gray_curr, dhash_curr, edge_curr, face_boxes,
                        candidate_gray, candidate_hash, candidate_edge,
                        sensitive=False
                    )
                    if not is_cand_diff:
                        stable_count += 1
                        if stable_count >= persistence_threshold:
                            slide_id += 1
                            slide_filename = f"slide_{slide_id:04d}.jpg"
                            slide_filepath = os.path.join(SLIDES_DIR, slide_filename)
                            cv2.imwrite(slide_filepath, candidate_frame)
                            
                            change_type = classifier.classify(candidate_frame, prev_accepted_frame, details)
                            slide_metadata_list.append({
                                "slide_id": slide_id,
                                "timestamp": candidate_timestamp,
                                "change_type": change_type,
                                "file": f"slides/{slide_filename}"
                            })
                            
                            prev_accepted_frame = candidate_frame.copy()
                            prev_accepted_gray = candidate_gray
                            prev_accepted_hash = candidate_hash
                            prev_accepted_edge = candidate_edge
                            
                            candidate_frame = None
                            candidate_gray = None
                            candidate_hash = None
                            candidate_edge = None
                            candidate_timestamp = None
                            stable_count = 0
                    else:
                        candidate_frame = current_frame.copy()
                        candidate_gray = gray_curr
                        candidate_hash = dhash_curr
                        candidate_edge = edge_curr
                        candidate_timestamp = timestamp_str
                        stable_count = 0
            else:
                if candidate_frame is not None:
                    candidate_frame = None
                    candidate_gray = None
                    candidate_hash = None
                    candidate_edge = None
                    candidate_timestamp = None
                    stable_count = 0
                    
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(slide_metadata_list, f, indent=4)
            
        # Step 2: OCR & PPT Generation
        status_slot.markdown(render_status(2), unsafe_allow_html=True)
        ocr_service.process_slides_parallel(
            slides_dir=SLIDES_DIR,
            metadata_path=METADATA_PATH,
            output_results_path=OCR_RESULTS_PATH,
            output_summary_path=OCR_SUMMARY_PATH
        )
        
        ppt_gen = PPTGenerator(output_dir=OUTPUT_DIR)
        ppt_gen.generate()
        
        # Step 3: PDF Export
        status_slot.markdown(render_status(3), unsafe_allow_html=True)
        try:
            pdf_gen = PDFGenerator(output_dir=OUTPUT_DIR)
            pdf_gen.convert(PPTX_PATH, PDF_PATH)
        except Exception as pdf_err:
            logger.warning(f"PDF export warning: {pdf_err}")
            
        status_slot.markdown("<div style='color:#10b981; font-weight:bold; margin-bottom:20px;'>✨ Lecture processing completed successfully!</div>", unsafe_allow_html=True)
        time.sleep(1)
        
        st.session_state.processed = check_existing_outputs()
        st.session_state.processing = False
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.session_state.processing = False
        st.session_state.processed = False


# Results & Downloads Display
if st.session_state.processed and not st.session_state.processing:
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            slides = json.load(f)
        
        video_title = "Educational Lecture Presentation"
        video_duration_str = "N/A"
        if os.path.exists(VIDEO_METADATA_PATH):
            with open(VIDEO_METADATA_PATH, "r", encoding="utf-8") as f:
                v_meta = json.load(f)
                video_title = v_meta.get("title", video_title)
                duration_seconds = v_meta.get("duration", 0.0)
                if duration_seconds > 0:
                    video_duration_str = format_seconds(duration_seconds)
            
        st.markdown("---")
        st.header(video_title)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Slides Captured", len(slides))
        with col2:
            st.metric("Lecture Duration", video_duration_str)
            
        st.subheader("📥 Downloads")
        col_ppt, col_pdf = st.columns(2)
        with col_ppt:
            if os.path.exists(PPTX_PATH):
                with open(PPTX_PATH, "rb") as f:
                    st.download_button(
                        label="Download PPTX Presentation",
                        data=f,
                        file_name="lecture_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        type="primary",
                        use_container_width=True
                    )
        with col_pdf:
            if os.path.exists(PDF_PATH):
                with open(PDF_PATH, "rb") as f:
                    st.download_button(
                        label="Download PDF Document",
                        data=f,
                        file_name="lecture_presentation.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    
        st.markdown("---")
        with st.expander("🖼️ View Extracted Slide Screenshots Gallery", expanded=True):
            cols = st.columns(3)
            for idx, s in enumerate(slides):
                with cols[idx % 3]:
                    thumb_path = os.path.join(OUTPUT_DIR, s["file"])
                    if os.path.exists(thumb_path):
                        st.image(thumb_path, use_container_width=True)
                        st.caption(f"Slide {s['slide_id']} ({s['timestamp']})")
                        
    except Exception as e:
        st.error(f"Failed to load output files: {e}")
