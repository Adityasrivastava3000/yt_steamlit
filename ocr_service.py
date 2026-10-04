import os
import json
import logging
import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any, Tuple, Optional

# Setup OCR-specific logger
os.makedirs("logs", exist_ok=True)
ocr_logger = logging.getLogger("ocr_service")
ocr_logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if the service is loaded multiple times
if not ocr_logger.handlers:
    # fh = logging.FileHandler(os.path.join("logs", "ocr.log"), mode="w")
    # fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    # ocr_logger.addHandler(fh)
    pass


class OCRService:
    """Service to perform OCR on slide images, preserving reading order. Configurable via OCR_ENGINE env var."""

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize the OCRService.

        :param languages: List of language codes for OCR (default: ['en']).
        """
        self.languages = languages or ["en"]
        self.engine = os.environ.get("OCR_ENGINE", "easyocr").lower()

        if self.engine == "paddleocr":
            ocr_logger.info("Configured to use PaddleOCR engine.")
            from services.paddle_ocr_service import PaddleOCRService
            self.paddle_service = PaddleOCRService(self.languages)
            self.reader = None
        else:
            ocr_logger.info(f"Configured to use EasyOCR reader with languages: {self.languages}")
            try:
                self.reader = easyocr.Reader(self.languages, gpu=True)
                ocr_logger.info("EasyOCR reader initialized successfully (GPU enabled if available).")
            except Exception as e:
                ocr_logger.warning(f"Failed to initialize EasyOCR with GPU. Retrying with CPU: {str(e)}")
                self.reader = easyocr.Reader(self.languages, gpu=False)
                ocr_logger.info("EasyOCR reader initialized successfully on CPU.")
            self.paddle_service = None

    def sort_ocr_results(self, results: List[Tuple[List[List[int]], str, float]]) -> List[Dict[str, Any]]:
        """
        Sort EasyOCR bounding box results to preserve reading order (top-to-bottom, left-to-right).
        
        :param results: EasyOCR readtext raw output: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
        :return: List of sorted dictionaries with text, confidence, and box coordinates.
        """
        if not results:
            return []

        boxes_with_info = []
        for bbox, text, conf in results:
            # bbox is 4 corners: top-left, top-right, bottom-right, bottom-left
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            centroid_y = (y_min + y_max) / 2.0
            centroid_x = (x_min + x_max) / 2.0
            height = max(1, y_max - y_min)

            boxes_with_info.append({
                "bbox": bbox,
                "text": text.strip(),
                "confidence": float(conf),
                "y_min": y_min,
                "y_max": y_max,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "height": height
            })

        # Sort initially by Y centroid (top to bottom)
        boxes_with_info.sort(key=lambda b: b["centroid_y"])

        # Group into lines/rows based on overlapping Y centroids
        rows = []
        for box in boxes_with_info:
            placed = False
            for row in rows:
                row_len = len(row)
                if row_len is None or row_len <= 0:
                    row_len = 1
                row_avg_y = sum(b["centroid_y"] for b in row) / row_len
                row_avg_height = sum(b["height"] for b in row) / row_len
                # If the current box centroid Y overlaps within 50% of the row height
                if abs(box["centroid_y"] - row_avg_y) < (row_avg_height * 0.5):
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        # Sort each row left-to-right by X centroid
        sorted_boxes = []
        for row in rows:
            row.sort(key=lambda b: b["centroid_x"])
            sorted_boxes.extend(row)

        return sorted_boxes

    def process_slides(
        self,
        slides_dir: str,
        metadata_path: str,
        output_results_path: str,
        output_summary_path: str
    ) -> bool:
        """
        Runs OCR on all slides in parallel for speed.
        """
        return self.process_slides_parallel(
            slides_dir=slides_dir,
            metadata_path=metadata_path,
            output_results_path=output_results_path,
            output_summary_path=output_summary_path
        )

    def process_slides_parallel(
        self,
        slides_dir: str,
        metadata_path: str,
        output_results_path: str,
        output_summary_path: str,
        max_workers: int = 4
    ) -> bool:
        """
        Runs OCR on all slides sequentially to avoid PyTorch/EasyOCR concurrency crashes on macOS.
        """
        # Load slide metadata
        if not os.path.exists(metadata_path):
            ocr_logger.error(f"Slide metadata not found at: {metadata_path}")
            return False

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            ocr_logger.error(f"Failed to read slide metadata: {str(e)}")
            return False
            
        ocr_logger.info(f"Starting OCR processing for {len(metadata)} slides...")
        
        ocr_results = []
        failed_slides = []
        
        for entry in metadata:
            slide_id = entry.get("slide_id")
            timestamp = entry.get("timestamp")
            relative_image_path = entry.get("file")
            image_name = os.path.basename(relative_image_path)
            image_path = os.path.join(os.path.dirname(metadata_path), relative_image_path)
            
            if not os.path.exists(image_path):
                failed_slides.append(image_name)
                continue
                
            try:
                img = cv2.imread(image_path)
                if img is None or img.size == 0:
                    failed_slides.append(image_name)
                    continue
            except Exception:
                failed_slides.append(image_name)
                continue
                
            try:
                if self.paddle_service:
                    raw_results = self.paddle_service.readtext(image_path)
                else:
                    raw_results = self.reader.readtext(image_path)
                sorted_boxes = self.sort_ocr_results(raw_results)
                slide_text = "\n".join([box["text"] for box in sorted_boxes])
                box_count = len(sorted_boxes)
                if box_count is None or box_count <= 0:
                    box_count = 1
                slide_confidence = sum(box["confidence"] for box in sorted_boxes) / box_count if sorted_boxes else 0.0
                
                ocr_results.append({
                    "slide_id": slide_id,
                    "timestamp": timestamp,
                    "image": image_name,
                    "confidence": round(slide_confidence, 4),
                    "text": slide_text
                })
            except Exception as e:
                ocr_logger.error(f"OCR failed for {image_name}: {e}")
                failed_slides.append(image_name)
                    
        # Sort results by slide_id to maintain chronological order
        ocr_results.sort(key=lambda x: x["slide_id"])
        
        # Post-hoc duplicate filtering pass
        filtered_ocr_results = []
        import difflib
        import imagehash
        from PIL import Image
        
        def get_ocr_similarity(t1, t2):
            t1_clean = "".join(t1.split()).lower()
            t2_clean = "".join(t2.split()).lower()
            if not t1_clean and not t2_clean:
                return 1.0
            if not t1_clean or not t2_clean:
                return 0.0
            return difflib.SequenceMatcher(None, t1_clean, t2_clean).ratio()
            
        def compute_ssim_fast(img1_path, img2_path):
            try:
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    return 0.0
                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                C1 = 6.5025
                C2 = 58.5225
                img1 = img1.astype(np.float32)
                img2 = img2.astype(np.float32)
                mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
                mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
                sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
                sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
                num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
                den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
                ssim_map = num / den
                return float(np.mean(ssim_map))
            except Exception:
                return 0.0

        def compute_hash_similarity(img1_path, img2_path):
            try:
                pil1 = Image.open(img1_path)
                pil2 = Image.open(img2_path)
                h1 = imagehash.dhash(pil1)
                h2 = imagehash.dhash(pil2)
                diff = (h1 - h2) / 64.0
                return 1.0 - diff
            except Exception:
                return 0.0

        for r_entry in ocr_results:
            if not filtered_ocr_results:
                filtered_ocr_results.append(r_entry)
            else:
                prev_entry = filtered_ocr_results[-1]
                p_path = os.path.join(slides_dir, prev_entry["image"])
                c_path = os.path.join(slides_dir, r_entry["image"])
                
                # Check conditions
                ocr_sim = get_ocr_similarity(r_entry["text"], prev_entry["text"])
                ssim_val = compute_ssim_fast(c_path, p_path)
                hash_sim = compute_hash_similarity(c_path, p_path)
                
                is_duplicate = (ocr_sim > 0.97) and (ssim_val > 0.98) and (hash_sim > 0.95)
                
                if is_duplicate:
                    ocr_logger.info(f"Filtered post-hoc duplicate slide: {r_entry['image']} (sims: OCR={ocr_sim:.3f}, SSIM={ssim_val:.3f}, Hash={hash_sim:.3f})")
                    # Delete the slide image file
                    try:
                        if os.path.exists(c_path):
                            os.remove(c_path)
                    except Exception as de:
                        ocr_logger.warning(f"Failed to delete duplicate slide file: {de}")
                else:
                    filtered_ocr_results.append(r_entry)

        # Update metadata.json to match filtered slides
        metadata_dict = {os.path.basename(e["file"]): e for e in metadata}
        filtered_metadata = []
        
        # Re-index slide_id for the remaining slides to keep them sequential
        new_slide_id = 1
        final_ocr_results = []
        for r_entry in filtered_ocr_results:
            orig_filename = r_entry["image"]
            meta_entry = metadata_dict.get(orig_filename)
            if meta_entry:
                new_filename = f"slide_{new_slide_id:04d}.jpg"
                # Rename the file physically if it differs
                if orig_filename != new_filename:
                    old_path = os.path.join(slides_dir, orig_filename)
                    new_path = os.path.join(slides_dir, new_filename)
                    try:
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                    except Exception as re:
                        ocr_logger.warning(f"Failed to rename slide file {orig_filename} to {new_filename}: {re}")
                
                meta_entry["slide_id"] = new_slide_id
                meta_entry["file"] = f"slides/{new_filename}"
                filtered_metadata.append(meta_entry)
                
                r_entry["slide_id"] = new_slide_id
                r_entry["image"] = new_filename
                final_ocr_results.append(r_entry)
                
                new_slide_id += 1

        # Save the updated metadata.json
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(filtered_metadata, f, indent=4)
        except Exception as e:
            ocr_logger.error(f"Failed to write filtered metadata.json: {e}")
            
        ocr_results = final_ocr_results
        
        total_words = sum(len(e["text"].split()) for e in ocr_results)
        total_confidence = sum(e["confidence"] for e in ocr_results)
        successful_slides_count = len(ocr_results)
        if successful_slides_count is None or successful_slides_count <= 0:
            successful_slides_count = 1
        avg_confidence = (total_confidence / successful_slides_count) if len(ocr_results) > 0 else 0.0
        
        summary = {
            "total_slides": len(ocr_results),
            "total_words": total_words,
            "average_confidence": round(avg_confidence, 4),
            "failed_slides": failed_slides
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
        try:
            with open(output_results_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, indent=4)
        except Exception as e:
            ocr_logger.error(f"Failed to save OCR results JSON: {str(e)}")
            
        try:
            with open(output_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            ocr_logger.error(f"Failed to save OCR summary JSON: {str(e)}")
            
        return len(failed_slides) == 0
