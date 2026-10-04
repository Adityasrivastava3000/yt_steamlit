import logging
import os
from typing import List, Tuple, Optional, Any

# Setup logging
ocr_logger = logging.getLogger("ocr_service.paddle")

class PaddleOCRService:
    """Service to perform OCR on slide images using PaddleOCR."""

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Initialize PaddleOCR engine.
        
        :param languages: List of language codes (currently defaults to 'en').
        """
        # PaddleOCR uses specific language tags, mapping standard defaults.
        lang = "en"
        if languages and "ch" in languages:
            lang = "ch"  # Support Chinese if explicitly provided
            
        ocr_logger.info(f"Initializing PaddleOCR reader with language: {lang}")
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR with orientation detection
            self.reader = PaddleOCR(use_textline_orientation=True, lang=lang)
            ocr_logger.info("PaddleOCR reader initialized successfully.")
        except Exception as e:
            ocr_logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise e

    def readtext(self, image_path: str) -> List[Tuple[List[List[int]], str, float]]:
        """
        Extract text and bounding boxes from an image in a format compatible with EasyOCR output.
        
        Format: [([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, confidence), ...]
        """
        if not os.path.exists(image_path):
            ocr_logger.error(f"Image not found at: {image_path}")
            return []

        try:
            result = self.reader.ocr(image_path)
            raw_results = []

            if result and result[0]:
                page_res = result[0]
                # Handle PaddleX/PaddleOCR v3.7.0 dictionary output
                if isinstance(page_res, dict):
                    rec_texts = page_res.get("rec_texts", [])
                    rec_scores = page_res.get("rec_scores", [])
                    dt_polys = page_res.get("dt_polys", [])
                    
                    for i in range(len(rec_texts)):
                        text = rec_texts[i]
                        conf = rec_scores[i] if i < len(rec_scores) else 0.0
                        poly = dt_polys[i] if i < len(dt_polys) else []
                        
                        # Convert numpy polygon to list
                        if hasattr(poly, "tolist"):
                            poly_list = poly.tolist()
                        else:
                            poly_list = list(poly)
                            
                        raw_results.append((poly_list, text, conf))
                
                # Handle standard list format: [ [ [ [x1,y1],... ], (text, conf) ], ... ]
                elif isinstance(page_res, list):
                    for line in page_res:
                        if isinstance(line, (list, tuple)) and len(line) > 1:
                            bbox = line[0]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) > 1:
                                text = text_info[0]
                                conf = text_info[1]
                                raw_results.append((bbox, text, conf))

            return raw_results

        except Exception as e:
            ocr_logger.error(f"PaddleOCR readtext failed for {image_path}: {str(e)}")
            return []
