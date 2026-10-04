import cv2
import logging
import numpy as np
from PIL import Image
import imagehash
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SlideDetector:
    """Detects educationally significant changes between video frames and filters presenter talking-heads."""

    def __init__(
        self,
        sensitivity: str = "medium",
        ignore_regions: Optional[List[Tuple[float, float, float, float]]] = None,
        ignore_bottom_ratio: float = 0.08,
        ignore_top_ratio: float = 0.0,
        grid_rows: int = 8,
        grid_cols: int = 8,
        edge_threshold: int = 30,
        # Combined score weights
        w_ssim: float = 0.4,
        w_hash: float = 0.3,
        w_pixel: float = 0.15,
        w_edge: float = 0.15,
    ):
        self.sensitivity = sensitivity.lower()
        self.ignore_regions = ignore_regions or []
        self.ignore_bottom_ratio = ignore_bottom_ratio
        self.ignore_top_ratio = ignore_top_ratio
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Weights and thresholds
        self.w_ssim = w_ssim
        self.w_hash = w_hash
        self.w_pixel = w_pixel
        self.w_edge = w_edge
        self.edge_threshold = edge_threshold

        # Initialize Haar Cascade face detectors (frontal and profile)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        # Temporal face tracking state
        self.last_face_boxes = []
        self.frames_since_face = 999

        # Set thresholds based on sensitivity preset
        self._apply_sensitivity_preset()

    def _apply_sensitivity_preset(self) -> None:
        """Applies sensitivity thresholds for change detection levels."""
        if self.sensitivity == "low":
            self.ssim_threshold = 0.20
            self.hash_threshold = 0.12
            self.local_edge_threshold = 0.15
            self.pixel_diff_threshold = 35
            self.min_pixel_change_ratio = 0.02
        elif self.sensitivity == "high":
            self.ssim_threshold = 0.08
            self.hash_threshold = 0.04
            self.local_edge_threshold = 0.04
            self.pixel_diff_threshold = 18
            self.min_pixel_change_ratio = 0.005
        else:  # medium
            self.ssim_threshold = 0.15
            self.hash_threshold = 0.08
            self.local_edge_threshold = 0.08
            self.pixel_diff_threshold = 25
            self.min_pixel_change_ratio = 0.01

        logger.info(
            f"Configured detector with sensitivity '{self.sensitivity}': "
            f"ssim_threshold={self.ssim_threshold:.3f}, "
            f"hash_threshold={self.hash_threshold:.3f}, "
            f"local_edge_threshold={self.local_edge_threshold:.3f}, "
            f"min_pixel_change_ratio={self.min_pixel_change_ratio:.5f}"
        )

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects frontal and profile faces in the frame. Returns bounding boxes as [(x, y, w, h)]."""
        orig_height, orig_width = frame.shape[:2]
        
        # Downscale for faster face detection
        target_width = 320
        scale = target_width / orig_width
        target_height = int(orig_height * scale)
        
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (target_width, target_height))
        min_size = int(target_height * 0.05)
        
        # Frontal faces
        faces = self.face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size)
        )
        
        # Profile faces
        profile_faces = self.profile_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(min_size, min_size)
        )
        
        all_faces_small = list(faces)
        for pf in profile_faces:
            # Avoid adding duplicates if they overlap significantly
            overlap = False
            for f in all_faces_small:
                if abs(pf[0] - f[0]) < f[2] and abs(pf[1] - f[1]) < f[3]:
                    overlap = True
                    break
            if not overlap:
                all_faces_small.append(pf)
                
        # Scale boxes back to original coordinates
        all_faces = []
        for (x, y, w, h) in all_faces_small:
            x_orig = int(round(x / scale))
            y_orig = int(round(y / scale))
            w_orig = int(round(w / scale))
            h_orig = int(round(h / scale))
            all_faces.append((x_orig, y_orig, w_orig, h_orig))
            
        if all_faces:
            self.last_face_boxes = [tuple(f) for f in all_faces]
            self.frames_since_face = 0
        else:
            self.frames_since_face += 1
            
        # Stabilize mask if face was detected recently (within last 3 steps)
        if not all_faces and self.frames_since_face <= 3:
            return self.last_face_boxes
            
        return [tuple(f) for f in all_faces]

    def compute_edge_density(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Computes the density of sharp edges in the image within the mask."""
        edges = self.compute_text_edges(gray)
        if mask is not None:
            masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
            edge_pixels = np.sum(masked_edges == 255)
            total_pixels = np.sum(mask == 1)
        else:
            edge_pixels = np.sum(edges == 255)
            total_pixels = gray.size
            
        if total_pixels == 0:
            return 0.0
        return float(edge_pixels / total_pixels)

    def compute_educational_score_with_details(self, frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Computes an educational score based on text, shapes, diagrams, code, charts,
        and other visual content, returning the total score and the breakdown of sub-scores.
        """
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Text Score (using horizontal projections)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            row_sums = np.sum(binary, axis=1)
            max_sum = np.max(row_sums) if len(row_sums) > 0 else 0
            
            text_score = 0.0
            peaks = 0
            if max_sum > 0:
                in_peak = False
                for val in row_sums:
                    is_active = val > (max_sum * 0.08)
                    if is_active and not in_peak:
                        in_peak = True
                        peaks += 1
                    elif not is_active and in_peak:
                        in_peak = False
                text_score = min(10.0, peaks * 1.5)
            
            # 2. Edges and Contours for Diagram & Shapes
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            geometric_shapes = 0
            for c in contours:
                approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
                if 3 <= len(approx) <= 8 and cv2.contourArea(c) > 150:
                    geometric_shapes += 1
            diagram_score = min(10.0, geometric_shapes * 2.0)
            
            # 3. Hough Lines for Chart & Graph Axis
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=10)
            num_lines = len(lines) if lines is not None else 0
            horizontal_lines = 0
            vertical_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dx > dy * 4:
                        horizontal_lines += 1
                    elif dy > dx * 4:
                        vertical_lines += 1
            
            chart_score = 0.0
            if num_lines >= 5:
                chart_score += 4.0
            if horizontal_lines >= 2 and vertical_lines >= 2:
                chart_score += 6.0
                
            # 4. Code Block Score
            code_score = 0.0
            if text_score > 3.0:
                starts = []
                for r in range(h):
                    if row_sums[r] > (max_sum * 0.1):
                        cols_active = np.where(binary[r] > 0)[0]
                        if len(cols_active) > 0:
                            starts.append(cols_active[0])
                if len(starts) >= 4:
                    indent_std = np.std(starts)
                    if indent_std > 10.0:
                        code_score = 8.0
                        
            # 5. Table Grid Score
            table_score = 0.0
            if horizontal_lines >= 3 and vertical_lines >= 3:
                table_score = 8.0
                
            # 6. Formula Score
            moderate_contours = sum(1 for c in contours if 50 < cv2.contourArea(c) < 5000)
            formula_score = min(5.0, moderate_contours * 0.2)
            
            # 7. Roadmap Score
            roadmap_score = 5.0 if (geometric_shapes >= 3 and text_score > 2.0) else 0.0
            
            # 8. Visual Explanation Score
            visual_explanation_score = 5.0 if (geometric_shapes >= 2 and moderate_contours > 10) else 0.0
            
            total_score = (
                text_score +
                diagram_score +
                chart_score +
                code_score +
                table_score +
                formula_score +
                roadmap_score +
                visual_explanation_score
            )
            
            details = {
                "text_score": text_score,
                "diagram_score": diagram_score,
                "chart_score": chart_score,
                "code_score": code_score,
                "table_score": table_score,
                "formula_score": formula_score,
                "roadmap_score": roadmap_score,
                "visual_explanation_score": visual_explanation_score
            }
            return total_score, details
        except Exception as e:
            logger.error(f"Error computing educational score with details: {e}")
            return 0.0, {}

    def compute_educational_score(self, frame: np.ndarray) -> float:
        """
        Computes an educational score based on text, shapes, diagrams, code, charts,
        and other visual content.
        """
        score, _ = self.compute_educational_score_with_details(frame)
        return score

    def is_presenter_only_scene(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]]) -> bool:
        """
        Determines if the frame represents a presenter-only scene (talking head) or camera-only scene.
        """
        height, width = frame.shape[:2]
        frame_area = height * width
        if frame_area <= 0:
            frame_area = 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute educational content score and details
        edu_score, details = self.compute_educational_score_with_details(frame)

        # 1. If there are faces, check if they are close-up talking heads
        if face_boxes:
            largest_face_area = max(w * h for (x, y, w, h) in face_boxes)
            largest_face_ratio = largest_face_area / frame_area
            
            # REMOVE only if:
            # Face occupies most of frame (ratio > 0.08)
            # AND Educational content score is very low (edu_score < 4.0)
            # AND No meaningful text exists (text_score < 2.0)
            # AND No diagrams exist (diagram_score < 1.0)
            # AND No charts exist (chart_score < 1.0)
            # AND No code exists (code_score < 1.0)
            face_occupies_most = largest_face_ratio > 0.08
            edu_score_low = edu_score < 4.0
            no_text = details.get("text_score", 0.0) < 2.0
            no_diagrams = details.get("diagram_score", 0.0) < 1.0
            no_charts = details.get("chart_score", 0.0) < 1.0
            no_code = details.get("code_score", 0.0) < 1.0

            if (face_occupies_most and edu_score_low and no_text and no_diagrams and no_charts and no_code):
                logger.info(f"Filtered presenter-only talking head: face_ratio={largest_face_ratio:.4f}, edu_score={edu_score:.1f}")
                return True
            else:
                logger.info(
                    f"Kept presenter frame: face_ratio={largest_face_ratio:.4f}, edu_score={edu_score:.1f} "
                    f"(details: text={details.get('text_score', 0.0)}, diag={details.get('diagram_score', 0.0)}, "
                    f"chart={details.get('chart_score', 0.0)}, code={details.get('code_score', 0.0)})"
                )
                return False
            
        # 2. General camera-only/blank scene check (low edge density overall)
        overall_density = self.compute_edge_density(gray)
        if overall_density < 0.005:
            logger.info(f"Filtered low-density camera/blank scene: density={overall_density:.4f}")
            return True

        return False

    def _get_ignore_mask(self, height: int, width: int, face_boxes: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
        """
        Generates a binary mask of regions to process (1 = keep, 0 = ignore).
        Dynamically masks out detected presenter faces and torso column to avoid tracking gestures.
        """
        mask = np.ones((height, width), dtype=np.uint8)

        # Apply bottom controls ignore
        if self.ignore_bottom_ratio > 0:
            ymin = int(height * (1.0 - self.ignore_bottom_ratio))
            mask[ymin:, :] = 0

        # Apply top header ignore
        if self.ignore_top_ratio > 0:
            ymax = int(height * self.ignore_top_ratio)
            mask[:ymax, :] = 0

        # Apply custom ignore regions
        for ymin_norm, xmin_norm, ymax_norm, xmax_norm in self.ignore_regions:
            ymin = int(ymin_norm * height)
            xmin = int(xmin_norm * width)
            ymax = int(ymax_norm * height)
            xmax = int(xmax_norm * width)
            mask[ymin:ymax, xmin:xmax] = 0

        # Dynamically ignore presenter regions (head and full column down to screen bottom)
        if face_boxes:
            for (x, y, w, h) in face_boxes:
                # Add padding to cover shoulders and hair
                pad_w = int(w * 0.3)
                pad_h = int(h * 0.5)
                ymin = max(0, y - pad_h)
                xmin = max(0, x - pad_w)
                ymax = height  # Extend to bottom of the frame to mask out presenter gestures
                xmax = min(width, x + w + pad_w)
                mask[ymin:ymax, xmin:xmax] = 0

        return mask

    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Structural Similarity Index (SSIM) between two grayscale images."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-10)
        return float(np.mean(ssim_map))

    def compute_image_hash_diff(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """dhash Hamming distance normalized difference."""
        if len(img1.shape) == 2:
            pil1 = Image.fromarray(img1)
            pil2 = Image.fromarray(img2)
        else:
            pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        hash1 = imagehash.dhash(pil1)
        hash2 = imagehash.dhash(pil2)

        return (hash1 - hash2) / 64.0

    def compute_text_edges(self, gray: np.ndarray) -> np.ndarray:
        """Compute text-like edges using Laplacian filter."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        abs_laplacian = np.abs(laplacian)
        _, edge_mask = cv2.threshold(
            abs_laplacian.astype(np.uint8), self.edge_threshold, 255, cv2.THRESH_BINARY
        )
        return edge_mask

    def ocr_validation_hook(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> bool:
        return True

    def analyze_change(
        self,
        current_frame: np.ndarray,
        prev_accepted_frame: np.ndarray,
        face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        sensitive: bool = False
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyzes change, ignoring regions covered by the presenter.
        """
        h, w = current_frame.shape[:2]
        ignore_mask = self._get_ignore_mask(h, w, face_boxes)
        total_valid_pixels = np.sum(ignore_mask)

        # Fallback if the face occupies 100% of the screen
        if total_valid_pixels == 0:
            return False, 0.0, {"change_level": "None"}

        # Convert to grayscale
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_accepted_frame, cv2.COLOR_BGR2GRAY)

        # 1. Image Hash (calculated first)
        hash_diff = self.compute_image_hash_diff(current_frame, prev_accepted_frame)

        # 2. SSIM surrogate (avoiding heavy SSIM computation for speed)
        ssim_diff = hash_diff

        # 3. Pixel Difference
        pixel_abs_diff = cv2.absdiff(gray_curr, gray_prev)
        _, pixel_thresh = cv2.threshold(
            pixel_abs_diff, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY
        )
        masked_pixel_diff = cv2.bitwise_and(pixel_thresh, pixel_thresh, mask=ignore_mask)
        changed_pixels_count = np.sum(masked_pixel_diff == 255)
        pixel_change_ratio = changed_pixels_count / total_valid_pixels

        # 4. Edge density
        edge_curr = self.compute_text_edges(gray_curr)
        edge_prev = self.compute_text_edges(gray_prev)
        edge_diff = cv2.subtract(edge_curr, edge_prev)
        masked_edge_diff = cv2.bitwise_and(edge_diff, edge_diff, mask=ignore_mask)
        new_edge_pixels = np.sum(masked_edge_diff == 255)
        edge_change_ratio = new_edge_pixels / total_valid_pixels

        # 5. Grid-based Localized change
        grid_h = h // self.grid_rows
        grid_w = w // self.grid_cols
        max_local_edge_change = 0.0
        local_change_detected = False

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y_start = r * grid_h
                y_end = (r + 1) * grid_h if r < self.grid_rows - 1 else h
                x_start = c * grid_w
                x_end = (c + 1) * grid_w if c < self.grid_cols - 1 else w

                cell_mask = ignore_mask[y_start:y_end, x_start:x_end]
                cell_valid_pixels = np.sum(cell_mask)
                if cell_valid_pixels < (grid_h * grid_w * 0.25):
                    continue

                cell_edge_curr = edge_curr[y_start:y_end, x_start:x_end]
                cell_edge_prev = edge_prev[y_start:y_end, x_start:x_end]
                
                cell_edge_diff = cv2.subtract(cell_edge_curr, cell_edge_prev)
                cell_edge_diff_masked = cv2.bitwise_and(cell_edge_diff, cell_edge_diff, mask=cell_mask)
                
                cell_edge_added = np.sum(cell_edge_diff_masked == 255)
                cell_edge_added_ratio = cell_edge_added / cell_valid_pixels

                max_local_edge_change = max(max_local_edge_change, cell_edge_added_ratio)

                if cell_edge_added_ratio > self.local_edge_threshold:
                    local_change_detected = True

        is_ssim_trigger = ssim_diff > self.ssim_threshold
        is_hash_trigger = hash_diff > self.hash_threshold
        is_local_trigger = local_change_detected
        
        if sensitive:
            is_significant_change = (
                pixel_change_ratio > 0.003
                and (hash_diff > 0.02 or local_change_detected)
            )
        else:
            is_significant_change = (
                pixel_change_ratio > self.min_pixel_change_ratio
                and (is_hash_trigger or is_local_trigger)
            )

        change_level = "None"
        if is_significant_change:
            if is_ssim_trigger and ssim_diff > (self.ssim_threshold * 1.5):
                change_level = "Level 1 (Large slide change)"
            elif is_hash_trigger:
                change_level = "Level 2 (Medium content addition)"
            else:
                change_level = "Level 3 (Small educational addition)"

        final_score = max(ssim_diff, hash_diff)

        details = {
            "ssim_diff": ssim_diff,
            "hash_diff": hash_diff,
            "pixel_change_ratio": pixel_change_ratio,
            "edge_change_ratio": edge_change_ratio,
            "max_local_edge_change": max_local_edge_change,
            "is_ssim_trigger": is_ssim_trigger,
            "is_hash_trigger": is_hash_trigger,
            "is_local_trigger": is_local_trigger,
            "change_level": change_level,
            "pixel_diff_mask": masked_pixel_diff
        }

        return is_significant_change, final_score, details

    def analyze_change_precomputed(
        self,
        gray_curr: np.ndarray,
        dhash_curr: Any,
        edge_curr: np.ndarray,
        face_boxes_curr: List[Tuple[int, int, int, int]],
        gray_prev: np.ndarray,
        dhash_prev: Any,
        edge_prev: np.ndarray,
        sensitive: bool = False
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyzes change using precomputed inputs, avoiding re-calculation of edge filters, grayscale conversion, and hashing.
        """
        h, w = gray_curr.shape[:2]
        ignore_mask = self._get_ignore_mask(h, w, face_boxes_curr)
        total_valid_pixels = np.sum(ignore_mask)

        # Fallback if the face occupies 100% of the screen
        if total_valid_pixels == 0:
            return False, 0.0, {"change_level": "None"}

        # 1. Image Hash (dhash Hamming distance normalized difference)
        hash_diff = (dhash_curr - dhash_prev) / 64.0
        ssim_diff = hash_diff

        # 2. Pixel Difference
        pixel_abs_diff = cv2.absdiff(gray_curr, gray_prev)
        _, pixel_thresh = cv2.threshold(
            pixel_abs_diff, self.pixel_diff_threshold, 255, cv2.THRESH_BINARY
        )
        masked_pixel_diff = cv2.bitwise_and(pixel_thresh, pixel_thresh, mask=ignore_mask)
        changed_pixels_count = np.sum(masked_pixel_diff == 255)
        pixel_change_ratio = changed_pixels_count / total_valid_pixels

        # 3. Edge density
        edge_diff = cv2.subtract(edge_curr, edge_prev)
        masked_edge_diff = cv2.bitwise_and(edge_diff, edge_diff, mask=ignore_mask)
        new_edge_pixels = np.sum(masked_edge_diff == 255)
        edge_change_ratio = new_edge_pixels / total_valid_pixels

        # 4. Grid-based Localized change
        grid_h = h // self.grid_rows
        grid_w = w // self.grid_cols
        max_local_edge_change = 0.0
        local_change_detected = False

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y_start = r * grid_h
                y_end = (r + 1) * grid_h if r < self.grid_rows - 1 else h
                x_start = c * grid_w
                x_end = (c + 1) * grid_w if c < self.grid_cols - 1 else w

                cell_mask = ignore_mask[y_start:y_end, x_start:x_end]
                cell_valid_pixels = np.sum(cell_mask)
                if cell_valid_pixels < (grid_h * grid_w * 0.25):
                    continue

                cell_edge_curr = edge_curr[y_start:y_end, x_start:x_end]
                cell_edge_prev = edge_prev[y_start:y_end, x_start:x_end]
                
                cell_edge_diff = cv2.subtract(cell_edge_curr, cell_edge_prev)
                cell_edge_diff_masked = cv2.bitwise_and(cell_edge_diff, cell_edge_diff, mask=cell_mask)
                
                cell_edge_added = np.sum(cell_edge_diff_masked == 255)
                cell_edge_added_ratio = cell_edge_added / cell_valid_pixels

                max_local_edge_change = max(max_local_edge_change, cell_edge_added_ratio)

                if cell_edge_added_ratio > self.local_edge_threshold:
                    local_change_detected = True

        is_ssim_trigger = ssim_diff > self.ssim_threshold
        is_hash_trigger = hash_diff > self.hash_threshold
        is_local_trigger = local_change_detected
        
        if sensitive:
            is_significant_change = (
                pixel_change_ratio > 0.003
                and (hash_diff > 0.02 or local_change_detected)
            )
        else:
            is_significant_change = (
                pixel_change_ratio > self.min_pixel_change_ratio
                and (is_hash_trigger or is_local_trigger)
            )

        change_level = "None"
        if is_significant_change:
            if is_ssim_trigger and ssim_diff > (self.ssim_threshold * 1.5):
                change_level = "Level 1 (Large slide change)"
            elif is_hash_trigger:
                change_level = "Level 2 (Medium content addition)"
            else:
                change_level = "Level 3 (Small educational addition)"

        final_score = max(ssim_diff, hash_diff)

        details = {
            "ssim_diff": ssim_diff,
            "hash_diff": hash_diff,
            "pixel_change_ratio": pixel_change_ratio,
            "edge_change_ratio": edge_change_ratio,
            "max_local_edge_change": max_local_edge_change,
            "is_ssim_trigger": is_ssim_trigger,
            "is_hash_trigger": is_hash_trigger,
            "is_local_trigger": is_local_trigger,
            "change_level": change_level,
            "pixel_diff_mask": masked_pixel_diff
        }

        return is_significant_change, final_score, details
