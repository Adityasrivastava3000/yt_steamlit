import cv2
import logging
import numpy as np
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

class ChangeClassifier:
    """Classifies the type of educational change detected between two frames."""

    def __init__(self):
        pass

    def classify(
        self,
        current_frame: np.ndarray,
        prev_accepted_frame: np.ndarray,
        details: Dict[str, Any]
    ) -> str:
        """
        Classify the type of educational change.

        Supported change types:
        - new_slide
        - graph_update
        - diagram_update
        - code_update
        - equation_update
        - whiteboard_update
        - bullet_update
        - chart_update
        - content_reveal

        :param current_frame: The current BGR frame.
        :param prev_accepted_frame: The previous accepted BGR frame.
        :param details: Dictionary of metrics from the SlideDetector.
        :return: String representation of the change type.
        """
        ssim_diff = details.get("ssim_diff", 0.0)
        hash_diff = details.get("hash_diff", 0.0)
        pixel_change_ratio = details.get("pixel_change_ratio", 0.0)
        pixel_diff_mask = details.get("pixel_diff_mask")

        # 1. New Slide Detection
        # If the SSIM difference is very high and a large portion of the screen changes, it's a new slide.
        if ssim_diff > 0.25 and hash_diff > 0.20 and pixel_change_ratio > 0.20:
            return "new_slide"

        if pixel_diff_mask is None or np.sum(pixel_diff_mask) == 0:
            return "content_reveal"

        # 2. Extract Bounding Box of Changes
        # Find contours in the difference mask to localize where the change happened
        contours, _ = cv2.findContours(
            pixel_diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter out extremely small contours (noise)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if not valid_contours:
            return "content_reveal"

        # Find the overall bounding rect enclosing all changes, or analyze the largest change
        # Sorting by area to get the primary change area
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        largest_contour = valid_contours[0]
        
        # Bounding box of the primary change
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ensure bounding box is within frame boundaries
        h_frame, w_frame = current_frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        if w < 10 or h < 10:
            return "content_reveal"

        # Extract regions from current frame
        roi_bgr = current_frame[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Analyze the ROI features
        return self._classify_roi(roi_bgr, roi_gray, w, h)

    def _classify_roi(
        self,
        roi_bgr: np.ndarray,
        roi_gray: np.ndarray,
        w: int,
        h: int
    ) -> str:
        """
        Classifies the ROI based on edge density, color uniformness, text line patterns,
        and geometric shapes.
        """
        # 1. Whiteboard check (Uniform background)
        # Whiteboards have low color variance in the background.
        # We compute the standard deviation of grayscale pixels.
        gray_std = np.std(roi_gray)
        
        # Also check dominant color of the ROI to see if it's black/white/green
        # Calculate standard deviation of BGR channels to check color saturation
        mean_bgr = np.mean(roi_bgr, axis=(0, 1))
        # If BGR channels are highly similar, it's grayscale (white/black board)
        bgr_std = np.std(mean_bgr)
        
        is_monochrome_bg = bgr_std < 15.0
        is_uniform_bg = gray_std < 45.0  # Whiteboard strokes add some variance, but background is uniform

        # If it is uniform, check if it fits whiteboard (writing)
        # Whiteboard changes usually have high contrast strokes on solid background
        if is_uniform_bg and is_monochrome_bg:
            # Check edge density: writing should have thin contours (strokes)
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_ratio = np.sum(edges == 255) / (w * h)
            if 0.005 < edge_ratio < 0.12:
                return "whiteboard_update"

        # 2. Text vs Image/Graph analysis using horizontal projections
        # Apply binary thresholding to isolate characters/strokes
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate horizontal projection profile (sum along rows)
        row_sums = np.sum(binary, axis=1)
        
        # Find horizontal lines of text: count how many rows are empty/non-empty
        # Text has distinct lines with spacing (valleys)
        zero_rows = np.sum(row_sums < (np.max(row_sums) * 0.05))
        non_zero_rows = h - zero_rows
        
        # Peak-valley pattern analysis for text lines
        peaks = 0
        in_peak = False
        peak_widths = []
        current_width = 0
        for val in row_sums:
            is_active = val > (np.max(row_sums) * 0.08)
            if is_active and not in_peak:
                in_peak = True
                peaks += 1
                current_width = 1
            elif is_active and in_peak:
                current_width += 1
            elif not is_active and in_peak:
                in_peak = False
                peak_widths.append(current_width)

        # Average peak width (height of a text row in pixels)
        avg_peak_height = np.mean(peak_widths) if peak_widths else 0

        # If we have distinct horizontal strips (lines of text)
        is_text_layout = peaks >= 1 and zero_rows / h > 0.25

        if is_text_layout:
            # 3. Code vs Bullet vs Equation
            # Code blocks have multiple lines with indentation differences
            # Let's inspect the starting pixel of each row (left indentation)
            starts = []
            for r in range(h):
                if row_sums[r] > (np.max(row_sums) * 0.1):
                    # Find first column with a pixel
                    cols_active = np.where(binary[r] > 0)[0]
                    if len(cols_active) > 0:
                        starts.append(cols_active[0])
            
            # Code: multiple lines with varying indentations (standard deviation of indentations is high, but grouped)
            if len(starts) >= 4:
                indent_std = np.std(starts)
                # Code blocks have monospaced look and structured indents (usually multiples of some indent size)
                # Let's check the number of lines
                if indent_std > 8.0 and w / h > 1.2:
                    return "code_update"

            # Equation: single or double lines, often centered, contains isolated thin symbols
            # Often math symbols have small vertical/horizontal spans or are isolated
            if peaks <= 2:
                # Math equations have high aspect ratio characters and symbols like =, +, -, fraction lines
                # Let's check if the width is relatively small compared to screen, or centered
                # Let's also check if it's very sparse
                density = np.sum(binary == 255) / (w * h)
                if density < 0.15:
                    # Check for horizontal fraction-like lines
                    # A horizontal line of pixels that spans at least 40% of the box width
                    col_sums = np.sum(binary, axis=0)
                    has_long_horizontal_line = False
                    for r in range(h):
                        # count continuous active pixels in row
                        row_pixels = binary[r]
                        runs = np.diff(np.where(np.concatenate(([0], row_pixels > 0, [0])))[0])
                        if len(runs) > 0 and np.max(runs) > (w * 0.4):
                            has_long_horizontal_line = True
                            break
                    if has_long_horizontal_line:
                        return "equation_update"
            
            # Otherwise, standard text addition
            return "bullet_update"

        # 4. Graph / Chart / Diagram detection
        # If it is not text layout, let's look at structural lines and shapes.
        # Use Hough lines to find grid/axis structures (graphs/charts)
        edges = cv2.Canny(roi_gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=max(15, int(min(w, h) * 0.2)), maxLineGap=5
        )
        
        num_lines = len(lines) if lines is not None else 0

        # Graphs/charts have strong vertical and horizontal lines (axes, grid, bars)
        horizontal_lines = 0
        vertical_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx > dy * 3:  # Horizontal
                    horizontal_lines += 1
                elif dy > dx * 3:  # Vertical
                    vertical_lines += 1

        # High line counts, especially horizontal/vertical, indicate graphs or charts
        if num_lines > 5:
            if horizontal_lines >= 2 and vertical_lines >= 2:
                return "graph_update"
            else:
                return "chart_update"

        # 5. Diagram check (Connected components, contours with shapes, arrows)
        # Diagrams have shapes like circles/rectangles connected by lines
        contours_roi, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        geometric_shapes = 0
        for c in contours_roi:
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            # Shapes like triangles (3), rectangles (4), pentagons/circles (>4)
            if 3 <= len(approx) <= 8 and cv2.contourArea(c) > 100:
                geometric_shapes += 1

        if geometric_shapes >= 2:
            return "diagram_update"

        # 6. Fallback
        # If it has some edges but doesn't map to text or line drawings cleanly
        return "content_reveal"
