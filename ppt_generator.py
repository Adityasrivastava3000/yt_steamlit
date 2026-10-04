import os
import sys
import json
import logging
import re
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Configure Logger for PPT Generator
logger = logging.getLogger("ppt_generator")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "ppt_generator.log"), mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Modern, professional color palette
BG_DARK = RGBColor(30, 41, 59)        # Slate Blue
TEXT_LIGHT = RGBColor(248, 250, 252)  # Off-White
BG_LIGHT = RGBColor(248, 250, 252)   # Off-White
TEXT_DARK = RGBColor(15, 23, 42)      # Dark Slate (almost black)
TEXT_MUTED = RGBColor(100, 116, 139)  # Muted Gray
ACCENT_GREEN = RGBColor(5, 150, 105)  # Emerald Green (for MCQ answer highlights)
CARD_BG = RGBColor(241, 245, 249)     # Light Gray for cards/options
CARD_BORDER = RGBColor(226, 232, 240)


def get_blank_layout(prs):
    """Finds a completely blank slide layout or defaults to index 6."""
    for layout in prs.slide_layouts:
        if len(layout.shapes) == 0:
            return layout
    return prs.slide_layouts[6]


def add_dark_slide(prs, title, subtitle=None, chapter_num=None):
    """Helper to add dark divider / header / title slides."""
    blank_layout = get_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_DARK
    
    # Modern decorative accent bar
    accent_bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(1.0), Inches(2.2), Inches(11.333), Inches(0.08)
    )
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = ACCENT_GREEN
    accent_bar.line.color.rgb = ACCENT_GREEN
    
    # Title Top offset depending on chapter info presence
    title_top = Inches(2.5)
    title_height = Inches(1.5)
    if chapter_num:
        num_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(0.8))
        tf_num = num_box.text_frame
        tf_num.word_wrap = True
        p_num = tf_num.paragraphs[0]
        p_num.text = chapter_num.upper()
        p_num.font.name = "Georgia"
        p_num.font.size = Pt(20)
        p_num.font.bold = True
        p_num.font.color.rgb = ACCENT_GREEN
        title_top = Inches(2.5)
    
    # Main Title
    title_box = slide.shapes.add_textbox(Inches(1.0), title_top, Inches(11.333), title_height)
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Georgia"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT
    
    # Subtitle / Summary
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(1.0), title_top + title_height + Inches(0.2), Inches(11.333), Inches(2.0))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(18)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.line_spacing = 1.2
        
    return slide


def add_toc_slide(prs, toc_entries):
    """Helper to add Table of Contents slide with auto columns."""
    blank_layout = get_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_LIGHT
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.333), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Table of Contents"
    p.font.name = "Georgia"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    # Divider line
    sep_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.4), Inches(11.333), Inches(0.02))
    sep_line.fill.solid()
    sep_line.fill.fore_color.rgb = CARD_BORDER
    sep_line.line.color.rgb = CARD_BORDER

    max_items_per_column = 9
    num_entries = len(toc_entries)
    
    if num_entries <= max_items_per_column:
        # Single column
        col_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(5.0))
        tf_col = col_box.text_frame
        tf_col.word_wrap = True
        for idx, entry in enumerate(toc_entries):
            p_entry = tf_col.add_paragraph() if idx > 0 else tf_col.paragraphs[0]
            p_entry.text = f"{entry.get('timestamp', '00:00')}  —  {entry.get('heading', '')}"
            p_entry.font.name = "Calibri"
            p_entry.font.size = Pt(16)
            p_entry.font.color.rgb = TEXT_DARK
            p_entry.space_after = Pt(10)
    else:
        # Two columns side by side
        mid = (num_entries + 1) // 2
        col1_entries = toc_entries[:mid]
        col2_entries = toc_entries[mid:]
        
        # Column 1
        col1_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.4), Inches(5.0))
        tf_col1 = col1_box.text_frame
        tf_col1.word_wrap = True
        for idx, entry in enumerate(col1_entries):
            p_entry = tf_col1.add_paragraph() if idx > 0 else tf_col1.paragraphs[0]
            p_entry.text = f"{entry.get('timestamp', '00:00')}  —  {entry.get('heading', '')}"
            p_entry.font.name = "Calibri"
            p_entry.font.size = Pt(14)
            p_entry.font.color.rgb = TEXT_DARK
            p_entry.space_after = Pt(8)
            
        # Column 2
        col2_box = slide.shapes.add_textbox(Inches(6.9), Inches(1.8), Inches(5.4), Inches(5.0))
        tf_col2 = col2_box.text_frame
        tf_col2.word_wrap = True
        for idx, entry in enumerate(col2_entries):
            p_entry = tf_col2.add_paragraph() if idx > 0 else tf_col2.paragraphs[0]
            p_entry.text = f"{entry.get('timestamp', '00:00')}  —  {entry.get('heading', '')}"
            p_entry.font.name = "Calibri"
            p_entry.font.size = Pt(14)
            p_entry.font.color.rgb = TEXT_DARK
            p_entry.space_after = Pt(8)
            
    return slide


def add_original_slide(prs, img_path, slide_id, timestamp, ocr_text):
    """Loads, scales, centers original slide image, and adds OCR as speaker notes."""
    blank_layout = get_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_LIGHT
    
    # Header bar
    header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.4), Inches(11.333), Inches(0.6))
    tf = header_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = f"Lecture Slide {slide_id}   •   Timestamp: {timestamp}"
    p.font.name = "Georgia"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    # Separator Line
    sep_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.0), Inches(11.333), Inches(0.01))
    sep_line.fill.solid()
    sep_line.fill.fore_color.rgb = CARD_BORDER
    sep_line.line.color.rgb = CARD_BORDER

    # Center and scale image preserving aspect ratio
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            if img_h is None or img_h <= 0:
                img_h = 1
            r = img_w / img_h
            
            # Constrain image within: width=11.333, height=5.6
            max_w = 11.333
            max_h = 5.6
            
            max_h_val = max_h if max_h is not None and max_h > 0 else 1
            if max_w / max_h_val > r:
                display_h = max_h
                display_w = display_h * r
            else:
                display_w = max_w
                r_val = r if r is not None and r > 0 else 1
                display_h = display_w / r_val
                
            left = 1.0 + (max_w - display_w) / 2
            top = 1.2 + (max_h - display_h) / 2
            
            slide.shapes.add_picture(img_path, Inches(left), Inches(top), width=Inches(display_w), height=Inches(display_h))
        except Exception as e:
            logger.error(f"Error loading slide image {img_path}: {e}")
            err_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.0))
            err_box.text_frame.text = f"[Image failed to load: {img_path}]"
    else:
        logger.warning(f"Image file not found: {img_path}")
        err_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.0))
        err_box.text_frame.text = f"[Slide image not found: {img_path}]"

    # Add OCR text as speaker notes
    if ocr_text:
        try:
            slide.notes_slide.notes_text_frame.text = ocr_text
        except Exception as e:
            logger.error(f"Failed to add speaker notes for slide {slide_id}: {e}")
            
    return slide


def add_concepts_slide(prs, concepts):
    """Helper to add Key Concepts slide."""
    blank_layout = get_blank_layout(prs)
    slide = prs.slides.add_slide(blank_layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG_LIGHT
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.333), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Lecture Concepts"
    p.font.name = "Georgia"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    # Divider
    sep_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.4), Inches(11.333), Inches(0.02))
    sep_line.fill.solid()
    sep_line.fill.fore_color.rgb = CARD_BORDER
    sep_line.line.color.rgb = CARD_BORDER

    # Concepts Bullet List
    list_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(5.0))
    tf_list = list_box.text_frame
    tf_list.word_wrap = True
    
    for idx, concept in enumerate(concepts):
        p_item = tf_list.add_paragraph() if idx > 0 else tf_list.paragraphs[0]
        p_item.text = f"•   {concept}"
        p_item.font.name = "Calibri"
        p_item.font.size = Pt(20)
        p_item.font.color.rgb = TEXT_DARK
        p_item.space_after = Pt(14)
        
    return slide


def add_definitions_slides(prs, definitions):
    """Helper to add Key Definitions formatted as card callouts."""
    chunk_size = 3
    chunks = [definitions[i:i + chunk_size] for i in range(0, len(definitions), chunk_size)]
    
    for page_idx, chunk in enumerate(chunks):
        blank_layout = get_blank_layout(prs)
        slide = prs.slides.add_slide(blank_layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG_LIGHT
        
        # Title
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.333), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        page_suffix = f" ({page_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        p.text = f"Key Definitions{page_suffix}"
        p.font.name = "Georgia"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        
        # Divider Line
        sep_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.4), Inches(11.333), Inches(0.02))
        sep_line.fill.solid()
        sep_line.fill.fore_color.rgb = CARD_BORDER
        sep_line.line.color.rgb = CARD_BORDER

        top_offset = 1.8
        for idx, item in enumerate(chunk):
            term = item.get("term", "")
            definition = item.get("definition", "")
            
            # Card shape background
            card = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(1.0), Inches(top_offset), Inches(11.333), Inches(1.5)
            )
            card.fill.solid()
            card.fill.fore_color.rgb = CARD_BG
            card.line.color.rgb = CARD_BORDER
            
            # Card Text Frame
            content_box = slide.shapes.add_textbox(
                Inches(1.2), Inches(top_offset + 0.1), Inches(10.933), Inches(1.3)
            )
            tf_content = content_box.text_frame
            tf_content.word_wrap = True
            
            # Term Heading
            p_term = tf_content.paragraphs[0]
            p_term.text = term
            p_term.font.name = "Georgia"
            p_term.font.size = Pt(18)
            p_term.font.bold = True
            p_term.font.color.rgb = ACCENT_GREEN
            p_term.space_after = Pt(4)
            
            # Definition body text
            p_def = tf_content.add_paragraph()
            p_def.text = definition
            p_def.font.name = "Calibri"
            p_def.font.size = Pt(14)
            p_def.font.color.rgb = TEXT_DARK
            p_def.line_spacing = 1.15
            
            top_offset += 1.7


def add_formulas_slides(prs, formulas):
    """Helper to add Formulas formatted as code cards."""
    chunk_size = 2
    chunks = [formulas[i:i + chunk_size] for i in range(0, len(formulas), chunk_size)]
    
    for page_idx, chunk in enumerate(chunks):
        blank_layout = get_blank_layout(prs)
        slide = prs.slides.add_slide(blank_layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG_LIGHT
        
        # Title
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.5), Inches(11.333), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        page_suffix = f" ({page_idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
        p.text = f"Mathematical Formulas{page_suffix}"
        p.font.name = "Georgia"
        p.font.size = Pt(28)
        p.font.bold = True
        p.font.color.rgb = TEXT_DARK
        
        # Divider Line
        sep_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.4), Inches(11.333), Inches(0.02))
        sep_line.fill.solid()
        sep_line.fill.fore_color.rgb = CARD_BORDER
        sep_line.line.color.rgb = CARD_BORDER

        top_offset = 2.0
        for idx, item in enumerate(chunk):
            formula = item.get("formula", "")
            description = item.get("description", "")
            
            # Card background
            card = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(1.0), Inches(top_offset), Inches(11.333), Inches(2.2)
            )
            card.fill.solid()
            card.fill.fore_color.rgb = CARD_BG
            card.line.color.rgb = CARD_BORDER
            
            # Card Text Frame
            content_box = slide.shapes.add_textbox(
                Inches(1.2), Inches(top_offset + 0.1), Inches(10.933), Inches(2.0)
            )
            tf_content = content_box.text_frame
            tf_content.word_wrap = True
            
            # Formula (formatted as code representation)
            p_form = tf_content.paragraphs[0]
            p_form.text = formula
            p_form.font.name = "Courier New"
            p_form.font.size = Pt(20)
            p_form.font.bold = True
            p_form.font.color.rgb = ACCENT_GREEN
            p_form.space_after = Pt(10)
            
            # Formula description
            p_desc = tf_content.add_paragraph()
            p_desc.text = f"Description: {description}"
            p_desc.font.name = "Calibri"
            p_desc.font.size = Pt(14)
            p_desc.font.color.rgb = TEXT_DARK
            p_desc.line_spacing = 1.15
            
            top_offset += 2.4


def add_mcq_slides(prs, mcq, mcq_index, total_mcqs):
    """Helper to add MCQ slide pairs (Question Slide + Answer-Highlighted Slide)."""
    question = mcq.get("question", "")
    options = mcq.get("options", [])
    correct_answer = mcq.get("answer", "")
    
    for is_answer_slide in [False, True]:
        blank_layout = get_blank_layout(prs)
        slide = prs.slides.add_slide(blank_layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG_LIGHT
        
        # Header Status
        header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.4), Inches(11.333), Inches(0.5))
        tf = header_box.text_frame
        tf.word_wrap = True
        p_hdr = tf.paragraphs[0]
        p_hdr.text = f"MCQ QUIZ  •  Question {mcq_index} of {total_mcqs}" + (" (Answer)" if is_answer_slide else "")
        p_hdr.font.name = "Georgia"
        p_hdr.font.size = Pt(14)
        p_hdr.font.bold = True
        p_hdr.font.color.rgb = ACCENT_GREEN if is_answer_slide else TEXT_MUTED
        
        # Question Title
        q_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.9), Inches(11.333), Inches(1.2))
        tf_q = q_box.text_frame
        tf_q.word_wrap = True
        p_q = tf_q.paragraphs[0]
        p_q.text = question
        p_q.font.name = "Georgia"
        p_q.font.size = Pt(20)
        p_q.font.bold = True
        p_q.font.color.rgb = TEXT_DARK
        p_q.line_spacing = 1.15
        
        # Display 4 options in vertical rounded cards
        top_start = 2.3
        for opt_idx, opt_text in enumerate(options):
            is_correct = (opt_text.strip().lower() == correct_answer.strip().lower())
            
            if is_answer_slide and is_correct:
                fill_color = ACCENT_GREEN
                border_color = ACCENT_GREEN
                font_color = TEXT_LIGHT
                bold_flag = True
            else:
                fill_color = CARD_BG
                border_color = CARD_BORDER
                font_color = TEXT_DARK
                bold_flag = False
                
            card = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(1.0), Inches(top_start), Inches(11.333), Inches(0.9)
            )
            card.fill.solid()
            card.fill.fore_color.rgb = fill_color
            card.line.color.rgb = border_color
            
            opt_box = slide.shapes.add_textbox(
                Inches(1.2), Inches(top_start + 0.05), Inches(10.933), Inches(0.8)
            )
            tf_opt = opt_box.text_frame
            tf_opt.word_wrap = True
            p_opt = tf_opt.paragraphs[0]
            
            label = chr(65 + opt_idx)  # 'A', 'B', 'C', 'D'
            p_opt.text = f"{label}.   {opt_text}"
            p_opt.font.name = "Calibri"
            p_opt.font.size = Pt(16)
            p_opt.font.bold = bold_flag
            p_opt.font.color.rgb = font_color
            
            top_start += 1.15


def add_flashcard_slides(prs, card_data, card_index, total_cards):
    """Helper to add Flashcard slide pairs (Front Side Prompt + Back Side Explanation)."""
    front_text = card_data.get("front", "")
    back_text = card_data.get("back", "")
    
    for is_back in [False, True]:
        blank_layout = get_blank_layout(prs)
        slide = prs.slides.add_slide(blank_layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BG_LIGHT
        
        # Header Status
        header_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.4), Inches(11.333), Inches(0.5))
        tf = header_box.text_frame
        tf.word_wrap = True
        p_hdr = tf.paragraphs[0]
        p_hdr.text = f"FLASHCARD  •  Card {card_index} of {total_cards}" + (" (Back)" if is_back else " (Front)")
        p_hdr.font.name = "Georgia"
        p_hdr.font.size = Pt(14)
        p_hdr.font.bold = True
        p_hdr.font.color.rgb = ACCENT_GREEN if is_back else TEXT_MUTED
        
        # Central card shape container
        card_w = 8.5
        card_h = 4.2
        left = (13.333 - card_w) / 2
        top = (7.5 - card_h) / 2 + 0.3
        
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left), Inches(top), Inches(card_w), Inches(card_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        
        content_box = slide.shapes.add_textbox(
            Inches(left + 0.4), Inches(top + 0.3), Inches(card_w - 0.8), Inches(card_h - 0.6)
        )
        tf_c = content_box.text_frame
        tf_c.word_wrap = True
        
        if not is_back:
            # Front side - large centered prompt term
            p_front = tf_c.paragraphs[0]
            p_front.text = front_text
            p_front.alignment = PP_ALIGN.CENTER
            p_front.font.name = "Georgia"
            p_front.font.size = Pt(28)
            p_front.font.bold = True
            p_front.font.color.rgb = TEXT_DARK
            p_front.space_before = Pt(80)
        else:
            # Back side - concept term at top + description text below
            p_concept = tf_c.paragraphs[0]
            p_concept.text = front_text
            p_concept.alignment = PP_ALIGN.CENTER
            p_concept.font.name = "Georgia"
            p_concept.font.size = Pt(22)
            p_concept.font.bold = True
            p_concept.font.color.rgb = ACCENT_GREEN
            p_concept.space_after = Pt(20)
            
            p_explain = tf_c.add_paragraph()
            p_explain.text = back_text
            p_explain.alignment = PP_ALIGN.CENTER
            p_explain.font.name = "Calibri"
            p_explain.font.size = Pt(16)
            p_explain.font.color.rgb = TEXT_DARK
            p_explain.line_spacing = 1.25
            p_explain.space_before = Pt(10)


class PPTGenerator:
    """Service to compile lecture presentation decks from structured outputs and slide screenshots."""
    
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.slides_dir = os.path.join(output_dir, "slides")
        self.ppt_path = os.path.join(output_dir, "lecture_presentation.pptx")
        
    def generate(self) -> bool:
        logger.info("Starting professional PPT creation pipeline...")
        
        # Load necessary JSON input resources
        try:
            # 1. Video Metadata (optional, for title)
            video_metadata_path = os.path.join(self.output_dir, "video_metadata.json")
            video_title = "Educational Lecture Presentation"
            if os.path.exists(video_metadata_path):
                with open(video_metadata_path, "r", encoding="utf-8") as f:
                    v_meta = json.load(f)
                    video_title = v_meta.get("title", video_title)

            # 2. OCR Results (required, for slides and notes)
            ocr_results_path = os.path.join(self.output_dir, "ocr_results.json")
            if not os.path.exists(ocr_results_path):
                logger.error(f"Missing required file: {ocr_results_path}")
                return False
            with open(ocr_results_path, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load JSON inputs: {e}", exc_info=True)
            return False
            
        # Initialize Presentation object
        prs = Presentation()
        # Enforce Widescreen 16:9 layout coordinates
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        # 1. TITLE SLIDE
        add_dark_slide(prs, title=video_title, subtitle="Compiled lecture slides and notes.")
        logger.info("Title Slide successfully added.")
        
        # Sort OCR results by slide ID to guarantee chronological sequence
        ocr_results_sorted = sorted(ocr_results, key=lambda x: x.get("slide_id", 0))
        
        # 2. ORIGINAL SLIDES
        for slide_entry in ocr_results_sorted:
            s_id = slide_entry.get("slide_id", 0)
            timestamp = slide_entry.get("timestamp", "")
            img_filename = slide_entry.get("image", f"slide_{s_id:04d}.jpg")
            img_path = os.path.join(self.slides_dir, img_filename)
            ocr_text = slide_entry.get("text", "")
            
            add_original_slide(prs, img_path, s_id, timestamp, ocr_text)
            logger.info(f"Inserted Slide {s_id} at {timestamp}.")
            
        # 3. FINAL RECAP SLIDE
        add_dark_slide(prs, title="Presentation Complete", subtitle="This concludes the lecture slides deck. Review all slides and speaker notes.")
        logger.info("Final Recap Slide added.")
        
        # Save PowerPoint file
        try:
            prs.save(self.ppt_path)
            logger.info(f"Successfully compiled PPT presentation at: {self.ppt_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save generated PPT presentation: {e}", exc_info=True)
            return False
