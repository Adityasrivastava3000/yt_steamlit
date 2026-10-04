import os
import sys
import shutil
import logging
import subprocess
from pypdf import PdfReader

# Configure Logger for PDF Generator
logger = logging.getLogger("pdf_generator")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join("logs", "pdf_generator.log"), mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def find_soffice():
    """Locates LibreOffice (soffice) executable in standard locations or PATH."""
    # 1. Search PATH
    for cmd in ["soffice", "libreoffice"]:
        path = shutil.which(cmd)
        if path:
            logger.info(f"Found LibreOffice in system PATH: {path}")
            return path
            
    # 2. Check standard platform-specific directories
    if sys.platform == "darwin":  # macOS
        mac_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if os.path.exists(mac_path):
            logger.info(f"Found LibreOffice standard macOS path: {mac_path}")
            return mac_path
    elif sys.platform == "win32":  # Windows
        win_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
        ]
        for p in win_paths:
            if os.path.exists(p):
                logger.info(f"Found LibreOffice standard Windows path: {p}")
                return p
                
    logger.warning("LibreOffice (soffice) executable could not be resolved in default paths or PATH environment.")
    return None


class PDFGenerator:
    """Service to automatically convert PPTX presentations to PDF format with fallback support."""
    
    def __init__(self, output_dir="outputs", soffice_path=None):
        self.output_dir = output_dir
        self.soffice_path = soffice_path or find_soffice()
        
    def convert(self, pptx_path: str, pdf_path: str) -> bool:
        logger.info(f"Initiating PDF conversion: {pptx_path} -> {pdf_path}")
        
        # 1. Validate inputs
        if not os.path.exists(pptx_path):
            logger.error(f"Input PowerPoint presentation not found: {pptx_path}")
            return False
            
        if os.path.getsize(pptx_path) == 0:
            logger.error(f"Input PowerPoint presentation is empty (0 bytes): {pptx_path}")
            return False
            
        # Ensure parent output directories exist
        os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
        
        success = False
        method_used = ""
        
        # 2. Try LibreOffice Headless Conversion
        if self.soffice_path:
            try:
                method_used = "LibreOffice Headless"
                success = self._convert_via_libreoffice(pptx_path, pdf_path)
            except Exception as e:
                logger.warning(f"LibreOffice conversion failed: {e}. Attempting fallback options if available.")
                
        # 3. Fallback: Try Windows PowerPoint COM Automation
        if not success and sys.platform == "win32":
            try:
                method_used = "Windows PowerPoint COM Automation"
                success = self._convert_via_com(pptx_path, pdf_path)
            except Exception as e:
                logger.error(f"Windows PowerPoint COM conversion failed: {e}")
                
        if not success:
            logger.error("All PDF conversion methods failed. Please ensure LibreOffice is installed or PowerPoint is available.")
            return False
            
        # 4. Verification Check
        try:
            # Check file exists
            if not os.path.exists(pdf_path):
                logger.error(f"Verification Failed: Output PDF file not found at {pdf_path}")
                return False
                
            # Verify structure and retrieve page count using pypdf
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            
            if page_count <= 0:
                logger.error("Verification Failed: PDF file has 0 pages or is corrupted.")
                return False
                
            # Get file size in MB
            file_size_bytes = os.path.getsize(pdf_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            logger.info(f"Verification Successful: Generated via {method_used}.")
            logger.info(f"PDF Pages: {page_count} | PDF Size: {file_size_mb:.2f} MB")
            
            # Print to stdout for user CLI reporting
            print("\n" + "=" * 50)
            print(" PDF EXPORT COMPLETED")
            print("=" * 50)
            print(f"Output File:     {pdf_path}")
            print(f"Method Used:     {method_used}")
            print(f"Total Pages:     {page_count}")
            print(f"File Size:       {file_size_mb:.2f} MB")
            print("=" * 50 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification Failed: PDF reading/verification raised exception: {e}")
            return False
            
    def _convert_via_libreoffice(self, pptx_path: str, pdf_path: str) -> bool:
        abs_input = os.path.abspath(pptx_path)
        abs_output_dir = os.path.abspath(self.output_dir)
        
        cmd = [
            self.soffice_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", abs_output_dir,
            abs_input
        ]
        
        logger.info(f"Executing LibreOffice command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120  # Widescreen presentation can take a moment to convert
        )
        
        if result.returncode != 0:
            logger.error(f"LibreOffice command failed with return code {result.returncode}. stdout: {result.stdout}, stderr: {result.stderr}")
            raise RuntimeError(f"LibreOffice error: {result.stderr.strip()}")
            
        # Determine LibreOffice output location
        input_filename = os.path.basename(pptx_path)
        base_name, _ = os.path.splitext(input_filename)
        expected_output = os.path.join(self.output_dir, f"{base_name}.pdf")
        
        if os.path.exists(expected_output):
            # Align output file with requested pdf_path
            if os.path.abspath(expected_output) != os.path.abspath(pdf_path):
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                os.rename(expected_output, pdf_path)
            return True
        else:
            raise FileNotFoundError(f"LibreOffice succeeded but expected PDF file not generated at {expected_output}")

    def _convert_via_com(self, pptx_path: str, pdf_path: str) -> bool:
        logger.info("Attempting fallback conversion via Windows PowerPoint COM Automation...")
        
        try:
            import win32com.client
        except ImportError as e:
            raise ImportError("pywin32 library is not installed, COM fallback is unavailable.") from e
            
        powerpoint = None
        presentation = None
        try:
            powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            abs_input = os.path.abspath(pptx_path)
            abs_output = os.path.abspath(pdf_path)
            
            # Open the presentation invisibly
            presentation = powerpoint.Presentations.Open(abs_input, WithWindow=False)
            
            # Format 32 is ppSaveAsPDF
            presentation.SaveAs(abs_output, 32)
            logger.info("Successfully exported PDF using Windows COM.")
            return True
            
        except Exception as e:
            raise RuntimeError(f"PowerPoint COM save raised error: {e}") from e
            
        finally:
            if presentation:
                try:
                    presentation.Close()
                except Exception:
                    pass
            if powerpoint:
                try:
                    powerpoint.Quit()
                except Exception:
                    pass
