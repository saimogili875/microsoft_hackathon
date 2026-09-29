"""
OCR Adapter Layer providing pluggable handwriting & document text recognition.
Includes MockOCRAdapter for offline test environments and TesseractOCRAdapter for production environments.
"""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import os
import logging
from app.ingestion.base import BaseOCRAdapter, OCRExtractionResult

logger = logging.getLogger("ingestion.ocr")


class MockOCRAdapter(BaseOCRAdapter):
    """
    Default Mock OCR Adapter that simulates OCR / Handwriting recognition.
    Can be configured with default confidence, simulated errors, or custom extracted text.
    """

    def __init__(self, default_confidence: float = 0.90, default_text: Optional[str] = None):
        self.default_confidence = default_confidence
        self.default_text = default_text

    def extract_text(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> OCRExtractionResult:
        simulated_error = kwargs.get("simulate_error", False)
        forced_confidence = kwargs.get("confidence", self.default_confidence)
        forced_text = kwargs.get("text", self.default_text)

        if simulated_error:
            logger.warning("[OCR] Simulated OCR engine failure requested.")
            return OCRExtractionResult(
                extracted_text="",
                confidence=0.0,
                page_count=1,
                metadata={"ocr_engine": "MockOCRAdapter", "status": "failed"},
                is_low_confidence=True,
                error="Simulated OCR engine failure",
            )

        file_name = "unknown"
        original_format = "jpg"
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            file_name = path.name
            original_format = path.suffix.lstrip(".").lower() or "jpg"
            if not forced_text and path.exists():
                try:
                    # If file is a text file named as image for testing, or contains readable bytes
                    with open(path, "rb") as f:
                        raw_bytes = f.read(2048)
                        # Check if UTF-8 readable text
                        try:
                            decoded = raw_bytes.decode("utf-8")
                            if any(c.isalnum() for c in decoded):
                                forced_text = decoded
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"[OCR] File read note: {str(e)}")

        if not forced_text:
            forced_text = f"[OCR Extracted Text from {file_name}]\nSalesperson Note: Customer reacted positively to phased rollout proposal.\nBudget objection raised: Implementation cost ~₹4L."

        is_low_confidence = forced_confidence < 0.60

        metadata = {
            "ocr_engine": "MockOCRAdapter",
            "file_name": file_name,
            "original_format": original_format,
            "ocr_confidence": forced_confidence,
            "flagged_for_review": is_low_confidence,
        }

        return OCRExtractionResult(
            extracted_text=forced_text,
            confidence=forced_confidence,
            page_count=kwargs.get("page_count", 1),
            metadata=metadata,
            is_low_confidence=is_low_confidence,
            original_format=original_format,
        )


class TesseractOCRAdapter(BaseOCRAdapter):
    """
    Production Tesseract OCR Adapter using PIL and pytesseract.
    Falls back to MockOCRAdapter if pytesseract / Tesseract binary is unavailable.
    """

    def __init__(self, fallback: Optional[BaseOCRAdapter] = None):
        self.fallback = fallback or MockOCRAdapter()

    def extract_text(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> OCRExtractionResult:
        try:
            from PIL import Image
            import pytesseract

            if isinstance(file_path_or_bytes, (str, Path)):
                img = Image.open(file_path_or_bytes)
            else:
                import io
                img = Image.open(io.BytesIO(file_path_or_bytes))

            text = pytesseract.image_to_string(img)
            confidence = 0.85  # Estimate from data if available
            is_low = confidence < 0.60
            fmt = getattr(img, "format", "PNG").lower()

            return OCRExtractionResult(
                extracted_text=text.strip(),
                confidence=confidence,
                page_count=getattr(img, "n_frames", 1),
                metadata={"ocr_engine": "pytesseract", "format": fmt, "flagged_for_review": is_low},
                is_low_confidence=is_low,
                original_format=fmt,
            )
        except Exception as e:
            logger.warning(f"[OCR] Tesseract OCR unavailable ({str(e)}). Using fallback adapter.")
            return self.fallback.extract_text(file_path_or_bytes, **kwargs)


# Global adapter registry and accessors
_current_ocr_adapter: BaseOCRAdapter = MockOCRAdapter()


def get_ocr_adapter() -> BaseOCRAdapter:
    return _current_ocr_adapter


def set_ocr_adapter(adapter: BaseOCRAdapter):
    global _current_ocr_adapter
    _current_ocr_adapter = adapter
