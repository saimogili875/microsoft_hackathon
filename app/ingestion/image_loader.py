"""
Image Loader for handwritten notes, whiteboard photos, and scanned documents (JPG, JPEG, PNG, BMP, TIFF).
Converts images to structured RawRecords using a pluggable BaseOCRAdapter.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import logging
from app.ingestion.base import BaseLoader, BaseOCRAdapter
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector
from app.ingestion.ocr import get_ocr_adapter

logger = logging.getLogger("ingestion.image")


class ImageLoader(BaseLoader):
    """
    Loader for handwritten notes and scanned document images.
    Delegates optical character recognition to a pluggable BaseOCRAdapter.
    """

    def __init__(self, ocr_adapter: Optional[BaseOCRAdapter] = None):
        self.ocr_adapter = ocr_adapter

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        adapter = self.ocr_adapter or get_ocr_adapter()
        stype = "handwritten_note" if source_type == "auto" else source_type
        source_id = f"img_{FileDetector.calculate_hash(path)[:12]}"
        original_format = path.suffix.lstrip(".").lower() or "jpg"

        # Execute OCR extraction via pluggable adapter
        ocr_res = adapter.extract_text(path)

        metadata = {
            "original_format": original_format,
            "ocr_engine": ocr_res.metadata.get("ocr_engine", "MockOCRAdapter"),
            "ocr_confidence": ocr_res.confidence,
            "page_count": ocr_res.page_count,
            "flagged_for_review": ocr_res.is_low_confidence,
            "status": "success" if not ocr_res.error else "failed",
            "ocr_error": ocr_res.error,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_format=original_format,
            raw_content=ocr_res.extracted_text or f"[Handwritten Note Image: {path.name}]",
            metadata=metadata,
            source_location=str(path.resolve()),
            extraction_method="ocr",
            extraction_confidence=ocr_res.confidence,
        )

        return [record]

    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        meta = metadata or {}
        adapter = self.ocr_adapter or get_ocr_adapter()

        if isinstance(content, (bytes, bytearray)):
            ocr_res = adapter.extract_text(content)
            extracted_text = ocr_res.extracted_text
            confidence = ocr_res.confidence
            is_low = ocr_res.is_low_confidence
        else:
            extracted_text = str(content)
            confidence = meta.get("ocr_confidence", 0.90)
            is_low = confidence < 0.60

        full_meta = {
            "ocr_confidence": confidence,
            "flagged_for_review": is_low,
            **meta,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "handwritten_note",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format=meta.get("original_format", "jpg"),
            raw_content=extracted_text,
            metadata=full_meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="ocr",
            extraction_confidence=confidence,
        )

        return [record]
