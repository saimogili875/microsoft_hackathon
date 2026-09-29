"""
PDF File Loader supporting text-based PDFs and scanned PDF OCR fallback.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import re
import logging
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector
from app.ingestion.ocr import get_ocr_adapter, BaseOCRAdapter

logger = logging.getLogger("ingestion.pdf")


class PdfLoader(BaseLoader):
    """
    Loader for PDF files.
    Extracts selectable text via pypdf / pdfplumber if available, or falls back to OCR adapter.
    """

    def __init__(self, ocr_adapter: Optional[BaseOCRAdapter] = None):
        self.ocr_adapter = ocr_adapter

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stype = FileDetector.detect_source_type(path, user_source_type=source_type)
        source_id = f"pdf_{FileDetector.calculate_hash(path)[:12]}"

        extracted_text = ""
        page_count = 1
        extraction_method = "pdf_text"
        confidence = 1.0
        is_low_confidence = False

        # Attempt native pypdf text extraction if installed
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            page_count = len(reader.pages)
            pages_text = []
            for idx, page in enumerate(reader.pages, 1):
                p_text = page.extract_text() or ""
                if p_text.strip():
                    pages_text.append(f"--- Page {idx} ---\n{p_text.strip()}")
            extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            logger.debug(f"[PDF] Native PDF library note ({str(e)}). Checking OCR fallback.")

        # If extracted text is empty or minimal (scanned PDF), use OCR adapter
        if not extracted_text or len(extracted_text.strip()) < 20:
            adapter = self.ocr_adapter or get_ocr_adapter()
            ocr_res = adapter.extract_text(path, page_count=page_count)
            extracted_text = ocr_res.extracted_text
            confidence = ocr_res.confidence
            extraction_method = "ocr"
            is_low_confidence = ocr_res.is_low_confidence
            if ocr_res.page_count:
                page_count = ocr_res.page_count

        deal_id = self._extract_field(extracted_text, r"(?:deal_id|deal):\s*([A-Za-z0-9_\-]+)")
        customer_id = self._extract_field(extracted_text, r"(?:customer_id|client_id|client):\s*([A-Za-z0-9_\-\s]+)")
        salesperson_id = self._extract_field(extracted_text, r"(?:salesperson_id|rep):\s*([A-Za-z0-9_\-\s]+)")

        metadata = {
            "page_count": page_count,
            "char_length": len(extracted_text),
            "flagged_for_review": is_low_confidence,
            "ocr_confidence": confidence if extraction_method == "ocr" else None,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_format="pdf",
            raw_content=extracted_text,
            metadata=metadata,
            deal_id=deal_id,
            customer_id=customer_id,
            salesperson_id=salesperson_id,
            source_location=str(path.resolve()),
            extraction_method=extraction_method,
            extraction_confidence=confidence,
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
        text_content = str(content)
        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "meeting_note",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format="pdf",
            raw_content=text_content,
            metadata=meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="pdf_text",
            extraction_confidence=1.0,
        )
        return [record]

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
