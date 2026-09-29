"""
DOCX File Loader supporting Word documents via python-docx or zipfile XML parser fallback.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import zipfile
import xml.etree.ElementTree as ET
import re
import logging
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector

logger = logging.getLogger("ingestion.docx")


class DocxLoader(BaseLoader):
    """
    Loader for DOCX files.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stype = FileDetector.detect_source_type(path, user_source_type=source_type)
        source_id = f"docx_{FileDetector.calculate_hash(path)[:12]}"
        extracted_text = ""

        # Attempt python-docx
        try:
            import docx
            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            extracted_text = "\n".join(paragraphs)
        except Exception:
            # Native zipfile/XML fallback for .docx
            try:
                with zipfile.ZipFile(path) as z:
                    xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    texts = []
                    for elem in tree.iter():
                        if elem.tag.endswith("}t") and elem.text:
                            texts.append(elem.text)
                    extracted_text = "".join(texts)
            except Exception as e:
                logger.warning(f"[DOCX] XML parsing note for {path}: {str(e)}")

        if not extracted_text:
            extracted_text = f"[DOCX Document Content from {path.name}]"

        deal_id = self._extract_field(extracted_text, r"(?:deal_id|deal):\s*([A-Za-z0-9_\-]+)")
        customer_id = self._extract_field(extracted_text, r"(?:customer_id|client_id|client):\s*([A-Za-z0-9_\-\s]+)")
        salesperson_id = self._extract_field(extracted_text, r"(?:salesperson_id|rep):\s*([A-Za-z0-9_\-\s]+)")

        metadata = {
            "char_length": len(extracted_text),
            "line_count": len(extracted_text.splitlines()),
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_format="docx",
            raw_content=extracted_text,
            metadata=metadata,
            deal_id=deal_id,
            customer_id=customer_id,
            salesperson_id=salesperson_id,
            source_location=str(path.resolve()),
            extraction_method="docx_text",
            extraction_confidence=1.0,
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
            file_format="docx",
            raw_content=text_content,
            metadata=meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="docx_text",
            extraction_confidence=1.0,
        )
        return [record]

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
