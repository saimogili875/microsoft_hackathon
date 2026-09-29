"""
Text File Loader for ingestion of TXT, LOG, and unformatted notes files.
Handles multi-encoding detection and safe fallback.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import os
import re
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class TxtLoader(BaseLoader):
    """
    Loader for plain text files with robust encoding fallback.
    """

    ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text_content = ""
        used_encoding = "utf-8"
        for enc in self.ENCODINGS:
            try:
                with open(path, "r", encoding=enc) as f:
                    text_content = f.read()
                used_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        source_id = f"txt_{FileDetector.calculate_hash(path)[:12]}"
        stype = FileDetector.detect_source_type(path, user_source_type=source_type)

        # Optional metadata extraction from text header
        deal_id = self._extract_field(text_content, r"(?:deal_id|deal):\s*([A-Za-z0-9_\-]+)")
        customer_id = self._extract_field(text_content, r"(?:customer_id|client_id|client):\s*([A-Za-z0-9_\-\s]+)")
        salesperson_id = self._extract_field(text_content, r"(?:salesperson_id|rep_id|rep):\s*([A-Za-z0-9_\-\s]+)")

        metadata = {
            "encoding": used_encoding,
            "char_length": len(text_content),
            "line_count": len(text_content.splitlines()),
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_format="txt",
            raw_content=text_content,
            metadata=metadata,
            deal_id=deal_id,
            customer_id=customer_id,
            salesperson_id=salesperson_id,
            source_location=str(path.resolve()),
            extraction_method="direct",
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
        text_content = str(content)
        meta = metadata or {}
        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "sales_notes",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format="txt",
            raw_content=text_content,
            metadata=meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="direct",
            extraction_confidence=1.0,
        )
        return [record]

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
