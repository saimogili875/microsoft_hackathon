"""
Markdown File Loader for ingestion of MD and documentation files.
Parses markdown frontmatter, headings, and raw text.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import re
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector


class MarkdownLoader(BaseLoader):
    """
    Loader for Markdown files with YAML frontmatter parsing.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        source_id = f"md_{FileDetector.calculate_hash(path)[:12]}"
        stype = FileDetector.detect_source_type(path, user_source_type=source_type)

        frontmatter, main_text = self._split_frontmatter(content)

        deal_id = frontmatter.get("deal_id") or self._extract_field(main_text, r"(?:deal_id|deal):\s*([A-Za-z0-9_\-]+)")
        customer_id = frontmatter.get("customer_id") or frontmatter.get("client_id") or self._extract_field(main_text, r"(?:customer_id|client_id|client):\s*([A-Za-z0-9_\-\s]+)")
        salesperson_id = frontmatter.get("salesperson_id") or self._extract_field(main_text, r"(?:salesperson_id|rep):\s*([A-Za-z0-9_\-\s]+)")

        metadata = {
            "frontmatter": frontmatter,
            "heading_count": len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE)),
            "char_length": len(content),
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=frontmatter.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format="md",
            raw_content=main_text,
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
        main_text = str(content)
        meta = metadata or {}
        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "meeting_note",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format="md",
            raw_content=main_text,
            metadata=meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="direct",
            extraction_confidence=1.0,
        )
        return [record]

    def _split_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        frontmatter = {}
        main_text = text
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if fm_match:
            fm_str = fm_match.group(1)
            main_text = fm_match.group(2)
            for line in fm_str.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    frontmatter[k.strip()] = v.strip().strip("\"'")
        return frontmatter, main_text

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
