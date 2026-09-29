"""
Text, Markdown, Email, and Transcript loader implementation.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import re
from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord


class TextLoader(BaseLoader):
    """
    Loader for plain text (.txt), Markdown (.md), Emails, Transcripts, and Notes.
    """

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_ext = path.suffix.lower()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Empty text file: {path}")

        # Auto-detect source type if auto
        if source_type == "auto":
            source_type = self._infer_source_type(content, file_ext, path.name)

        source_id = path.stem

        record = RawRecord(
            source_type=source_type,
            source_id=source_id,
            file_format=file_ext.lstrip("."),
            raw_content=content,
            metadata={"file_path": str(path), "file_name": path.name},
        )
        return [record]

    def load_raw_content(
        self, content: Union[str, Dict[str, Any], List[Any]], source_type: str, source_id: str, metadata: Dict[str, Any] = None
    ) -> List[RawRecord]:
        metadata = metadata or {}
        text_str = str(content)
        record = RawRecord(
            source_type=source_type,
            source_id=source_id,
            file_format="txt",
            raw_content=text_str,
            metadata=metadata,
        )
        return [record]

    def _infer_source_type(self, content: str, ext: str, filename: str) -> str:
        fn_lower = filename.lower()
        content_lower = content.lower()

        if "interview" in fn_lower or "interview" in content_lower[:200]:
            return "salesperson_interview"
        if "call" in fn_lower or "transcript" in fn_lower or "speaker 1" in content_lower or "rep:" in content_lower:
            return "call_transcript"
        if "email" in fn_lower or "subject:" in content_lower or "from:" in content_lower:
            return "email"
        if "market" in fn_lower or "competitor" in content_lower:
            return "market_info"
        if "feedback" in fn_lower:
            return "salesperson_feedback"
        if ext == ".md":
            return "sales_notes"
        return "sales_conversation"
