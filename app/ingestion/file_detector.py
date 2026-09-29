"""
File Detector providing automatic format identification, source-type classification,
and content hashing for duplicate detection.
"""

from typing import Union, Optional, Tuple, Dict, Any
from pathlib import Path
import hashlib
import json
import logging

logger = logging.getLogger("ingestion.detector")

# File format extensions mapping
EXTENSION_FORMAT_MAP = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".txt": "txt",
    ".text": "txt",
    ".log": "txt",
    ".md": "md",
    ".markdown": "md",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".eml": "eml",
    ".msg": "eml",
    ".wav": "wav",
    ".mp3": "mp3",
    ".m4a": "m4a",
    ".mp4": "mp4",
    ".flac": "wav",
    ".ogg": "wav",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".bmp": "bmp",
    ".tiff": "tiff",
}

# Category to source_type heuristic mapping
FILENAME_SOURCE_MAP = {
    "deal": "deal_history",
    "crm": "crm_record",
    "customer": "crm_record",
    "account": "crm_record",
    "contact": "crm_record",
    "conversation": "sales_conversation",
    "chat": "sales_conversation",
    "transcript": "call_transcript",
    "call": "call_transcript",
    "audio": "audio_call",
    "recording": "audio_call",
    "meeting": "meeting_note",
    "note": "sales_notes",
    "handwritten": "handwritten_note",
    "scan": "handwritten_note",
    "email": "email",
    "market": "market_info",
    "competitor": "market_info",
    "feedback": "feedback",
}


class FileDetector:
    """
    Utility service to classify file formats, detect sales source types, and calculate SHA-256 hashes.
    """

    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext in EXTENSION_FORMAT_MAP:
            return EXTENSION_FORMAT_MAP[ext]

        # Magic byte detection fallback if extension is missing/ambiguous
        if path.exists():
            try:
                with open(path, "rb") as f:
                    header = f.read(16)
                    if header.startswith(b"%PDF"):
                        return "pdf"
                    if header.startswith(b"\x50\x4b\x03\x04"):  # Zip/Docx
                        return "docx"
                    if header.startswith(b"RIFF") and b"WAVE" in header:
                        return "wav"
                    if header.startswith(b"\xff\xd8\xff"):
                        return "jpeg"
                    if header.startswith(b"\x89PNG\r\n\x1a\n"):
                        return "png"
            except Exception as e:
                logger.debug(f"Magic byte check error: {str(e)}")

        return "unknown"

    @staticmethod
    def detect_source_type(file_path: Union[str, Path], user_source_type: str = "auto") -> str:
        if user_source_type and user_source_type != "auto":
            return user_source_type

        path = Path(file_path)
        name_lower = path.name.lower()

        # Check format-specific defaults first
        fmt = FileDetector.detect_format(path)
        if fmt in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            return "handwritten_note"
        if fmt in ["wav", "mp3", "m4a", "mp4"]:
            return "audio_call"
        if fmt == "eml":
            return "email"

        # Check filename heuristics
        for kw, stype in FILENAME_SOURCE_MAP.items():
            if kw in name_lower:
                return stype

        return "sales_notes"

    @staticmethod
    def calculate_hash(file_path_or_content: Union[str, Path, bytes, dict, list]) -> str:
        """
        Computes deterministic SHA-256 content hash for duplicate detection.
        """
        hasher = hashlib.sha256()
        if isinstance(file_path_or_content, (str, Path)) and Path(file_path_or_content).exists():
            with open(file_path_or_content, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
        elif isinstance(file_path_or_content, bytes):
            hasher.update(file_path_or_content)
        elif isinstance(file_path_or_content, str):
            hasher.update(file_path_or_content.encode("utf-8"))
        else:
            dumped = json.dumps(file_path_or_content, sort_keys=True, default=str)
            hasher.update(dumped.encode("utf-8"))

        return hasher.hexdigest()
