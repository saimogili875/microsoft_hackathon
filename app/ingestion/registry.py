"""
Ingestion Registry and Manager orchestrating format detection, loader lookup,
and content deduplication across all 11 sales input formats.
"""

from typing import Dict, Type, Union, List, Any, Optional, Set
from pathlib import Path
import logging

from app.ingestion.base import BaseLoader
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector

from app.ingestion.txt_loader import TxtLoader
from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.csv_loader import CSVLoader
from app.ingestion.json_loader import JsonLoader
from app.ingestion.jsonl_loader import JsonlLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.email_loader import EmailLoader
from app.ingestion.audio_loader import AudioLoader
from app.ingestion.image_loader import ImageLoader
from app.ingestion.crm_loader import CrmLoader

logger = logging.getLogger("ingestion.registry")


class IngestionRegistry:
    """
    Central Manager for sales data ingestion.
    Routes file paths and raw objects to appropriate loaders with deduplication.
    """

    def __init__(self):
        self._format_loaders: Dict[str, BaseLoader] = {
            "txt": TxtLoader(),
            "md": MarkdownLoader(),
            "csv": CSVLoader(),
            "json": JsonLoader(),
            "jsonl": JsonlLoader(),
            "pdf": PdfLoader(),
            "docx": DocxLoader(),
            "eml": EmailLoader(),
            "wav": AudioLoader(),
            "mp3": AudioLoader(),
            "m4a": AudioLoader(),
            "mp4": AudioLoader(),
            "jpg": ImageLoader(),
            "jpeg": ImageLoader(),
            "png": ImageLoader(),
            "bmp": ImageLoader(),
            "tiff": ImageLoader(),
        }
        self._crm_loader = CrmLoader()
        self._seen_hashes: Set[str] = set()

    def get_loader(self, format_name: str) -> BaseLoader:
        fmt = format_name.lower()
        if fmt in self._format_loaders:
            return self._format_loaders[fmt]
        return self._format_loaders["txt"]

    def register_loader(self, format_name: str, loader: BaseLoader):
        """
        Allows registering custom or extended loaders.
        """
        self._format_loaders[format_name.lower()] = loader

    def ingest_file(self, file_path: Union[str, Path], source_type: str = "auto", check_duplicates: bool = True) -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check for duplicates via SHA-256 hash
        content_hash = FileDetector.calculate_hash(path)
        if check_duplicates and content_hash in self._seen_hashes:
            logger.warning(f"[INGESTION] Duplicate file skipped: {path.name} (hash: {content_hash[:10]})")
            return []

        fmt = FileDetector.detect_format(path)
        if fmt not in self._format_loaders:
            raise ValueError(f"Unsupported file format '{fmt}' for file: {path.name}")

        loader = self._format_loaders[fmt]
        records = loader.load_file(path, source_type=source_type)

        if check_duplicates and records:
            self._seen_hashes.add(content_hash)

        logger.info(f"[INGESTION] Successfully ingested {len(records)} RawRecords from {path.name} (Format: {fmt})")
        return records

    # Compatibility alias
    load = ingest_file

    def ingest_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        file_format: str = "json",
        metadata: Optional[Dict[str, Any]] = None,
        check_duplicates: bool = True,
    ) -> List[RawRecord]:
        content_hash = FileDetector.calculate_hash(content)
        if check_duplicates and content_hash in self._seen_hashes:
            logger.warning(f"[INGESTION] Duplicate content skipped for source_id: {source_id}")
            return []

        fmt = file_format.lower()
        if fmt == "dict" or source_type in ["crm_record", "deal_history"]:
            loader = self._crm_loader
        elif fmt in self._format_loaders:
            loader = self._format_loaders[fmt]
        else:
            loader = self._format_loaders["json"]

        records = loader.load_raw_content(content, source_type=source_type, source_id=source_id, metadata=metadata)

        if check_duplicates and records:
            self._seen_hashes.add(content_hash)

        return records

    def clear_duplicate_cache(self):
        self._seen_hashes.clear()


# Compatibility aliases for legacy ARCH-1 callers
LoaderRegistry = IngestionRegistry
default_loader_registry = IngestionRegistry()
