"""
Ingestion Package exposing loaders, OCR/STT adapters, and the unified IngestionRegistry.
"""

from app.ingestion.base import BaseLoader, BaseOCRAdapter, BaseSTTAdapter, OCRExtractionResult, STTTranscriptionResult
from app.ingestion.file_detector import FileDetector
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.csv_loader import CSVLoader
from app.ingestion.json_loader import JsonLoader, JSONLoader
from app.ingestion.jsonl_loader import JsonlLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.email_loader import EmailLoader
from app.ingestion.audio_loader import AudioLoader
from app.ingestion.image_loader import ImageLoader
from app.ingestion.crm_loader import CrmLoader
from app.ingestion.ocr import MockOCRAdapter, TesseractOCRAdapter, get_ocr_adapter, set_ocr_adapter
from app.ingestion.transcription import MockSTTAdapter, WhisperSTTAdapter, get_stt_adapter, set_stt_adapter
from app.ingestion.registry import IngestionRegistry

__all__ = [
    "BaseLoader",
    "BaseOCRAdapter",
    "BaseSTTAdapter",
    "OCRExtractionResult",
    "STTTranscriptionResult",
    "FileDetector",
    "TxtLoader",
    "MarkdownLoader",
    "CSVLoader",
    "JsonLoader",
    "JSONLoader",
    "JsonlLoader",
    "PdfLoader",
    "DocxLoader",
    "EmailLoader",
    "AudioLoader",
    "ImageLoader",
    "CrmLoader",
    "MockOCRAdapter",
    "TesseractOCRAdapter",
    "get_ocr_adapter",
    "set_ocr_adapter",
    "MockSTTAdapter",
    "WhisperSTTAdapter",
    "get_stt_adapter",
    "set_stt_adapter",
    "IngestionRegistry",
    "LoaderRegistry",
    "default_loader_registry",
]
