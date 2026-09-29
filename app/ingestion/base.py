"""
Base Interfaces and Contracts for Ingestion, OCR, and Speech-to-Text Adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.models.raw_data import RawRecord


@dataclass
class OCRExtractionResult:
    """
    Standard result returned by any pluggable OCR Adapter.
    """
    extracted_text: str
    confidence: float
    page_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_low_confidence: bool = False
    original_format: Optional[str] = None
    error: Optional[str] = None


@dataclass
class STTTranscriptionResult:
    """
    Standard result returned by any pluggable Speech-to-Text Adapter.
    """
    transcript: str
    confidence: float
    duration_seconds: float = 0.0
    speaker_segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    audio_format: Optional[str] = None
    error: Optional[str] = None


class BaseOCRAdapter(ABC):
    """
    Pluggable OCR Adapter interface for handwritten notes, scanned PDFs, and image documents.
    """

    @abstractmethod
    def extract_text(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> OCRExtractionResult:
        """
        Extract text from an image or scanned document.
        Must never raise unhandled exceptions; return error status in OCRExtractionResult on failure.
        """
        pass


class BaseSTTAdapter(ABC):
    """
    Pluggable Speech-to-Text Adapter interface for audio recordings and calls.
    """

    @abstractmethod
    def transcribe(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> STTTranscriptionResult:
        """
        Transcribe audio recording into text with speaker labels and confidence.
        Must never raise unhandled exceptions; return error status in STTTranscriptionResult on failure.
        """
        pass


class BaseLoader(ABC):
    """
    Abstract base class for all file and source loaders.
    """

    @abstractmethod
    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        """
        Load records from a file path.
        """
        pass

    @abstractmethod
    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        """
        Load records directly from raw in-memory content.
        """
        pass
