"""
Raw Data Model representing un-normalized input from all sales data sources.
Follows the Common Input Contract for ARCH-1 and ARCH-2 ingestion.
"""

from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    """
    Common Input Contract: Represents an ingested raw record with original
    provenance, metadata, and un-altered content preserved.
    """
    raw_id: str = Field(default_factory=lambda: f"raw_{uuid.uuid4().hex[:10]}")
    source_id: str = Field(..., description="Original identifier from external system or file hash")
    source_type: str = Field(..., description="e.g. crm_record, sales_conversation, email, call_transcript, audio_call, handwritten_note, meeting_note, sales_notes, deal_history, market_info, feedback")
    file_name: Optional[str] = Field(None, description="Original filename if ingested from a file")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_format: str = Field("unknown", description="json, jsonl, csv, txt, md, pdf, docx, eml, wav, mp3, m4a, mp4, jpg, png")
    raw_content: Union[Dict[str, Any], str, list] = Field(..., description="Original un-altered content or extracted text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source provenance metadata, OCR/STT info, page/line numbers")
    
    # Optional contextual attributes (must not be hallucinated if missing)
    deal_id: Optional[str] = Field(None, description="Associated Deal ID if available")
    customer_id: Optional[str] = Field(None, description="Associated Customer/Client ID if available")
    salesperson_id: Optional[str] = Field(None, description="Associated Salesperson ID if available")
    source_location: Optional[str] = Field(None, description="File path, URL, or database table reference")
    
    # Quality & Extraction provenance
    extraction_method: str = Field("direct", description="direct, ocr, stt, pdf_text, docx_text, email_parse")
    extraction_confidence: Optional[float] = Field(1.0, description="Extraction or OCR/STT confidence score (0.0 to 1.0)")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
