"""
Raw Data Model representing un-normalized input from various sales data sources.
"""

from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    """
    Represents an ingested raw record with original metadata and content preserved.
    """
    raw_id: str = Field(default_factory=lambda: f"raw_{uuid.uuid4().hex[:10]}")
    source_type: str = Field(..., description="e.g. sales_conversation, crm_record, email, call_transcript, sales_notes, interview, market_info, feedback")
    source_id: str = Field(..., description="Original identifier from external system")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_format: str = Field("unknown", description="json, jsonl, csv, txt, md")
    raw_content: Union[Dict[str, Any], str, list] = Field(..., description="Original un-altered content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source provenance metadata")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
