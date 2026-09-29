"""
Normalized Data Model representing standardized internal structure for all sales records.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class NormalizedRecord(BaseModel):
    """
    Unified canonical schema for ingested sales events.
    Fields without explicit data are left as None or empty lists.
    """
    record_id: str = Field(default_factory=lambda: f"norm_{uuid.uuid4().hex[:10]}")
    source_id: str = Field(..., description="Original source ID")
    source_type: str = Field(..., description="Source classification type")
    deal_id: Optional[str] = None
    customer_context: Optional[str] = None
    deal_stage: Optional[str] = None
    stakeholders: List[str] = Field(default_factory=list)
    customer_goal: Optional[str] = None
    pain_points: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)
    pricing_information: List[str] = Field(default_factory=list)
    salesperson_actions: List[str] = Field(default_factory=list)
    customer_reactions: List[str] = Field(default_factory=list)
    outcome: Optional[str] = Field(None, description="won | lost | no_decision | in_progress | None")
    raw_text_content: Optional[str] = None
    source_references: List[Dict[str, Any]] = Field(default_factory=list, description="Traceability links to raw input")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    normalized_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
