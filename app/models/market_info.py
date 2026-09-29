"""
Market Info Model representing verified external market facts and competitor intelligence.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class MarketInfoRecord(BaseModel):
    """
    Temporal market fact record preserving historical validity.
    """
    market_id: str = Field(default_factory=lambda: f"mkt_{uuid.uuid4().hex[:10]}")
    competitor: str = Field(..., description="Target competitor or market segment")
    topic: str = Field(..., description="Pricing, Feature, Positioning, Strategy")
    statement: str = Field(..., description="Fact statement")
    pricing_context: Optional[str] = Field(None, description="Observed pricing details")
    source_ids: List[str] = Field(default_factory=list, description="Traceability references")
    valid_from: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    valid_until: Optional[str] = Field(None, description="ISO date when fact was superseded or expired")
    status: str = Field("pending_review", description="active | superseded | pending_review")
    verified_by_role: Optional[str] = None
    verified_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
