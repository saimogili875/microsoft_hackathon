"""
Episode Model representing extracted, structured sales experience units.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class Episode(BaseModel):
    """
    Conceptual unit of sales experience extracted from normalized records.
    Contains separate extraction and causal confidence scores.
    """
    episode_id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:10]}")
    deal_id: Optional[str] = None
    situation: str = Field(..., description="Contextual scenario and customer environment")
    objection: str = Field(..., description="Primary friction point or customer objection")
    tactic: str = Field(..., description="Sales strategy/action executed")
    customer_reaction: str = Field(..., description="Immediate customer response to tactic")
    outcome: str = Field(..., description="won | lost | no_decision")
    why: str = Field(..., description="Causal reasoning / underlying root cause")
    lesson: str = Field(..., description="Actionable principle learned")
    applies_when: List[str] = Field(default_factory=list, description="Conditions where tactic works")
    did_not_hold_when: List[str] = Field(default_factory=list, description="Counter-examples/boundary conditions")
    pricing_context: Optional[str] = Field(None, description="Pricing terms or constraints")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valid_as_of: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    source_ids: List[str] = Field(default_factory=list, description="Traceability list of raw/normalized source IDs")
    
    # Confidence metrics - kept strictly separate!
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in accuracy of fact extraction")
    causal_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence that tactic caused the outcome")
    
    # Verification metadata
    verification_status: str = Field("pending_review", description="pending_review | confirmed | rejected | corrected")
    verified_by_role: Optional[str] = None
    verified_at: Optional[str] = None
    corrections: List[Dict[str, Any]] = Field(default_factory=list, description="Audit log of human edits")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
