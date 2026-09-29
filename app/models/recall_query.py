"""
Pydantic model representing structured query analysis for ARCH-2 Hindsight Recall.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecallQuery(BaseModel):
    """
    Structured query understanding model for Hindsight recall.
    Does NOT hallucinate unknown fields (uses None or [] when unavailable).
    """
    original_query: str = Field(..., description="Raw salesperson query string")
    client_id: Optional[str] = Field(None, description="Target client or company name if identified")
    deal_id: Optional[str] = Field(None, description="Target deal ID if identified")
    topic: Optional[str] = Field(None, description="General topic (e.g. pricing, implementation, competitor)")
    stage: Optional[str] = Field(None, description="Deal stage if specified")
    objection: Optional[str] = Field(None, description="Specific objection raised")
    competitor: Optional[str] = Field(None, description="Competitor name mentioned")
    pricing_topic: Optional[str] = Field(None, description="Pricing constraint or terms")
    tactic: Optional[str] = Field(None, description="Target salesperson tactic")
    requested_information: List[str] = Field(
        default_factory=lambda: ["tactic", "reaction", "outcome", "lesson"],
        description="Types of details requested (previous tactics, outcomes, etc.)"
    )
    memory_types: List[str] = Field(
        default_factory=lambda: ["sales_experience"],
        description="Target memory classification"
    )
    time_range: Optional[str] = Field(None, description="Time period filter if specified")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    retrieval_strategy: str = Field("semantic_narrow", description="semantic_narrow | client_scoped | broad")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
