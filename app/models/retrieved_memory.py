"""
Pydantic models for ARCH-2 Retrieved Memories and Arch-3 Context Object.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.recall_query import RecallQuery


class RetrievedMemory(BaseModel):
    """
    Representation of a memory retrieved from Hindsight for ARCH-2 reasoning.
    Preserves all provenance, verification, and temporal metadata without altering original memory.
    """
    memory_id: str = Field(..., description="Document ID e.g. episode:<id> or market:<comp>:<date>")
    episode_id: Optional[str] = None
    client_id: Optional[str] = None
    deal_id: Optional[str] = None
    content: str = Field(..., description="Self-contained memory narrative text")
    timestamp: Optional[str] = None
    valid_as_of: Optional[str] = None
    verification_status: str = Field("confirmed", description="pending_review | confirmed | corrected | rejected | superseded")
    source_ids: List[str] = Field(default_factory=list)
    relevance_score: float = Field(1.0, ge=0.0, le=1.0)
    is_superseded: bool = False
    is_pending: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Arch2ContextObject(BaseModel):
    """
    Structured ARCH-2 Output Object ready for ARCH-3 consumption.
    """
    query: str
    query_analysis: RecallQuery
    memory_found: bool = True
    retrieved_memory_count: int = 0
    retrieved_memories: List[RetrievedMemory] = Field(default_factory=list)
    conflicts_detected: List[Dict[str, Any]] = Field(default_factory=list)
    past_context: Dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
