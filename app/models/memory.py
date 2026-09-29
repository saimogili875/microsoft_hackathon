"""
Memory Model representing the standardized payload sent to Hindsight RETAIN.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field


class HindsightMemoryPayload(BaseModel):
    """
    Verified Hindsight payload ready for retention.
    """
    document_id: str = Field(..., description="Stable document ID e.g. episode:<id> or market:<competitor>:<date>")
    content: str = Field(..., description="Self-contained concise narrative text block")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including source_ids, verification, validity")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
