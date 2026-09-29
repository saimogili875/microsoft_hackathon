"""
Models for Mode 01 — Live Interaction Intelligence.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class LiveTranscriptChunk(BaseModel):
    """
    Inbound live transcript segment from call/meeting stream.
    """
    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:10]}")
    session_id: str = Field(..., description="Active call/meeting session identifier")
    speaker: str = Field("Unknown", description="Rep | Customer | Executive | Speaker Name")
    text: str = Field(..., description="Transcript text snippet")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LiveSignals(BaseModel):
    """
    Real-time extracted signals from live call stream.
    Fields may be incomplete or unknown during an active meeting.
    """
    situation: Optional[str] = None
    customer_concerns: List[str] = Field(default_factory=list)
    customer_questions: List[str] = Field(default_factory=list)
    salesperson_tactics: List[str] = Field(default_factory=list)
    customer_reactions: List[str] = Field(default_factory=list)
    competitors_mentioned: List[str] = Field(default_factory=list)
    pricing_discussed: List[str] = Field(default_factory=list)
    stakeholders_mentioned: List[str] = Field(default_factory=list)
    tools_currently_used: List[str] = Field(default_factory=list)
    customer_goals: List[str] = Field(default_factory=list)
    commitments: List[str] = Field(default_factory=list)
    problems: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    changes_noted: List[str] = Field(default_factory=list)
    
    # Incomplete fields during live call
    outcome: str = Field("unknown", description="won | lost | no_decision | unknown")
    lesson: str = Field("pending", description="Lesson extraction pending call completion")
    causal_confidence: float = Field(0.1, description="Low/unknown during live call")


class LiveInteractionState(BaseModel):
    """
    State container for an ongoing live call session.
    """
    session_id: str = Field(..., description="Call session ID")
    deal_id: Optional[str] = None
    customer_context: Optional[str] = None
    chunks: List[LiveTranscriptChunk] = Field(default_factory=list)
    extracted_signals: LiveSignals = Field(default_factory=LiveSignals)
    recalled_experiences: List[Dict[str, Any]] = Field(default_factory=list)
    live_recommendations: List[str] = Field(default_factory=list)
    start_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
