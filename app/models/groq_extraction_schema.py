"""
Pydantic Schema matching the exact Groq Raw Sales Data -> Memory Extraction JSON payload.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ContextSchema(BaseModel):
    stage: Optional[str] = None
    situation: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)


class CustomerSchema(BaseModel):
    objections: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    reaction: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)
    priorities: List[str] = Field(default_factory=list)


class SalespersonSchema(BaseModel):
    tactic: Optional[str] = None
    actions_taken: List[str] = Field(default_factory=list)
    commitments: List[str] = Field(default_factory=list)


class OutcomeSchema(BaseModel):
    status: Optional[str] = Field(None, description="won | lost | no_decision | in_progress | null")
    explicit_reason: Optional[str] = None
    next_step: Optional[str] = None


class CommercialSchema(BaseModel):
    pricing_context: Optional[str] = None
    quoted_price: Optional[str] = None
    budget: Optional[str] = None
    discount: Optional[str] = None


class LearningSchema(BaseModel):
    lesson: Optional[str] = None
    applies_when: List[str] = Field(default_factory=list)
    did_not_hold_when: List[str] = Field(default_factory=list)


class TemporalSchema(BaseModel):
    event_timestamp: Optional[str] = None
    valid_as_of: Optional[str] = None


class ProvenanceSchema(BaseModel):
    source_ids: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)


class ConfidenceSchema(BaseModel):
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0)
    causal_confidence: float = Field(0.0, ge=0.0, le=1.0)


class VerificationSchema(BaseModel):
    status: str = Field("pending_review", description="pending_review | confirmed | corrected | rejected | superseded")


class ExtractedEpisodeSchema(BaseModel):
    episode_id: Optional[str] = None
    deal_id: Optional[str] = None
    client_id: Optional[str] = None
    context: ContextSchema = Field(default_factory=ContextSchema)
    customer: CustomerSchema = Field(default_factory=CustomerSchema)
    salesperson: SalespersonSchema = Field(default_factory=SalespersonSchema)
    outcome: OutcomeSchema = Field(default_factory=OutcomeSchema)
    competition: List[str] = Field(default_factory=list)
    commercial: CommercialSchema = Field(default_factory=CommercialSchema)
    stakeholders: List[str] = Field(default_factory=list)
    products_and_tools: List[str] = Field(default_factory=list)
    learning: LearningSchema = Field(default_factory=LearningSchema)
    temporal: TemporalSchema = Field(default_factory=TemporalSchema)
    provenance: ProvenanceSchema = Field(default_factory=ProvenanceSchema)
    confidence: ConfidenceSchema = Field(default_factory=ConfidenceSchema)
    verification: VerificationSchema = Field(default_factory=VerificationSchema)


class ExtractionMetadataSchema(BaseModel):
    source_id: str
    source_type: str
    source_timestamp: Optional[str] = None
    extraction_version: str = "1.0"
    overall_extraction_confidence: float = Field(0.0, ge=0.0, le=1.0)


class GroqExtractionResult(BaseModel):
    extraction_metadata: ExtractionMetadataSchema
    episodes: List[ExtractedEpisodeSchema] = Field(default_factory=list)
    unresolved_information: List[str] = Field(default_factory=list)
    conflicts_detected: List[str] = Field(default_factory=list)
    market_information: List[Dict[str, Any]] = Field(default_factory=list)
