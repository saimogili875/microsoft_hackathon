"""
Models for Mode 02-B — Post-Meeting Change Detection.
Supports detection across 18 specific change categories.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class ChangeCategory(str, Enum):
    NEW_COMPETITOR = "new_competitor"
    COMPETITOR_PRICE_CHANGE = "competitor_price_change"
    CUSTOMER_SWITCHING_TOOLS = "customer_switching_tools"
    CUSTOMER_ADOPTING_TOOL = "customer_adopting_tool"
    CUSTOMER_USING_PRODUCT_DIFFERENTLY = "customer_using_product_differently"
    NEW_STAKEHOLDER = "new_stakeholder"
    STAKEHOLDER_LEAVING = "stakeholder_leaving"
    NEW_CUSTOMER_REQUIREMENT = "new_customer_requirement"
    NEW_OBJECTION = "new_objection"
    RESOLVED_OBJECTION = "resolved_objection"
    NEW_PRICING_EXPECTATION = "new_pricing_expectation"
    NEW_BUSINESS_PRIORITY = "new_business_priority"
    NEW_COMPANY_WORKFLOW_CHANGE = "new_company_workflow_change"
    NEW_PAIN_POINT = "new_pain_point"
    NEW_COMMITMENT = "new_commitment"
    PREVIOUS_COMMITMENT_INVALID = "previous_commitment_invalid"
    NEW_MARKET_INFORMATION = "new_market_information"
    CUSTOMER_RELATIONSHIP_CHANGE = "customer_relationship_change"


class ChangeEvent(BaseModel):
    """
    Structured change record produced when comparing previous client state with a new interaction.
    Does NOT automatically overwrite old memory. Remains in pending_review until human confirmed.
    """
    change_id: str = Field(default_factory=lambda: f"chg_{uuid.uuid4().hex[:10]}")
    deal_id: Optional[str] = None
    category: ChangeCategory = Field(..., description="One of the 18 change categories")
    old_information: str = Field(..., description="Previous baseline knowledge")
    new_evidence: str = Field(..., description="Newly observed evidence from recent interaction")
    difference: str = Field(..., description="Analytical summary of the shift or impact")
    status: str = Field("pending_review", description="pending_review | confirmed | rejected | superseded")
    source_ids: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ClientState(BaseModel):
    """
    Consolidated baseline state for a client / deal.
    """
    deal_id: str
    customer_context: str
    stakeholders: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)
    competitor_pricing: Dict[str, str] = Field(default_factory=dict)
    active_objections: List[str] = Field(default_factory=list)
    resolved_objections: List[str] = Field(default_factory=list)
    commitments: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
