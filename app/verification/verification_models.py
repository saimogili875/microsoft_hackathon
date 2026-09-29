"""
Human Verification models and actions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class VerificationAction(str, Enum):
    CONFIRM = "confirm"
    CORRECT = "correct"
    REJECT = "reject"


class VerificationRequest(BaseModel):
    """
    Payload sent by human reviewer (e.g. Account Executive, Sales Manager).
    """
    episode_id: str = Field(..., description="Target episode identifier")
    action: VerificationAction = Field(..., description="confirm | correct | reject")
    reviewer_role: str = Field("account_executive", description="e.g. account_executive, sales_manager, admin")
    corrections: Optional[Dict[str, Any]] = Field(
        None, description="Updated field values if action == 'correct'"
    )
    comments: Optional[str] = Field(None, description="Reviewer feedback or notes")
