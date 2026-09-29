"""
Hindsight Deal Memory Schema for persistent historical experience storage.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class HindsightDealMemory:
    document_id: str
    case_id: str
    customer_context: str
    actual_outcome: str  # WIN / LOSS
    predicted_outcome: str
    win_probability: float
    key_price_factors: List[str] = field(default_factory=list)
    key_product_factors: List[str] = field(default_factory=list)
    key_org_factors: List[str] = field(default_factory=list)
    expert_observation: Optional[str] = None
    extracted_lesson: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_hindsight_payload(self) -> Dict[str, Any]:
        content = (
            f"[DEAL HISTORY EXPERIENCES]\n"
            f"Case: {self.case_id}\n"
            f"Customer Context: {self.customer_context}\n"
            f"Outcome: {self.actual_outcome}\n"
            f"AI Model Prediction: {self.predicted_outcome} (Win Prob: {self.win_probability:.1%})\n"
            f"Price Factors: {', '.join(self.key_price_factors) or 'None'}\n"
            f"Product Factors: {', '.join(self.key_product_factors) or 'None'}\n"
            f"Organization Factors: {', '.join(self.key_org_factors) or 'None'}\n"
            f"Expert Observation: {self.expert_observation or 'None'}\n"
            f"Extracted Lesson: {self.extracted_lesson}\n"
        )
        metadata = {
            "case_id": self.case_id,
            "outcome": self.actual_outcome,
            "timestamp": self.timestamp,
            "verification_status": "confirmed",
        }
        return {
            "document_id": self.document_id,
            "content": content,
            "metadata": metadata,
        }
