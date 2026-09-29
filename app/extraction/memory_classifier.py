"""
Memory Classifier module determining if data is a sales experience, market info, or temporary context.
"""

from typing import Any, Dict
from app.models.normalized_data import NormalizedRecord


class MemoryClassifier:
    """
    Classifies input normalized records into memory categories.
    Categories:
    - sales_experience (Specific deal interactions, tactics, objections, outcomes)
    - market_information (Competitor pricing, market trends, external facts)
    - temporary_context (Unverified drafts, operational logs, pipeline noise)
    """

    @classmethod
    def classify(cls, record: NormalizedRecord) -> str:
        st = record.source_type.lower()
        if st in ["market_info", "competitor_update", "market_intelligence"]:
            return "market_information"

        text = (record.raw_text_content or "").lower()
        if "competitor price" in text or "market pricing" in text or "competitor offers" in text:
            return "market_information"

        if record.objections or record.salesperson_actions or record.outcome or record.deal_id:
            return "sales_experience"

        if len(text.strip()) < 15:
            return "temporary_context"

        return "sales_experience"
