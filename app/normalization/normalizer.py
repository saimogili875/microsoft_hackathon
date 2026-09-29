"""
Data Normalizer module converting RawRecord into NormalizedRecord.
"""

from typing import Any, Dict, List, Optional
import re
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.normalization.field_mapper import FieldMapper
from app.normalization.cleaners import DataCleaner


class DataNormalizer:
    """
    Normalizes raw structured/unstructured inputs into unified internal representation.
    Guarantees no hallucinated missing data (uses None/empty list when unavailable).
    """

    def __init__(self, anonymize: bool = True):
        self.anonymize = anonymize

    def normalize(self, raw_record: RawRecord) -> NormalizedRecord:
        content = raw_record.raw_content
        raw_text_content: Optional[str] = None
        data_dict: Dict[str, Any] = {}

        if isinstance(content, dict):
            data_dict = content
            raw_text_content = str(content)
        elif isinstance(content, str):
            raw_text_content = content
            data_dict = self._parse_text_to_dict(content)
        elif isinstance(content, list):
            raw_text_content = str(content)
            data_dict = {"items": content}

        # Scrub PII from raw_text_content if string
        if raw_text_content:
            raw_text_content = DataCleaner.scrub_pii(raw_text_content, anonymize=self.anonymize)

        deal_id = FieldMapper.find_key(data_dict, "deal_id")
        if deal_id:
            deal_id = str(deal_id).strip()

        customer_context = FieldMapper.find_key(data_dict, "customer_context")
        if customer_context and isinstance(customer_context, str):
            customer_context = DataCleaner.scrub_pii(customer_context, anonymize=self.anonymize)

        deal_stage = FieldMapper.find_key(data_dict, "deal_stage")
        if deal_stage and isinstance(deal_stage, str):
            deal_stage = deal_stage.strip()

        stakeholders = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "stakeholders"), anonymize=self.anonymize
        )

        customer_goal = FieldMapper.find_key(data_dict, "customer_goal")
        if customer_goal and isinstance(customer_goal, str):
            customer_goal = DataCleaner.scrub_pii(customer_goal, anonymize=self.anonymize)

        pain_points = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "pain_points"), anonymize=self.anonymize
        )
        objections = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "objections"), anonymize=self.anonymize
        )
        competitors = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "competitors"), anonymize=self.anonymize
        )
        pricing_information = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "pricing_information"), anonymize=self.anonymize
        )
        salesperson_actions = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "salesperson_actions"), anonymize=self.anonymize
        )
        customer_reactions = DataCleaner.clean_string_list(
            FieldMapper.find_key(data_dict, "customer_reactions"), anonymize=self.anonymize
        )

        outcome = FieldMapper.find_key(data_dict, "outcome")
        if outcome:
            outcome = self._normalize_outcome(str(outcome))

        source_ref = {
            "raw_id": raw_record.raw_id,
            "source_id": raw_record.source_id,
            "source_type": raw_record.source_type,
            "file_format": raw_record.file_format,
            "metadata": raw_record.metadata,
        }

        return NormalizedRecord(
            source_id=raw_record.source_id,
            source_type=raw_record.source_type,
            deal_id=deal_id or raw_record.metadata.get("deal_id"),
            customer_context=customer_context,
            deal_stage=deal_stage,
            stakeholders=stakeholders,
            customer_goal=customer_goal,
            pain_points=pain_points,
            objections=objections,
            competitors=competitors,
            pricing_information=pricing_information,
            salesperson_actions=salesperson_actions,
            customer_reactions=customer_reactions,
            outcome=outcome,
            raw_text_content=raw_text_content,
            source_references=[source_ref],
            timestamp=raw_record.timestamp,
        )

    def _normalize_outcome(self, val: str) -> Optional[str]:
        v = val.lower().strip()
        if any(w in v for w in ["won", "closed_won", "closed won", "success", "signed"]):
            return "won"
        if any(w in v for w in ["lost", "closed_lost", "closed lost", "rejected", "failed"]):
            return "lost"
        if any(w in v for w in ["no_decision", "no decision", "stalled", "ghosted"]):
            return "no_decision"
        if any(w in v for w in ["in_progress", "in progress", "open", "negotiating"]):
            return "in_progress"
        return v

    def _parse_text_to_dict(self, text: str) -> Dict[str, Any]:
        """
        Parses structured key-value lines or bullet blocks from text if present.
        """
        parsed = {}
        lines = text.splitlines()
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                k = parts[0].strip().lower().replace(" ", "_")
                v = parts[1].strip()
                if k and v:
                    parsed[k] = v
        return parsed
