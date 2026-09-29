"""
Data Validator module enforcing schema, format, outcome, and duplicate rules.
"""

from typing import Any, Dict, List, Set, Union
from datetime import datetime
import hashlib
import json
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.validation.schema import (
    ValidationError,
    ValidationResult,
    SUPPORTED_SOURCE_TYPES,
    VALID_OUTCOMES,
)


class DataValidator:
    """
    Validator engine detecting missing fields, invalid timestamps/outcomes,
    unsupported sources, duplicates, and conflicting identifiers.
    """

    def __init__(self):
        self._seen_hashes: Set[str] = set()
        self._seen_source_ids: Set[str] = set()

    def validate_raw(self, raw_record: RawRecord) -> ValidationResult:
        result = ValidationResult()

        # Check required fields
        if not raw_record.source_id or not str(raw_record.source_id).strip():
            result.add_error("source_id", "missing_field", "Raw record is missing a valid source_id.")

        if not raw_record.source_type or not str(raw_record.source_type).strip():
            result.add_error("source_type", "missing_field", "Raw record is missing a valid source_type.")
        elif raw_record.source_type not in SUPPORTED_SOURCE_TYPES:
            result.add_warning(
                "source_type",
                "unsupported_source",
                f"Source type '{raw_record.source_type}' is unrecognized.",
                {"supported": list(SUPPORTED_SOURCE_TYPES)},
            )

        # Check timestamp
        if not self._is_valid_iso_timestamp(raw_record.timestamp):
            result.add_error("timestamp", "invalid_date", f"Invalid ISO 8601 timestamp: '{raw_record.timestamp}'")

        # Check content
        if raw_record.raw_content is None or raw_record.raw_content == "" or raw_record.raw_content == {}:
            result.add_error("raw_content", "malformed_structure", "Raw record content is empty or null.")

        # Duplicate check by source_id + content hash
        content_hash = self._compute_content_hash(raw_record.source_id, raw_record.raw_content)
        if content_hash in self._seen_hashes:
            result.add_error(
                "record", "duplicate_record", f"Duplicate record detected for source_id '{raw_record.source_id}'."
            )
        else:
            self._seen_hashes.add(content_hash)

        return result

    def validate_normalized(self, norm_record: NormalizedRecord) -> ValidationResult:
        result = ValidationResult()

        if not norm_record.source_id:
            result.add_error("source_id", "missing_field", "Normalized record missing source_id.")

        if norm_record.outcome and norm_record.outcome not in VALID_OUTCOMES:
            result.add_error(
                "outcome",
                "invalid_outcome",
                f"Invalid deal outcome '{norm_record.outcome}'. Allowed: {VALID_OUTCOMES}",
            )

        if not self._is_valid_iso_timestamp(norm_record.timestamp):
            result.add_error("timestamp", "invalid_date", f"Invalid timestamp: {norm_record.timestamp}")

        # Check for conflicting deal details (e.g. outcome stated as won but deal stage is closed lost)
        if norm_record.outcome == "won" and norm_record.deal_stage:
            if "lost" in norm_record.deal_stage.lower():
                result.add_error(
                    "outcome_conflict",
                    "conflicting_identifiers",
                    f"Contradictory data: outcome is 'won' but deal_stage is '{norm_record.deal_stage}'",
                )

        return result

    def validate_batch(self, records: List[NormalizedRecord]) -> ValidationResult:
        result = ValidationResult()
        deal_outcomes: Dict[str, str] = {}

        for rec in records:
            single_res = self.validate_normalized(rec)
            for err in single_res.errors:
                result.errors.append(err)
                result.is_valid = False
            for warn in single_res.warnings:
                result.warnings.append(warn)

            # Check cross-record deal conflicts
            if rec.deal_id and rec.outcome:
                if rec.deal_id in deal_outcomes and deal_outcomes[rec.deal_id] != rec.outcome:
                    result.add_error(
                        "deal_id",
                        "conflicting_information",
                        f"Conflicting outcomes for deal '{rec.deal_id}': '{deal_outcomes[rec.deal_id]}' vs '{rec.outcome}'",
                    )
                else:
                    deal_outcomes[rec.deal_id] = rec.outcome

        return result

    def _is_valid_iso_timestamp(self, ts: str) -> bool:
        if not ts or not isinstance(ts, str):
            return False
        try:
            # Flexible ISO parsing
            ts_clean = ts.replace("Z", "+00:00")
            datetime.fromisoformat(ts_clean)
            return True
        except ValueError:
            # Check YYYY-MM-DD
            try:
                datetime.strptime(ts[:10], "%Y-%m-%d")
                return True
            except ValueError:
                return False

    def _compute_content_hash(self, source_id: str, content: Any) -> str:
        s = f"{source_id}:{json.dumps(content, sort_keys=True, default=str)}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
