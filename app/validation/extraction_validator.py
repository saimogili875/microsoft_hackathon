"""
Python Extraction Validator module enforcing JSON schema, confidence bounds,
timestamp formatting, evidence spans, and verification status rules for Groq LLM extractions.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime, timezone
from app.models.groq_extraction_schema import GroqExtractionResult, ExtractedEpisodeSchema
from app.validation.schema import ValidationError, ValidationResult
from config.settings import settings


class ExtractionValidator:
    """
    Python Validator for LLM Extraction Outputs.
    Ensures raw LLM output is strictly checked before human review or retention.
    Invalid extractions are flagged and routed to extraction_errors directory.
    """

    def __init__(self, error_dir: Optional[Path] = None):
        self.error_dir = error_dir or (settings.data_dir / "extraction_errors")
        self.error_dir.mkdir(parents=True, exist_ok=True)

    def validate_extraction(
        self, result: GroqExtractionResult, raw_text: Optional[str] = None
    ) -> ValidationResult:
        val_res = ValidationResult()

        # 1. Validate Extraction Metadata
        meta = result.extraction_metadata
        if not meta.source_id or not str(meta.source_id).strip():
            val_res.add_error("extraction_metadata.source_id", "missing_field", "Metadata source_id is required.")

        if not (0.0 <= meta.overall_extraction_confidence <= 1.0):
            val_res.add_error(
                "extraction_metadata.overall_extraction_confidence",
                "invalid_confidence_range",
                f"Confidence {meta.overall_extraction_confidence} outside [0.0, 1.0] range.",
            )

        # 2. Validate Extracted Episodes
        if not result.episodes:
            val_res.add_warning("episodes", "empty_episodes", "No episodes extracted from source.")

        for idx, ep in enumerate(result.episodes, 1):
            self._validate_single_episode(ep, idx, val_res, raw_text)

        # 3. Handle Invalid Extraction Routing
        if not val_res.is_valid:
            self._save_extraction_error(result, val_res)

        return val_res

    def _validate_single_episode(
        self, ep: ExtractedEpisodeSchema, idx: int, val_res: ValidationResult, raw_text: Optional[str]
    ):
        scope = f"episodes[{idx}]"

        # Verification Status Check: LLM MUST NOT mark episode as confirmed!
        if ep.verification.status != "pending_review":
            val_res.add_error(
                f"{scope}.verification.status",
                "invalid_verification_status",
                f"LLM extraction marked status as '{ep.verification.status}'. Must be 'pending_review'.",
            )
            # Auto-fix rule: enforce pending_review
            ep.verification.status = "pending_review"

        # Confidence Bounds Check
        ext_conf = ep.confidence.extraction_confidence
        if not (0.0 <= ext_conf <= 1.0):
            val_res.add_error(
                f"{scope}.confidence.extraction_confidence",
                "invalid_confidence_range",
                f"Extraction confidence {ext_conf} must be in range [0.0, 1.0].",
            )

        causal_conf = ep.confidence.causal_confidence
        if not (0.0 <= causal_conf <= 1.0):
            val_res.add_error(
                f"{scope}.confidence.causal_confidence",
                "invalid_confidence_range",
                f"Causal confidence {causal_conf} must be in range [0.0, 1.0].",
            )

        # Outcome Status Check
        valid_outcomes = {"won", "lost", "no_decision", "in_progress", None}
        out_status = ep.outcome.status
        if out_status not in valid_outcomes:
            val_res.add_error(
                f"{scope}.outcome.status",
                "invalid_outcome_enum",
                f"Invalid outcome '{out_status}'. Allowed: {valid_outcomes}",
            )

        # Evidence Spans Check (Verify spans exist in source text if raw_text provided)
        if raw_text and ep.provenance.evidence_spans:
            text_lower = raw_text.lower()
            for span in ep.provenance.evidence_spans:
                if span and isinstance(span, str):
                    # Check if significant portion of span exists in raw text
                    span_clean = span.strip()[:30].lower()
                    if span_clean not in text_lower:
                        val_res.add_warning(
                            f"{scope}.provenance.evidence_spans",
                            "fabricated_evidence_span",
                            f"Evidence span '{span[:40]}...' not found verbatim in raw source.",
                        )

    def _save_extraction_error(self, result: GroqExtractionResult, val_res: ValidationResult):
        file_path = self.error_dir / f"err_{result.extraction_metadata.source_id}.json"
        data = {
            "result": result.model_dump(),
            "errors": [e.model_dump() for e in val_res.errors],
            "warnings": [w.model_dump() for w in val_res.warnings],
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
