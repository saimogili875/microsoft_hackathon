"""
Validation models and constants for pipeline validation rules.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationError(BaseModel):
    """
    Structured validation error report item.
    """
    field: str = Field(..., description="Target field name or scope")
    error_type: str = Field(..., description="e.g. missing_field, invalid_date, invalid_outcome, duplicate, unsupported_source, conflict")
    message: str = Field(..., description="Human readable description")
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """
    Overall validation assessment result.
    """
    is_valid: bool = True
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)

    def add_error(self, field: str, error_type: str, message: str, details: Dict[str, Any] = None):
        self.is_valid = False
        self.errors.append(ValidationError(field=field, error_type=error_type, message=message, details=details or {}))

    def add_warning(self, field: str, error_type: str, message: str, details: Dict[str, Any] = None):
        self.warnings.append(ValidationError(field=field, error_type=error_type, message=message, details=details or {}))


SUPPORTED_SOURCE_TYPES = {
    "salesperson_interview",
    "sales_conversation",
    "crm_record",
    "sales_notes",
    "email",
    "call_transcript",
    "deal_history",
    "market_info",
    "salesperson_feedback",
    "json_record",
    "csv_row",
}

VALID_OUTCOMES = {"won", "lost", "no_decision", "in_progress", None}
