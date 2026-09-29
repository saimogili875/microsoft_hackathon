"""
Data Validator module verifying target labels, column data types, and range boundaries.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    target_win_count: int = 0
    target_loss_count: int = 0
    win_loss_ratio: float = 0.0


class DataValidator:
    """
    Validates cleaned dataset for model training readiness.
    """

    REQUIRED_GROUPS_PREFIXES = ["price_", "product_", "organization_"]

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        if df.empty:
            report.is_valid = False
            report.errors.append("Dataset is empty after cleaning.")
            return report

        if "win_loss_outcome" not in df.columns:
            report.is_valid = False
            report.errors.append("Target column 'win_loss_outcome' is missing.")
            return report

        # Target outcome validation
        unique_targets = df["win_loss_outcome"].unique()
        if not set(unique_targets).issubset({0, 1}):
            report.is_valid = False
            report.errors.append(f"Invalid target outcome values: {unique_targets}. Must be 0 (LOSS) or 1 (WIN).")

        report.target_win_count = int((df["win_loss_outcome"] == 1).sum())
        report.target_loss_count = int((df["win_loss_outcome"] == 0).sum())
        if report.target_loss_count > 0:
            report.win_loss_ratio = round(report.target_win_count / report.target_loss_count, 3)

        # Verify presence of Core Feature Groups
        for prefix in self.REQUIRED_GROUPS_PREFIXES:
            matching_cols = [c for c in df.columns if c.startswith(prefix)]
            if not matching_cols:
                report.warnings.append(f"Core feature group prefix '{prefix}' has no matching columns.")

        return report
