"""
Data Cleaner module handling missing values, duplicate detection, invalid rows,
and strictly preventing post-offer information leakage into pre-offer predictions.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# List of forbidden post-offer fields that must never leak into pre-offer predictions
POST_OFFER_FORBIDDEN_FIELDS = [
    "post_deal_churn",
    "actual_implementation_time_days",
    "customer_satisfaction_score_after_sale",
    "final_actual_revenue_realized",
    "contract_renewal_status",
    "cancellation_reason_post_close",
]


@dataclass
class CleaningReport:
    initial_records: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    missing_values_imputed: Dict[str, int] = field(default_factory=dict)
    leaked_fields_removed: List[str] = field(default_factory=list)
    final_valid_records: int = 0


class DataCleaner:
    """
    Cleans raw data, imputes missing values reproducibly, drops duplicates, and strips post-offer leaked fields.
    """

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        report = CleaningReport(initial_records=len(df))
        clean_df = df.copy()

        # 1. Post-Offer Leakage Protection: Remove post-deal fields
        leaked_cols = [col for col in clean_df.columns if col.lower() in POST_OFFER_FORBIDDEN_FIELDS or col.startswith("post_")]
        if leaked_cols:
            clean_df.drop(columns=leaked_cols, inplace=True)
            report.leaked_fields_removed = leaked_cols

        # 2. Duplicate Case Detection & Deduplication
        if "case_id" in clean_df.columns:
            dups = clean_df.duplicated(subset=["case_id"], keep="first")
            num_dups = dups.sum()
            if num_dups > 0:
                clean_df = clean_df[~dups].copy()
                report.duplicates_removed = int(num_dups)

        # 3. Invalid Row Stripping (e.g. missing target label or missing case_id)
        invalid_mask = pd.Series(False, index=clean_df.index)
        if "win_loss_outcome" in clean_df.columns:
            invalid_mask |= clean_df["win_loss_outcome"].isna() | (~clean_df["win_loss_outcome"].isin([0, 1]))
        if "case_id" in clean_df.columns:
            invalid_mask |= clean_df["case_id"].isna()

        num_invalid = invalid_mask.sum()
        if num_invalid > 0:
            clean_df = clean_df[~invalid_mask].copy()
            report.invalid_rows_removed = int(num_invalid)

        # 4. Impute Missing Values reproducibly
        for col in clean_df.columns:
            missing_count = clean_df[col].isna().sum()
            if missing_count > 0:
                report.missing_values_imputed[col] = int(missing_count)
                if pd.api.types.is_numeric_dtype(clean_df[col]):
                    median_val = clean_df[col].median()
                    clean_df[col] = clean_df[col].fillna(median_val)
                else:
                    mode_val = clean_df[col].mode()[0] if not clean_df[col].mode().empty else "Unknown"
                    clean_df[col] = clean_df[col].fillna(mode_val)

        report.final_valid_records = len(clean_df)
        return clean_df, report
