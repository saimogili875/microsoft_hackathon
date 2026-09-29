"""
Feature Processor module engineering numerical & categorical pre-offer features
while preserving Core Feature Groups (PRICE, PRODUCT, ORGANIZATION, CUSTOMER).
Maintains exact feature-to-group traceability.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class ProcessedDataset:
    X: pd.DataFrame
    y: pd.Series
    case_ids: pd.Series
    feature_names: List[str]
    feature_to_group: Dict[str, str]  # feature_name -> group_name (PRICE, PRODUCT, ORGANIZATION, CUSTOMER, OTHER)
    group_to_features: Dict[str, List[str]]
    feature_group_summary: Dict[str, int]


class FeatureProcessor:
    """
    Processes raw DataFrames into ML feature matrices while maintaining feature-group traceability metadata.
    """

    GROUP_MAPPINGS = {
        "price_": "PRICE",
        "product_": "PRODUCT",
        "organization_": "ORGANIZATION",
        "customer_": "CUSTOMER",
    }

    def process(self, df: pd.DataFrame) -> ProcessedDataset:
        clean_df = df.copy()

        # Separate case_id and target label
        case_ids = clean_df["case_id"] if "case_id" in clean_df.columns else pd.Series([f"CASE-{i:03d}" for i in range(len(clean_df))])
        y = clean_df["win_loss_outcome"] if "win_loss_outcome" in clean_df.columns else pd.Series(np.zeros(len(clean_df)))

        # Drop non-feature columns
        non_feature_cols = ["case_id", "win_loss_outcome", "timestamp", "created_at", "source_id"]
        feature_cols = [c for c in clean_df.columns if c not in non_feature_cols]

        feature_df = clean_df[feature_cols].copy()

        # Build feature-to-group map before encoding
        feature_to_group: Dict[str, str] = {}
        group_to_features: Dict[str, List[str]] = {
            "PRICE": [],
            "PRODUCT": [],
            "ORGANIZATION": [],
            "CUSTOMER": [],
            "OTHER": [],
        }

        # One-hot encode categorical features while propagating feature group metadata
        processed_dfs = []
        for col in feature_df.columns:
            group = self._determine_group(col)
            
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                processed_dfs.append(feature_df[[col]])
                feature_to_group[col] = group
                group_to_features[group].append(col)
            else:
                # One-hot encode categorical column
                dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=False, dtype=float)
                processed_dfs.append(dummies)
                for dummy_col in dummies.columns:
                    feature_to_group[dummy_col] = group
                    group_to_features[group].append(dummy_col)

        X = pd.concat(processed_dfs, axis=1)

        # Build summary count of features per group
        summary = {grp: len(cols) for grp, cols in group_to_features.items()}

        return ProcessedDataset(
            X=X,
            y=y.astype(int),
            case_ids=case_ids,
            feature_names=list(X.columns),
            feature_to_group=feature_to_group,
            group_to_features=group_to_features,
            feature_group_summary=summary,
        )

    def _determine_group(self, col_name: str) -> str:
        col_lower = col_name.lower()
        for prefix, grp in self.GROUP_MAPPINGS.items():
            if col_lower.startswith(prefix):
                return grp
        return "OTHER"
