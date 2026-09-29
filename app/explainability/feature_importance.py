"""
Feature Importance Analyzer aggregating mean absolute SHAP values per feature
and summarizing SHAP importance across Core Feature Groups (PRICE, PRODUCT, ORGANIZATION, CUSTOMER).
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class GroupShapSummary:
    feature_group: str
    total_shap_importance: float
    percentage_importance: float
    feature_count: int
    top_features: List[Tuple[str, float]]


class FeatureImportanceAnalyzer:
    """
    Computes global mean absolute SHAP importance per feature and per Core Feature Group.
    """

    def analyze(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_to_group: Dict[str, str],
    ) -> Tuple[Dict[str, float], List[GroupShapSummary]]:

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importances = dict(zip(X.columns, [round(float(v), 4) for v in mean_abs_shap]))

        # Group-level aggregations
        group_totals: Dict[str, float] = {
            "PRICE": 0.0,
            "PRODUCT": 0.0,
            "ORGANIZATION": 0.0,
            "CUSTOMER": 0.0,
            "OTHER": 0.0,
        }
        group_features: Dict[str, List[Tuple[str, float]]] = {
            "PRICE": [],
            "PRODUCT": [],
            "ORGANIZATION": [],
            "CUSTOMER": [],
            "OTHER": [],
        }

        for feat_name, imp in feature_importances.items():
            grp = feature_to_group.get(feat_name, "OTHER")
            if grp not in group_totals:
                group_totals[grp] = 0.0
                group_features[grp] = []

            group_totals[grp] += imp
            group_features[grp].append((feat_name, imp))

        total_sum = sum(group_totals.values()) or 1.0

        group_summaries: List[GroupShapSummary] = []
        for grp, total_imp in group_totals.items():
            feats = group_features[grp]
            feats.sort(key=lambda x: x[1], reverse=True)
            pct = round((total_imp / total_sum) * 100.0, 2)

            group_summaries.append(
                GroupShapSummary(
                    feature_group=grp,
                    total_shap_importance=round(total_imp, 4),
                    percentage_importance=pct,
                    feature_count=len(feats),
                    top_features=feats[:5],
                )
            )

        group_summaries.sort(key=lambda x: x.total_shap_importance, reverse=True)
        return feature_importances, group_summaries
