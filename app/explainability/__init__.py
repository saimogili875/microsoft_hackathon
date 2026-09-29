"""
Explainability Package for SHAP (SHapley Additive exPlanations) generation.
Computes per-case feature contributions, direction of impact, and core feature-group SHAP importance.
"""

from app.explainability.shap_explainer import ShapExplainerService, CaseShapExplanation
from app.explainability.feature_importance import FeatureImportanceAnalyzer, GroupShapSummary

__all__ = [
    "ShapExplainerService",
    "CaseShapExplanation",
    "FeatureImportanceAnalyzer",
    "GroupShapSummary",
]
