"""
Expert vs AI + SHAP Comparison Engine.
Compares Sales Expert predictions and reasoning against XGBoost predictions and SHAP explanations per case.
Preserves disagreements cleanly without forcing artificial consensus.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from app.model.predict import CasePrediction
from app.explainability.shap_explainer import CaseShapExplanation
from app.experts.expert_predictions import ExpertEvaluationRecord


@dataclass
class CaseComparisonResult:
    case_id: str
    expert_id: str
    expert_prediction: str  # WIN / LOSS
    ai_prediction: str      # WIN / LOSS
    prediction_agreement: bool
    expert_win_prob: float
    ai_win_prob: float
    expert_important_factors: List[str]
    ai_shap_important_features: List[str]
    agreed_factors: List[str]
    expert_only_factors: List[str]
    ai_only_factors: List[str]
    comparison_summary: str


class ExpertAiComparisonEngine:
    """
    Compares Sales Expert qualitative reasoning and predictions against XGBoost + SHAP output.
    """

    def compare_case(
        self,
        ai_pred: CasePrediction,
        shap_expl: CaseShapExplanation,
        expert_record: ExpertEvaluationRecord,
    ) -> CaseComparisonResult:

        case_id = ai_pred.case_id
        expert_id = expert_record.expert_id

        expert_outcome = expert_record.q5_assessed_outcome.upper()
        if expert_outcome not in ["WIN", "LOSS"]:
            expert_outcome = "WIN" if expert_record.expert_predicted_prob >= 0.50 else "LOSS"

        ai_outcome = ai_pred.predicted_label.upper()
        prediction_agreement = (expert_outcome == ai_outcome)

        # Compare important factors
        expert_factors = set(expert_record.expert_important_factors)
        ai_shap_features = [f.feature_name for f in shap_expl.top_features[:5]]
        ai_factors_set = set(ai_shap_features)

        # Match factor prefixes or exact names
        agreed = []
        for ef in expert_factors:
            for af in ai_factors_set:
                if ef in af or af in ef:
                    agreed.append(ef)
                    break

        agreed_set = set(agreed)
        expert_only = list(expert_factors - agreed_set)
        ai_only = [af for af in ai_shap_features if not any(af in ef or ef in af for ef in expert_factors)]

        # Construct comparison summary text
        agreement_text = "AGREED" if prediction_agreement else "DISAGREED"
        agreed_str = ", ".join(agreed) if agreed else "None"
        exp_only_str = ", ".join(expert_only) if expert_only else "None"
        ai_only_str = ", ".join(ai_only) if ai_only else "None"

        summary = (
            f"Case {case_id} [{expert_id} vs AI]: Predictions {agreement_text} ({expert_outcome} vs {ai_outcome}). "
            f"Agreed Factors: [{agreed_str}]. "
            f"Expert-Only: [{exp_only_str}]. "
            f"AI-Identified: [{ai_only_str}]."
        )

        return CaseComparisonResult(
            case_id=case_id,
            expert_id=expert_id,
            expert_prediction=expert_outcome,
            ai_prediction=ai_outcome,
            prediction_agreement=prediction_agreement,
            expert_win_prob=expert_record.expert_predicted_prob,
            ai_win_prob=ai_pred.win_probability,
            expert_important_factors=list(expert_factors),
            ai_shap_important_features=ai_shap_features,
            agreed_factors=agreed,
            expert_only_factors=expert_only,
            ai_only_factors=ai_only,
            comparison_summary=summary,
        )
