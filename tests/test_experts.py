"""
Unit tests for Q1-Q5 Questionnaire, Expert Predictions, Expert vs AI comparison, and Prediction Alone vs Prediction + Explanation evaluation.
"""

import pytest
from app.experts.questionnaire import ExpertQuestionnaire
from app.experts.expert_predictions import ExpertEvaluationRecord
from app.experts.expert_comparison import ExpertAiComparisonEngine
from app.evaluation.usefulness import UsefulnessEvaluator
from app.model.predict import CasePrediction
from app.explainability.shap_explainer import CaseShapExplanation, FeatureShapContribution


def test_questionnaire_q1_to_q5_validation():
    q = ExpertQuestionnaire()
    r1 = q.create_response("E1", "C1", "Q1", "3-10 years")
    r2 = q.create_response("E1", "C1", "Q2", 4)
    r4 = q.create_response("E1", "C1", "Q4", "Yes")
    r5 = q.create_response("E1", "C1", "Q5", "WIN")
    
    assert r1.response == "3-10 years"
    assert r2.response == 4
    assert r5.response == "WIN"
    
    with pytest.raises(ValueError):
        q.create_response("E1", "C1", "Q1", "Invalid_Experience")


def test_expert_vs_ai_comparison():
    engine = ExpertAiComparisonEngine()
    ai_pred = CasePrediction(case_id="C1", predicted_class=1, predicted_label="WIN", win_probability=0.8, loss_probability=0.2, confidence=0.8)
    
    feat_contrib = FeatureShapContribution(
        feature_name="price_amount", shap_value=0.5, feature_value=100, direction="POSITIVE_WIN", feature_group="PRICE", explanation_text="High impact"
    )
    shap_expl = CaseShapExplanation(
        case_id="C1", predicted_label="WIN", win_probability=0.8, loss_probability=0.2,
        top_features=[feat_contrib], positive_contributors=[feat_contrib], negative_contributors=[], human_readable_explanation="WIN text"
    )
    
    exp_rec = ExpertEvaluationRecord(
        case_id="C1", expert_id="EX-1", q1_b2b_experience="3-10 years", q2_ai_tool_usage=3, q3_predicting_experience="Yes",
        q4_information_sufficiency="Yes", q5_assessed_outcome="WIN", expert_predicted_prob=0.85, expert_confidence=0.9,
        expert_important_factors=["price_amount"], expert_explanation="Price is competitive",
        rating_prediction_alone=4.0, rating_prediction_and_explanation=6.0
    )
    
    comp = engine.compare_case(ai_pred, shap_expl, exp_rec)
    assert comp.prediction_agreement is True
    assert "price_amount" in comp.agreed_factors


def test_usefulness_evaluation():
    evaluator = UsefulnessEvaluator()
    exp_recs = [
        ExpertEvaluationRecord(
            case_id=f"C{i}", expert_id="EX-1", q1_b2b_experience="3-10 years", q2_ai_tool_usage=3, q3_predicting_experience="Yes",
            q4_information_sufficiency="Yes", q5_assessed_outcome="WIN", expert_predicted_prob=0.8, expert_confidence=0.9,
            expert_important_factors=["price_amount"], expert_explanation="Explanation",
            rating_prediction_alone=3.5, rating_prediction_and_explanation=5.5
        )
        for i in range(5)
    ]
    res = evaluator.evaluate_records(exp_recs)
    assert res.mean_rating_prediction_alone == 3.5
    assert res.mean_rating_prediction_and_explanation == 5.5
    assert res.difference_mean_usefulness == 2.0
    assert res.hypothesis_supported is True
