"""
Unit tests for SHAP explainer, per-case positive/negative contributions, and group SHAP totals.
"""

import pytest
import pandas as pd
from app.data.feature_processor import FeatureProcessor
from app.model.train import ModelTrainer
from app.model.predict import ModelPredictor
from app.explainability.shap_explainer import ShapExplainerService
from app.explainability.feature_importance import FeatureImportanceAnalyzer


def test_shap_explanations_and_group_totals():
    df = pd.DataFrame([
        {"case_id": f"CASE-{i:03d}", "price_amount": float(i*100), "product_score": float(i%5), "win_loss_outcome": i%2}
        for i in range(20)
    ])
    processor = FeatureProcessor()
    dataset = processor.process(df)
    
    trainer = ModelTrainer(test_size=0.25, random_state=42)
    train_res = trainer.train(dataset)
    
    predictor = ModelPredictor()
    pred_res = predictor.predict_dataset(train_res.model, train_res.X_test, train_res.test_case_ids)
    
    explainer = ShapExplainerService(train_res.model, dataset.feature_to_group)
    explanations, shap_vals = explainer.explain_dataset(train_res.X_test, train_res.test_case_ids, pred_res.y_pred, pred_res.y_proba)
    
    assert len(explanations) == 5
    assert explanations[0].top_features is not None
    
    analyzer = FeatureImportanceAnalyzer()
    _, group_shaps = analyzer.analyze(train_res.X_test, shap_vals, dataset.feature_to_group)
    assert len(group_shaps) > 0
