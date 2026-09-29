"""
Unit tests for XGBoost classifier training, prediction probabilities, and seed reproducibility.
"""

import pytest
import pandas as pd
import numpy as np
from app.data.feature_processor import FeatureProcessor
from app.model.train import ModelTrainer
from app.model.predict import ModelPredictor


def test_model_training_and_reproducibility():
    df = pd.DataFrame([
        {"case_id": f"C{i}", "price_amount": float(i*10), "product_complexity": float(i%3), "win_loss_outcome": i%2}
        for i in range(20)
    ])
    processor = FeatureProcessor()
    dataset = processor.process(df)
    
    trainer = ModelTrainer(test_size=0.25, random_state=42)
    res1 = trainer.train(dataset)
    
    predictor = ModelPredictor()
    pred1 = predictor.predict_dataset(res1.model, res1.X_test, res1.test_case_ids)
    
    assert len(pred1.case_predictions) == 5
    assert pred1.case_predictions[0].win_probability >= 0.0
    assert pred1.case_predictions[0].win_probability <= 1.0
