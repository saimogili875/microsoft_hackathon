"""
Model Package for XGBoost Win/Loss Classification.
Handles model training, inference prediction, reproducible random state, and evaluation metrics.
"""

from app.model.xgboost_model import DealWinLossModel
from app.model.train import ModelTrainer, TrainingResult
from app.model.predict import ModelPredictor, PredictionResult
from app.model.evaluation import ModelEvaluator, PerformanceMetrics

__all__ = [
    "DealWinLossModel",
    "ModelTrainer",
    "TrainingResult",
    "ModelPredictor",
    "PredictionResult",
    "ModelEvaluator",
    "PerformanceMetrics",
]
