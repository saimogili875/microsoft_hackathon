"""
Model Predictor module running inference predictions on single cases or test batches.
"""

from typing import List, Dict, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from app.model.xgboost_model import DealWinLossModel


@dataclass
class CasePrediction:
    case_id: str
    predicted_class: int  # 1 = WIN, 0 = LOSS
    predicted_label: str  # "WIN" or "LOSS"
    win_probability: float
    loss_probability: float
    confidence: float


@dataclass
class PredictionResult:
    case_predictions: List[CasePrediction]
    y_pred: np.ndarray
    y_proba: np.ndarray


class ModelPredictor:
    """
    Executes inference predictions and formats output with prediction confidence.
    """

    def predict_dataset(self, model: DealWinLossModel, X: pd.DataFrame, case_ids: pd.Series) -> PredictionResult:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        case_preds = []
        for idx, (_, row) in enumerate(X.iterrows()):
            c_id = str(case_ids.iloc[idx]) if idx < len(case_ids) else f"CASE-{idx}"
            pred_class = int(y_pred[idx])
            loss_prob = float(y_proba[idx][0])
            win_prob = float(y_proba[idx][1])
            confidence = float(max(loss_prob, win_prob))

            case_preds.append(
                CasePrediction(
                    case_id=c_id,
                    predicted_class=pred_class,
                    predicted_label="WIN" if pred_class == 1 else "LOSS",
                    win_probability=round(win_prob, 4),
                    loss_probability=round(loss_prob, 4),
                    confidence=round(confidence, 4),
                )
            )

        return PredictionResult(
            case_predictions=case_preds,
            y_pred=y_pred,
            y_proba=y_proba,
        )
