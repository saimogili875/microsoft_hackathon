"""
XGBoost Classifier wrapper for WIN vs LOSS deal prediction.
Ensures reproducible random seed and safe fallback.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("model.xgboost")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier


class DealWinLossModel:
    """
    XGBoost Classifier wrapper for pre-offer B2B deal Win/Loss classification.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.is_fitted = False
        self.feature_names: List[str] = []

        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            self.model_type = "XGBoost"
        else:
            logger.info("XGBoost package not found. Using sklearn HistGradientBoostingClassifier fallback.")
            self.model = HistGradientBoostingClassifier(
                random_state=self.random_state,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
            )
            self.model_type = "HistGradientBoosting"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting probabilities.")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, [float(imp) for imp in importances]))
        return {feat: 1.0 / len(self.feature_names) for feat in self.feature_names}
