"""
Model Trainer module executing reproducible stratified train/test split and XGBoost model fitting.
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from app.model.xgboost_model import DealWinLossModel
from app.data.feature_processor import ProcessedDataset


@dataclass
class TrainingResult:
    model: DealWinLossModel
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_case_ids: pd.Series
    test_case_ids: pd.Series
    feature_names: list
    feature_to_group: dict


class ModelTrainer:
    """
    Executes reproducible train/test split and fits DealWinLossModel.
    """

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def train(self, dataset: ProcessedDataset) -> TrainingResult:
        can_stratify = (dataset.y.nunique() > 1) and (dataset.y.value_counts().min() >= 2)
        X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
            dataset.X,
            dataset.y,
            dataset.case_ids,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=dataset.y if can_stratify else None,
        )

        model = DealWinLossModel(random_state=self.random_state)
        model.fit(X_train, y_train)

        return TrainingResult(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_case_ids=train_ids,
            test_case_ids=test_ids,
            feature_names=dataset.feature_names,
            feature_to_group=dataset.feature_to_group,
        )
