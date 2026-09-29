"""
Model Evaluator module calculating standard classification performance metrics.
"""

from typing import Dict, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@dataclass
class PerformanceMetrics:
    balanced_accuracy: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    win_recall: float
    loss_recall: float
    confusion_matrix: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "win_recall": self.win_recall,
            "loss_recall": self.loss_recall,
            "confusion_matrix": self.confusion_matrix,
        }


class ModelEvaluator:
    """
    Computes exact evaluation metrics including Balanced Accuracy and Confusion Matrix.
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> PerformanceMetrics:
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        # cm structure for labels=[1, 0]:
        # [[TP, FN],
        #  [FP, TN]]
        if cm.shape == (2, 2):
            tp = int(cm[0, 0])
            fn = int(cm[0, 1])
            fp = int(cm[1, 0])
            tn = int(cm[1, 1])
        else:
            tp, fn, fp, tn = 0, 0, 0, 0

        win_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        loss_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = 0.5 * (win_recall + loss_recall)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm_dict = {"TP": tp, "FN": fn, "FP": fp, "TN": tn}

        return PerformanceMetrics(
            balanced_accuracy=round(balanced_accuracy, 4),
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            win_recall=round(win_recall, 4),
            loss_recall=round(loss_recall, 4),
            confusion_matrix=cm_dict,
        )
