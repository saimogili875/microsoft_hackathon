"""
Balanced Accuracy Calculator calculating the exact research primary metric.
"""

from typing import Dict, Any
import numpy as np


class BalancedAccuracyCalculator:
    """
    Computes Balanced Accuracy = 0.5 * (TP / (TP + FN) + TN / (TN + FP)).
    """

    @staticmethod
    def calculate(tp: int, fn: int, fp: int, tn: int) -> float:
        win_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        loss_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return round(0.5 * (win_recall + loss_recall), 4)
