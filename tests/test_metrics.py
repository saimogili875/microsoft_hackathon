"""
Unit tests for exact Balanced Accuracy and Confusion Matrix calculation.
"""

import pytest
import numpy as np
from app.evaluation.balanced_accuracy import BalancedAccuracyCalculator
from app.evaluation.confusion_matrix import ConfusionMatrixGenerator
from app.model.evaluation import ModelEvaluator


def test_balanced_accuracy_formula():
    # TP=10, FN=2 (Win recall = 10/12 = 0.8333)
    # FP=1, TN=9 (Loss recall = 9/10 = 0.9000)
    # Balanced Accuracy = 0.5 * (0.83333 + 0.90000) = 0.8667
    bal_acc = BalancedAccuracyCalculator.calculate(tp=10, fn=2, fp=1, tn=9)
    assert round(bal_acc, 4) == 0.8667


def test_confusion_matrix_generator():
    y_true = np.array([1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1])
    # TP=2, FN=1, FP=1, TN=1
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    assert metrics.confusion_matrix["TP"] == 2
    assert metrics.confusion_matrix["FN"] == 1
    assert metrics.confusion_matrix["FP"] == 1
    assert metrics.confusion_matrix["TN"] == 1
    assert round(metrics.win_recall, 4) == 0.6667
    assert round(metrics.loss_recall, 4) == 0.5000
    assert round(metrics.balanced_accuracy, 4) == 0.5833
