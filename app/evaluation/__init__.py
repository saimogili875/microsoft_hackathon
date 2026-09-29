"""
Evaluation Package for Balanced Accuracy, Confusion Matrix, Prediction Alone vs Prediction + Explanation usefulness,
and master research report generation.
"""

from app.evaluation.balanced_accuracy import BalancedAccuracyCalculator
from app.evaluation.confusion_matrix import ConfusionMatrixGenerator
from app.evaluation.usefulness import UsefulnessEvaluator, UsefulnessComparisonResult
from app.evaluation.research_report import ResearchReportGenerator

__all__ = [
    "BalancedAccuracyCalculator",
    "ConfusionMatrixGenerator",
    "UsefulnessEvaluator",
    "UsefulnessComparisonResult",
    "ResearchReportGenerator",
]
