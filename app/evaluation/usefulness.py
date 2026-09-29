"""
Usefulness Evaluator module comparing "Prediction Alone" vs "Prediction + Explanation" ratings.
Computes empirical mean ratings, difference, and sample statistics.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from app.experts.expert_predictions import ExpertEvaluationRecord


@dataclass
class UsefulnessComparisonResult:
    total_evaluations: int
    num_experts: int
    mean_rating_prediction_alone: float
    mean_rating_prediction_and_explanation: float
    difference_mean_usefulness: float
    hypothesis_supported: bool  # True if Explanation condition rated higher
    summary_text: str


class UsefulnessEvaluator:
    """
    Evaluates the research hypothesis comparing Prediction Alone vs. Prediction + Explanation usefulness.
    """

    def evaluate_records(self, records: List[ExpertEvaluationRecord]) -> UsefulnessComparisonResult:
        if not records:
            return UsefulnessComparisonResult(
                total_evaluations=0,
                num_experts=0,
                mean_rating_prediction_alone=0.0,
                mean_rating_prediction_and_explanation=0.0,
                difference_mean_usefulness=0.0,
                hypothesis_supported=False,
                summary_text="No expert evaluation records available.",
            )

        alone_ratings = [r.rating_prediction_alone for r in records]
        expl_ratings = [r.rating_prediction_and_explanation for r in records]
        experts = set([r.expert_id for r in records])

        mean_alone = float(np.mean(alone_ratings))
        mean_expl = float(np.mean(expl_ratings))
        diff = round(mean_expl - mean_alone, 4)
        hypothesis_supported = (diff > 0)

        summary = (
            f"Evaluated {len(records)} case ratings from {len(experts)} experts. "
            f"Prediction Alone Mean Rating: {mean_alone:.2f}/7.00 | "
            f"Prediction + Explanation Mean Rating: {mean_expl:.2f}/7.00. "
            f"Usefulness Lift: +{diff:.2f} points "
            f"({'Hypothesis Supported' if hypothesis_supported else 'Hypothesis Not Supported'})."
        )

        return UsefulnessComparisonResult(
            total_evaluations=len(records),
            num_experts=len(experts),
            mean_rating_prediction_alone=round(mean_alone, 4),
            mean_rating_prediction_and_explanation=round(mean_expl, 4),
            difference_mean_usefulness=diff,
            hypothesis_supported=hypothesis_supported,
            summary_text=summary,
        )
