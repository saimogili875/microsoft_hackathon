"""
Expert Predictions Manager loading independent Sales Expert predictions and Q1-Q5 evaluation records.
Guarantees expert predictions are NEVER fed as input features into the XGBoost model.
Preserves individual expert responses without averaging.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ExpertEvaluationRecord:
    case_id: str
    expert_id: str
    q1_b2b_experience: str
    q2_ai_tool_usage: int
    q3_predicting_experience: str
    q4_information_sufficiency: str
    q5_assessed_outcome: str  # WIN, LOSS, Uncertain
    expert_predicted_prob: float
    expert_confidence: float
    expert_important_factors: List[str]
    expert_explanation: str
    rating_prediction_alone: float
    rating_prediction_and_explanation: float


class ExpertPredictionManager:
    """
    Manages loading and querying of Sales Expert evaluation records.
    """

    def __init__(self):
        self._records: Dict[str, List[ExpertEvaluationRecord]] = {}  # case_id -> list of records

    def load_file(self, file_path: Path) -> List[ExpertEvaluationRecord]:
        if not file_path.exists():
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        records = []
        for item in raw_list:
            rec = ExpertEvaluationRecord(
                case_id=item["case_id"],
                expert_id=item["expert_id"],
                q1_b2b_experience=item.get("q1_b2b_experience", "3-10 years"),
                q2_ai_tool_usage=int(item.get("q2_ai_tool_usage", 3)),
                q3_predicting_experience=item.get("q3_predicting_experience", "Regularly predicts deal win probability"),
                q4_information_sufficiency=item.get("q4_information_sufficiency", "Yes"),
                q5_assessed_outcome=item.get("q5_assessed_outcome", "WIN"),
                expert_predicted_prob=float(item.get("expert_predicted_prob", 0.5)),
                expert_confidence=float(item.get("expert_confidence", 0.8)),
                expert_important_factors=item.get("expert_important_factors", []),
                expert_explanation=item.get("expert_explanation", ""),
                rating_prediction_alone=float(item.get("rating_prediction_alone", 4.0)),
                rating_prediction_and_explanation=float(item.get("rating_prediction_and_explanation", 6.0)),
            )
            records.append(rec)

            if rec.case_id not in self._records:
                self._records[rec.case_id] = []
            self._records[rec.case_id].append(rec)

        return records

    def get_expert_evaluations_for_case(self, case_id: str) -> List[ExpertEvaluationRecord]:
        return self._records.get(case_id, [])

    def get_all_records(self) -> List[ExpertEvaluationRecord]:
        all_recs = []
        for case_recs in self._records.values():
            all_recs.extend(case_recs)
        return all_recs
