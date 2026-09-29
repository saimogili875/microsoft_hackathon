"""
Sales Expert Questionnaire module implementing research Q1-Q5 questions,
response validation, and structured expert response storage.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class QuestionResponse:
    expert_id: str
    case_id: str
    question_id: str  # Q1, Q2, Q3, Q4, Q5
    question_text: str
    response: Any
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    explanation: Optional[str] = None


class ExpertQuestionnaire:
    """
    Research-inspired Sales Expert Questionnaire (Q1 to Q5) manager.
    """

    QUESTIONS = {
        "Q1": "Experience in B2B sales of configurable products (0-2 years, 3-10 years, 11+ years)",
        "Q2": "Experience using AI tools (1=Not at all to 5=Daily)",
        "Q3": "Previous experience predicting winning probability in CRM",
        "Q4": "Is it possible to determine winning probability based on presented information? (Yes, Maybe, No, I don't know)",
        "Q5": "How would you assess the outcome of this offer? (WIN, LOSS, Uncertain)",
    }

    VALID_Q1 = ["0-2 years", "3-10 years", "11+ years"]
    VALID_Q4 = ["Yes", "Maybe", "No", "I don't know"]
    VALID_Q5 = ["WIN", "LOSS", "Uncertain"]

    def create_response(
        self,
        expert_id: str,
        case_id: str,
        question_id: str,
        response: Any,
        explanation: Optional[str] = None,
    ) -> QuestionResponse:

        if question_id not in self.QUESTIONS:
            raise ValueError(f"Invalid question_id '{question_id}'. Must be one of {list(self.QUESTIONS.keys())}")

        # Validate response options
        if question_id == "Q1" and response not in self.VALID_Q1:
            raise ValueError(f"Invalid Q1 response '{response}'. Expected one of {self.VALID_Q1}")
        if question_id == "Q2" and not (isinstance(response, int) and 1 <= response <= 5):
            raise ValueError(f"Invalid Q2 rating '{response}'. Expected integer scale 1 to 5.")
        if question_id == "Q4" and response not in self.VALID_Q4:
            raise ValueError(f"Invalid Q4 response '{response}'. Expected one of {self.VALID_Q4}")
        if question_id == "Q5" and response not in self.VALID_Q5:
            raise ValueError(f"Invalid Q5 response '{response}'. Expected one of {self.VALID_Q5}")

        return QuestionResponse(
            expert_id=expert_id,
            case_id=case_id,
            question_id=question_id,
            question_text=self.QUESTIONS[question_id],
            response=response,
            explanation=explanation,
        )
