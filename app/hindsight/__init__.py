"""
Hindsight Integration Package for B2B Deal Intelligence Research Engine.
Converts verified model predictions, SHAP explanations, and expert comparison insights into persistent memory.
"""

from app.hindsight.memory_schema import HindsightDealMemory
from app.hindsight.retain import HindsightRetainService
from app.hindsight.recall import HindsightRecallService

__all__ = [
    "HindsightDealMemory",
    "HindsightRetainService",
    "HindsightRecallService",
]
