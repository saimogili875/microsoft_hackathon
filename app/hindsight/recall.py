"""
Recall module querying Hindsight memory.
"""

from typing import Any, Dict, List, Optional
from app.hindsight.client import HindsightClient


class MemoryRecallService:
    """
    Service for querying retained memories in Hindsight.
    """

    def __init__(self, client: HindsightClient):
        self.client = client

    def recall_experiences(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recalls sales experience memories matching user query.
        """
        raw_results = self.client.recall(query, top_k=top_k)
        return raw_results
