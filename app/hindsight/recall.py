"""
Hindsight Recall Service querying persistent Hindsight deal memories for strategic insight.
"""

from typing import List, Dict, Any, Optional
from app.hindsight.client import HindsightClient


class HindsightRecallService:
    """
    Recalls strategic deal memories from Hindsight memory bank.
    """

    def __init__(self, hindsight_client: Optional[HindsightClient] = None):
        self.client = hindsight_client or HindsightClient()

    def recall_deal_experiences(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.client.recall(query, top_k=top_k)

    def recall_experiences(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.client.recall(query, top_k=top_k)

    def recall_episodes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.client.recall(query, top_k=top_k)


# Alias for backward compatibility
MemoryRecallService = HindsightRecallService
