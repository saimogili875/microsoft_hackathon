"""
Memory Manager handling memory updates, replacement of corrected episodes,
and temporal versioning of market facts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from app.models.episode import Episode
from app.models.market_info import MarketInfoRecord
from app.hindsight.client import HindsightClient
from app.hindsight.retain import retain_episode, retain_market_info


class MemoryManager:
    """
    Manager coordinating Hindsight retention, document updates via stable IDs,
    and market intelligence temporal state transitions.
    """

    def __init__(self, client: HindsightClient):
        self.client = client
        self.market_records: Dict[str, MarketInfoRecord] = {}

    def retain_verified_episode(self, episode: Episode) -> Dict[str, Any]:
        """
        Retains or updates a confirmed/corrected episode.
        Because doc_id is 'episode:<episode_id>', re-retaining automatically replaces/updates
        the memory in Hindsight safely.
        """
        if episode.verification_status not in ["confirmed", "corrected"]:
            raise ValueError(f"Episode {episode.episode_id} is not verified. Status: {episode.verification_status}")

        payload = retain_episode(episode, self.client)
        return {
            "status": "retained",
            "document_id": payload.document_id,
            "verification_status": episode.verification_status,
        }

    def process_new_market_evidence(
        self,
        competitor: str,
        topic: str,
        statement: str,
        pricing_context: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        reviewer_role: str = "market_analyst",
    ) -> MarketInfoRecord:
        """
        Processes new market evidence without destroying historical memory.
        1. Checks for active old memories for the same competitor + topic.
        2. Marks old memory as 'superseded' with valid_until = now.
        3. Creates new active market record with valid_from = now.
        """
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        comp_key = competitor.lower().strip()

        # Update existing active market memory for this competitor if present
        for m_id, existing in self.market_records.items():
            if existing.competitor.lower().strip() == comp_key and existing.topic.lower().strip() == topic.lower().strip():
                if existing.status == "active":
                    existing.status = "superseded"
                    existing.valid_until = now_date
                    # Re-retain updated historical memory
                    retain_market_info(existing, self.client)

        new_record = MarketInfoRecord(
            competitor=competitor,
            topic=topic,
            statement=statement,
            pricing_context=pricing_context,
            source_ids=source_ids or [],
            valid_from=now_date,
            valid_until=None,
            status="active",
            verified_by_role=reviewer_role,
            verified_at=datetime.now(timezone.utc).isoformat(),
        )

        self.market_records[new_record.market_id] = new_record
        retain_market_info(new_record, self.client)
        return new_record
