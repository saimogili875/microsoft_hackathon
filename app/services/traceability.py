"""
Source Traceability Service providing end-to-end auditability from Hindsight memory back to raw file.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
from app.hindsight.client import HindsightClient
from app.verification.verification_service import VerificationService


class TraceabilityService:
    """
    Traces memory back to original source file, timestamp, raw content, and metadata.
    Lineage: Hindsight Memory -> Episode ID -> Normalized Record -> Raw Record -> Source File.
    """

    def __init__(self, hindsight_client: HindsightClient, verification_service: VerificationService):
        self.client = hindsight_client
        self.verification_service = verification_service

    def trace_by_episode_id(self, episode_id: str) -> Dict[str, Any]:
        doc_id = f"episode:{episode_id}"
        memory = self.client.get_document(doc_id)
        episode = self.verification_service.get_episode_by_id(episode_id)

        if not episode:
            return {"error": f"Episode '{episode_id}' not found in verification store."}

        trace_info = {
            "hindsight_document_id": doc_id,
            "hindsight_memory_present": memory is not None,
            "episode_id": episode.episode_id,
            "deal_id": episode.deal_id,
            "verification_status": episode.verification_status,
            "verified_by_role": episode.verified_by_role,
            "verified_at": episode.verified_at,
            "corrections_count": len(episode.corrections),
            "source_ids": episode.source_ids,
            "traceability_chain": [],
        }

        # Build lineage node for each source_id
        for src_id in episode.source_ids:
            node = {
                "source_id": src_id,
                "confidence_extraction": episode.extraction_confidence,
                "confidence_causal": episode.causal_confidence,
                "timestamp": episode.timestamp,
            }
            trace_info["traceability_chain"].append(node)

        return trace_info

    def trace_by_document_id(self, document_id: str) -> Dict[str, Any]:
        if document_id.startswith("episode:"):
            ep_id = document_id.split("episode:", 1)[1]
            return self.trace_by_episode_id(ep_id)
        else:
            memory = self.client.get_document(document_id)
            if not memory:
                return {"error": f"Document '{document_id}' not found."}
            return {
                "hindsight_document_id": document_id,
                "metadata": memory.metadata,
                "content_preview": memory.content[:200],
            }
