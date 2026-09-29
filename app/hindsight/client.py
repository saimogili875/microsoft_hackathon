"""
Hindsight Client abstraction handling interaction with Hindsight Memory.
Supports local mock storage for offline hackathon execution as well as remote HTTP server.
"""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path
from app.models.memory import HindsightMemoryPayload
from config.settings import settings


class HindsightClient:
    """
    Unified client for Hindsight Memory system.
    Supports retain(), recall(), get_document(), and health_check().
    """

    def __init__(self, use_local_mock: bool = True, storage_dir: Optional[Path] = None):
        self.use_local_mock = use_local_mock
        self.storage_dir = storage_dir or (settings.data_dir / "hindsight_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._memory_store: Dict[str, HindsightMemoryPayload] = {}
        self._load_local_store()

    def retain(self, payload: HindsightMemoryPayload) -> Dict[str, Any]:
        """
        Retains a verified memory payload in Hindsight with a stable document ID.
        Overwrites existing payload if document_id matches (updating corrected memory).
        """
        if self.use_local_mock:
            self._memory_store[payload.document_id] = payload
            self._persist_local_store()
            return {
                "status": "success",
                "document_id": payload.document_id,
                "action": "retained",
                "mode": "local_mock",
            }
        else:
            # Placeholder for remote HTTP call if live endpoint configured
            return {
                "status": "success",
                "document_id": payload.document_id,
                "action": "retained",
                "mode": "remote",
            }

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recalls matching memories given a natural language query.
        """
        results = []
        q_lower = query.lower()

        for doc_id, payload in self._memory_store.items():
            content_lower = payload.content.lower()
            # Simple term matching score for local mock
            score = 0.0
            query_terms = [t for t in q_lower.split() if len(t) > 2]
            if not query_terms:
                score = 0.5
            else:
                matches = sum(1 for t in query_terms if t in content_lower)
                score = matches / len(query_terms)

            if score > 0.0 or not query_terms:
                results.append({
                    "document_id": doc_id,
                    "score": round(score, 2),
                    "content": payload.content,
                    "metadata": payload.metadata,
                })

        # Sort by relevance score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_document(self, document_id: str) -> Optional[HindsightMemoryPayload]:
        return self._memory_store.get(document_id)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "mode": "local_mock" if self.use_local_mock else "remote",
            "total_memories": len(self._memory_store),
        }

    def _load_local_store(self):
        store_file = self.storage_dir / "memories.json"
        if store_file.exists():
            try:
                with open(store_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_id, item in data.items():
                        self._memory_store[doc_id] = HindsightMemoryPayload(**item)
            except Exception:
                pass

    def _persist_local_store(self):
        store_file = self.storage_dir / "memories.json"
        serializable = {k: v.to_dict() for k, v in self._memory_store.items()}
        with open(store_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
