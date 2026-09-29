"""
Hindsight Client abstraction handling interaction with Hindsight Memory.
Supports official Hindsight API HTTP integration as well as local mock adapter.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
from pathlib import Path
import httpx
from app.models.memory import HindsightMemoryPayload
from config.settings import settings

logger = logging.getLogger(__name__)


class HindsightClient:
    """
    Unified client for Hindsight Memory system.
    Supports retain(), recall(), get_document(), and health_check().
    Logs with [HINDSIGHT] RETAIN and [HINDSIGHT] RECALL tags.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        use_local_mock: Optional[bool] = None,
        storage_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or settings.hindsight_api_key
        self.api_url = api_url or settings.hindsight_api_url
        self.use_local_mock = settings.use_local_hindsight_mock if use_local_mock is None else use_local_mock
        self.storage_dir = storage_dir or (settings.data_dir / "hindsight_store")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._memory_store: Dict[str, HindsightMemoryPayload] = {}
        self._load_local_store()

    def retain(
        self,
        payload_or_doc_id: Union[HindsightMemoryPayload, str],
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retains a verified memory payload in Hindsight with a stable document ID.
        """
        if isinstance(payload_or_doc_id, str):
            payload = HindsightMemoryPayload(
                document_id=payload_or_doc_id,
                content=content or "",
                metadata=metadata or {},
            )
        else:
            payload = payload_or_doc_id

        doc_id = payload.document_id
        # Log tag (NEVER log secrets!)
        print(f"[HINDSIGHT] RETAIN -> Doc ID: {doc_id} | Length: {len(payload.content)} chars")

        # Always update local store for fast fallback / audit
        self._memory_store[doc_id] = payload
        self._persist_local_store()

        if not self.use_local_mock and self.api_key:
            try:
                url = f"{self.api_url.rstrip('/')}/v1/memories"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                body = payload.to_dict()
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    return {"status": "success", "document_id": doc_id, "mode": "remote_api"}
            except Exception as e:
                print(f"[HINDSIGHT] ERROR -> Remote retain failed: {str(e)}. Preserved in local store.")
                return {"status": "success", "document_id": doc_id, "mode": "local_fallback"}

        return {
            "status": "success",
            "document_id": doc_id,
            "action": "retained",
            "mode": "local_mock",
        }

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recalls matching memories given a natural language query.
        """
        print(f"[HINDSIGHT] RECALL -> Query: '{query}' | Top-K: {top_k}")

        if not self.use_local_mock and self.api_key:
            try:
                url = f"{self.api_url.rstrip('/')}/v1/recall"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                body = {"query": query, "top_k": top_k}
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("memories", [])
            except Exception as e:
                print(f"[HINDSIGHT] ERROR -> Remote recall failed: {str(e)}. Using local recall fallback.")

        # Local mock recall algorithm
        results = []
        q_lower = query.lower()

        STOP_WORDS = {"what", "happened", "with", "when", "where", "have", "that", "this", "from", "they", "them", "some", "were", "been", "would", "could", "should", "does", "done", "doing", "before", "after", "about", "above", "below", "there", "their", "which"}
        for doc_id, payload in self._memory_store.items():
            content_lower = payload.content.lower()
            query_terms = [t for t in q_lower.split() if len(t) > 2 and t not in STOP_WORDS]
            if not query_terms:
                score = 0.0
            else:
                matches = sum(1 for t in query_terms if t in content_lower)
                score = matches / len(query_terms)

            if score > 0.0:
                results.append({
                    "document_id": doc_id,
                    "score": round(score, 2),
                    "content": payload.content,
                    "metadata": payload.metadata,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_document(self, document_id: str) -> Optional[HindsightMemoryPayload]:
        return self._memory_store.get(document_id)

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "mode": "local_mock" if self.use_local_mock else "remote_api",
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
