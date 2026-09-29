"""
Manual Salesperson Chat Service implementing Mode 02 Chat using Hindsight RECALL + Groq Reasoning.
"""

from typing import Any, Dict, List, Optional
from app.hindsight.client import HindsightClient
from app.hindsight.recall import MemoryRecallService
from app.services.groq_service import GroqService
from app.services.traceability import TraceabilityService
from app.verification.verification_service import VerificationService


class SalespersonChatService:
    """
    Handles Mode 02 Manual Salesperson Chat Interface / API.
    Enforces Hindsight RECALL -> Groq Reasoning workflow.
    Never searches raw JSON files directly for the final response.
    """

    def __init__(
        self,
        hindsight_client: HindsightClient,
        groq_service: GroqService,
        verification_service: VerificationService,
    ):
        self.hindsight_client = hindsight_client
        self.recall_service = MemoryRecallService(hindsight_client)
        self.groq_service = groq_service
        self.verification_service = verification_service
        self.traceability_service = TraceabilityService(hindsight_client, verification_service)

    def answer_query(self, user_query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes manual chat workflow:
        User Question -> Hindsight RECALL -> Filter Relevant Memories -> Groq Reasoning -> Answer.
        """
        # 1. Hindsight RECALL (Primary Memory Retrieval)
        recalled_memories = self.recall_service.recall_experiences(user_query, top_k=5)

        # 2. Filter unrelated memories (only pass memories with score > 0.0 to Groq)
        filtered_memories = [m for m in recalled_memories if m.get("score", 0) > 0.0]

        # 3. Retrieve raw source excerpts for traceability if memory exists
        source_excerpts = []
        for mem in filtered_memories:
            doc_id = mem.get("document_id", "")
            if doc_id.startswith("episode:"):
                ep_id = doc_id.split("episode:", 1)[1]
                trace_node = self.traceability_service.trace_by_episode_id(ep_id)
                if "source_ids" in trace_node:
                    source_excerpts.append({
                        "episode_id": ep_id,
                        "source_ids": trace_node["source_ids"],
                        "traceability": trace_node,
                    })

        # 4. Prepare structured context payload for Groq
        groq_context = {
            "task": "Manual Salesperson Chat Answer",
            "user_query": user_query,
            "client_context": client_context,
            "hindsight_memories": filtered_memories,
            "relevant_source_excerpts": source_excerpts,
            "total_memories_found": len(filtered_memories),
        }

        # 5. Send to Groq Reasoning Layer
        answer = self.groq_service.generate_response(groq_context)

        return {
            "user_query": user_query,
            "answer": answer,
            "hindsight_memories_used": filtered_memories,
            "source_excerpts": source_excerpts,
        }
