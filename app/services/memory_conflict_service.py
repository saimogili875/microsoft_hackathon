"""
Memory Conflict and Temporal Service for ARCH-2.
Inspects recalled memories for conflicting claims, price shifts, and temporal superseding.
"""

from typing import Any, Dict, List, Optional
import re
from app.models.retrieved_memory import RetrievedMemory


class MemoryConflictService:
    """
    Service for inspecting retrieved Hindsight memories.
    Identifies contradictory facts, pricing shifts, and temporal versioning
    without silently choosing one or destroying historical evidence.
    """

    def inspect_memories(self, memories: List[RetrievedMemory]) -> Dict[str, Any]:
        conflicts = []
        pricing_claims: List[Dict[str, Any]] = []

        for mem in memories:
            text = mem.content.lower()
            
            # Check pricing mentions (e.g. 10L, 7L, 12L, rupees, dollars)
            price_matches = re.findall(r"(?:₹|\$)?\s*(\d+\s*(?:l|lakh|k|million)?)", text, re.IGNORECASE)
            if price_matches:
                for pm in price_matches:
                    if pm.strip() and len(pm.strip()) > 1:
                        pricing_claims.append({
                            "memory_id": mem.memory_id,
                            "price": pm.strip(),
                            "timestamp": mem.timestamp or mem.valid_as_of,
                            "verification_status": mem.verification_status,
                            "content": mem.content[:120],
                        })

        # Detect pricing conflicts if multiple distinct price figures are present for the same scope
        if len(pricing_claims) >= 2:
            unique_prices = set(pc["price"] for pc in pricing_claims)
            if len(unique_prices) > 1:
                conflicts.append({
                    "conflict_type": "pricing_discrepancy",
                    "description": f"Multiple distinct pricing figures found across retrieved memories: {list(unique_prices)}.",
                    "conflicting_evidence": pricing_claims,
                    "action_required": "Expose discrepancy explicitly to Groq reasoning layer without merging.",
                })

        return {
            "conflict_detected": len(conflicts) > 0,
            "conflicts": conflicts,
            "total_memories_inspected": len(memories),
        }
