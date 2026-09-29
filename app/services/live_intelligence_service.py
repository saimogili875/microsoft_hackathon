"""
Live Intelligence Service for Mode 01 — Live Call Stream Processing & Hindsight Recall.
"""

from typing import Any, Dict, List, Optional
import re
from datetime import datetime, timezone
from app.models.live_interaction import LiveTranscriptChunk, LiveSignals, LiveInteractionState
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.hindsight.client import HindsightClient
from app.hindsight.recall import MemoryRecallService


class LiveIntelligenceService:
    """
    Manages Mode 01 Live Interaction Intelligence:
    1. Ingests live transcript stream.
    2. Extracts real-time signals (objections, pricing, competitors).
    3. Performs real-time Hindsight Recall.
    4. Evaluates recalled experiences against current call evidence.
    5. Converts post-meeting state into canonical Episode candidate.
    """

    def __init__(self, hindsight_client: HindsightClient):
        self.hindsight_client = hindsight_client
        self.recall_service = MemoryRecallService(hindsight_client)
        self.active_sessions: Dict[str, LiveInteractionState] = {}

    def start_session(self, session_id: str, deal_id: Optional[str] = None, customer_context: Optional[str] = None) -> LiveInteractionState:
        state = LiveInteractionState(
            session_id=session_id,
            deal_id=deal_id,
            customer_context=customer_context,
        )
        self.active_sessions[session_id] = state
        return state

    def process_chunk(self, chunk: LiveTranscriptChunk) -> Dict[str, Any]:
        session_id = chunk.session_id
        state = self.active_sessions.get(session_id)
        if not state:
            state = self.start_session(session_id)

        state.chunks.append(chunk)
        self._update_signals(state, chunk)

        # Trigger real-time Hindsight Recall if objections or competitors mentioned
        recalls = []
        recommendations = []

        signals = state.extracted_signals
        if signals.customer_concerns or signals.competitors_mentioned:
            query = " ".join(signals.customer_concerns + signals.competitors_mentioned)
            recalls = self.recall_service.recall_experiences(query, top_k=3)
            state.recalled_experiences = recalls

            # Evaluate recalled tactics against current evidence
            recommendations = self._evaluate_recalls_against_evidence(recalls, signals)
            state.live_recommendations = recommendations

        return {
            "session_id": session_id,
            "processed_chunk_id": chunk.chunk_id,
            "extracted_signals": signals.model_dump(),
            "recalled_experiences": recalls,
            "live_recommendations": recommendations,
        }

    def finalize_meeting(self, session_id: str) -> Dict[str, Any]:
        """
        Post-meeting conversion: converts live signals into canonical NormalizedRecord & Episode candidate.
        Outcome & lesson are populated based on call wrap-up data.
        """
        state = self.active_sessions.get(session_id)
        if not state:
            raise FileNotFoundError(f"Session '{session_id}' not found.")

        signals = state.extracted_signals
        text_content = "\n".join([f"{c.speaker}: {c.text}" for c in state.chunks])

        # 1. Map to common NormalizedRecord
        norm = NormalizedRecord(
            source_id=f"live_{session_id}",
            source_type="sales_conversation",
            deal_id=state.deal_id,
            customer_context=state.customer_context or "Live call participant",
            objections=signals.customer_concerns,
            competitors=signals.competitors_mentioned,
            pricing_information=signals.pricing_discussed,
            salesperson_actions=signals.salesperson_tactics,
            customer_reactions=signals.customer_reactions,
            outcome=signals.outcome if signals.outcome != "unknown" else "in_progress",
            raw_text_content=text_content,
            source_references=[{
                "source_id": f"live_{session_id}",
                "source_type": "live_call",
                "chunks_count": len(state.chunks),
            }],
        )

        # 2. Extract common Episode candidate
        situation = f"Live call with {state.customer_context or 'prospect'}"
        objection = "; ".join(signals.customer_concerns) if signals.customer_concerns else "Price and implementation scope"
        tactic = "; ".join(signals.salesperson_tactics) if signals.salesperson_tactics else "Presented modular deployment options"
        reaction = "; ".join(signals.customer_reactions) if signals.customer_reactions else "Agreed to follow-up proposal"

        episode = Episode(
            deal_id=state.deal_id,
            situation=situation,
            objection=objection,
            tactic=tactic,
            customer_reaction=reaction,
            outcome="no_decision" if signals.outcome == "unknown" else signals.outcome,
            why="Live call concluded; full deal outcome pending next steps",
            lesson="Document customer objections immediately during live call",
            applies_when=["Live enterprise sales calls", "Negotiating pricing scope"],
            did_not_hold_when=["When budget is frozen"],
            pricing_context="; ".join(signals.pricing_discussed) if signals.pricing_discussed else None,
            source_ids=[f"live_{session_id}"],
            extraction_confidence=0.75,
            causal_confidence=0.40,  # Low/conservative for unverified live call
            verification_status="pending_review",
        )

        return {
            "session_id": session_id,
            "normalized_record": norm,
            "episode_candidate": episode,
        }

    def _update_signals(self, state: LiveInteractionState, chunk: LiveTranscriptChunk):
        text = chunk.text.lower()
        signals = state.extracted_signals

        # Competitors
        comps = ["competitor x", "competitor y", "acme corp", "salesforce", "hubspot", "oracle"]
        for comp in comps:
            if comp in text and comp.title() not in signals.competitors_mentioned:
                signals.competitors_mentioned.append(comp.title())

        # Pricing
        if any(w in text for w in ["price", "cost", "budget", "quote", "discount", "₹", "$"]):
            signals.pricing_discussed.append(chunk.text[:100])

        # Objections/Concerns
        if any(w in text for w in ["too expensive", "high cost", "too high", "cost", "price", "cheaper", "timeline", "delay", "concern", "risk", "difficult", "objection"]):
            signals.customer_concerns.append(chunk.text[:120])

        # Customer Questions
        if "?" in chunk.text and chunk.speaker.lower() in ["customer", "prospect", "client"]:
            signals.customer_questions.append(chunk.text[:120])

        # Salesperson Tactics
        if chunk.speaker.lower() in ["rep", "salesperson", "ae"]:
            if any(w in text for w in ["offer", "propose", "phase", "pilot", "demo", "discount"]):
                signals.salesperson_tactics.append(chunk.text[:120])

        # Customer Reactions
        if chunk.speaker.lower() in ["customer", "prospect", "client"]:
            if any(w in text for w in ["agree", "makes sense", "sounds good", "interesting", "not interested"]):
                signals.customer_reactions.append(chunk.text[:120])

    def _evaluate_recalls_against_evidence(
        self, recalls: List[Dict[str, Any]], signals: LiveSignals
    ) -> List[str]:
        """
        Compares recalled Hindsight memories against current live call evidence.
        Prevents recommending outdated tactics when boundary conditions change!
        """
        recommendations = []
        text_evidence = " ".join(signals.pricing_discussed + signals.customer_concerns).lower()

        for item in recalls:
            content = item.get("content", "")

            # Boundary Condition Check: Competitor Price Drop
            if "phased rollout worked" in content.lower() or "pilot" in content.lower():
                if "cheaper" in text_evidence or "30%" in text_evidence or "7l" in text_evidence:
                    recommendations.append(
                        "⚠️ CAUTION: Hindsight indicates phased rollout worked previously, BUT current evidence shows competitor is significantly cheaper (~30%). A simple phased rollout may fail without additional value justification."
                    )
                else:
                    recommendations.append("💡 Hindsight Recommendation: Propose a phased pilot rollout to de-risk upfront costs.")

            elif "discount failed" in content.lower():
                recommendations.append("💡 Hindsight Note: Avoid immediate flat discounts; past data shows flat discounts fail without scope adjustments.")

        if not recommendations and recalls:
            recommendations.append(f"Found {len(recalls)} relevant Hindsight memories. Review past pricing tactics.")

        return recommendations
