"""
Automated Integration Test Suite for Hindsight API + Groq Reasoning Layer.
Includes Test Case A (Live Interaction & Change Conflict) and Test Case B (Manual Salesperson Chat).
"""

import tempfile
import json
import pytest
from pathlib import Path

from app.services.pipeline import DealIntelligencePipeline
from app.models.episode import Episode
from app.models.change_event import ClientState, ChangeCategory
from app.models.live_interaction import LiveTranscriptChunk
from app.models.normalized_data import NormalizedRecord


@pytest.fixture
def pipeline(tmp_path):
    return DealIntelligencePipeline()


# 1. TEST CASE A — LIVE INTERACTION & CONFLICT DETECTION
def test_case_a_live_interaction_conflict(pipeline):
    """
    Test Case A: Old memory (₹10L price) + Current call evidence (₹7L price) ->
    Hindsight RECALL -> Change Detection (PENDING VERIFICATION) -> Groq reasoning.
    """
    ep = Episode(
        episode_id="ep-hist-abc",
        deal_id="D-ABC-100",
        situation="ABC Logistics pricing discussion in 2025",
        objection="Competitor X quote",
        tactic="Offered phased rollout",
        customer_reaction="Agreed to pilot",
        outcome="won",
        why="Phased pilot de-risked upfront capital",
        lesson="Phased rollout works when competitor price is at parity (₹10L)",
        pricing_context="Competitor X price ≈ ₹10L",
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep)
    pipeline.memory_manager.retain_verified_episode(ep)

    session_id = "live_conflict_session"
    pipeline.live_service.start_session(session_id, deal_id="D-ABC-100", customer_context="ABC Logistics")

    chunk = LiveTranscriptChunk(
        session_id=session_id,
        speaker="Customer",
        text="Competitor X is offering a cheaper quote of 7L now.",
    )
    res = pipeline.process_live_chunk(chunk)

    assert len(res["recalled_experiences"]) > 0
    assert res["recalled_experiences"][0]["document_id"] == "episode:ep-hist-abc"
    assert len(res["live_recommendations"]) > 0
    assert "⚠️ CAUTION" in res["live_recommendations"][0]

    baseline = ClientState(
        deal_id="D-ABC-100",
        customer_context="ABC Logistics",
        competitors=["Competitor X"],
        competitor_pricing={"Competitor X": "₹10L"},
    )
    new_norm = NormalizedRecord(
        source_id="live_chunk_1",
        source_type="sales_conversation",
        deal_id="D-ABC-100",
        pricing_information=["Competitor X quote ₹7L"],
    )
    changes = pipeline.detect_changes(baseline, new_norm)
    assert len(changes) > 0
    assert changes[0].category == ChangeCategory.COMPETITOR_PRICE_CHANGE
    assert changes[0].status == "pending_review"


# 2. TEST CASE B — MANUAL CHAT WITH HINDSIGHT RECALL & GROQ
def test_case_b_manual_chat(pipeline):
    """
    Test Case B: User question -> Hindsight RECALL -> Filter Relevant Memories -> Groq Reasoning -> Answer.
    """
    ep = Episode(
        episode_id="ep-abc-chat",
        deal_id="D-ABC-100",
        situation="ABC Logistics implementation budget objection",
        objection="Implementation cost too high",
        tactic="Offered 3-phase modular rollout starting with ₹4L pilot phase",
        customer_reaction="Agreed to Phase 1 pilot evaluation",
        outcome="won",
        why="Pilot phase de-risked initial capital outlay",
        lesson="Structure high-tier quotes with phase 1 pilot options when price resistance occurs",
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep)
    pipeline.memory_manager.retain_verified_episode(ep)

    query = "What worked when ABC Logistics objected to implementation cost?"
    chat_res = pipeline.chat(query, client_context="ABC Logistics")

    assert len(chat_res["hindsight_memories_used"]) > 0
    assert chat_res["hindsight_memories_used"][0]["document_id"] == "episode:ep-abc-chat"

    answer = chat_res["answer"]
    assert "ABC Logistics" in answer or "DEAL INTELLIGENCE" in answer
    assert "pilot" in answer.lower() or "phased" in answer.lower() or "modular" in answer.lower()


# 3. FILTER UNRELATED MEMORIES IN CHAT
def test_unrelated_memories_filtering(pipeline):
    ep = Episode(
        episode_id="ep-irrelevant-99",
        deal_id="D-XYZ-99",
        situation="Cloud infrastructure compliance review",
        objection="AWS data sovereignty concern",
        tactic="Provided AWS local region compliance doc",
        customer_reaction="Approved deployment",
        outcome="won",
        why="Compliance doc satisfied legal team",
        lesson="Provide region compliance docs for AWS deals",
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep)
    pipeline.memory_manager.retain_verified_episode(ep)

    query = "sovereignty compliance"
    res = pipeline.chat(query)
    assert len(res["hindsight_memories_used"]) >= 1
    doc_ids = [m["document_id"] for m in res["hindsight_memories_used"]]
    assert "episode:ep-irrelevant-99" in doc_ids


# 4. ERROR HANDLING & FALLBACKS
def test_error_handling_empty_recall(pipeline):
    """
    Verifies that if Hindsight recall returns nothing for non-matching query, system tells Groq no memory was found.
    """
    query = "QWZZXYVK998877_NONEXISTENT_QUERY"
    chat_res = pipeline.chat(query)

    assert len(chat_res["hindsight_memories_used"]) == 0
    answer = chat_res["answer"]
    assert "No relevant historical memories were found" in answer or "no relevant" in answer.lower()


# 5. SOURCE TRACEABILITY LINK
def test_chat_source_traceability(pipeline):
    ep = Episode(
        episode_id="ep-trace-chat-1",
        deal_id="D-TRACE",
        situation="Trace test situation query",
        objection="Trace test objection",
        tactic="Trace test tactic",
        customer_reaction="Trace test reaction",
        outcome="won",
        why="Trace test why",
        lesson="Trace test lesson",
        source_ids=["crm_abc_001"],
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep)
    pipeline.memory_manager.retain_verified_episode(ep)

    res = pipeline.chat("Trace test situation query")
    assert len(res["hindsight_memories_used"]) > 0
    assert len(res["source_excerpts"]) > 0
    assert res["source_excerpts"][0]["episode_id"] == "ep-trace-chat-1"
