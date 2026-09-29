"""
Test suite for extended pipeline capabilities:
1. Mode 01 Live Interaction Intelligence & Real-time Hindsight Recall
2. Mode 02-B Post-Meeting Change Detection (18 categories)
3. Common Hindsight Input Contract Compliance
4. Generation of all 5 Report Types
"""

import tempfile
import json
import pytest
from pathlib import Path

from app.services.pipeline import DealIntelligencePipeline
from app.models.live_interaction import LiveTranscriptChunk
from app.models.change_event import ClientState, ChangeCategory
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode


@pytest.fixture
def pipeline(tmp_path):
    return DealIntelligencePipeline()


# 1. Mode 01 Live Interaction & Real-time Hindsight Recall
def test_mode01_live_interaction(pipeline):
    # Pre-populate Hindsight memory with past experience
    ep = Episode(
        episode_id="ep-hist-1",
        deal_id="D-PREV",
        situation="Competitor price parity deal",
        objection="Implementation cost too high",
        tactic="Phased rollout worked when competitor was at price parity",
        customer_reaction="Agreed to trial phase",
        outcome="won",
        why="Phased pilot de-risked upfront capital",
        lesson="Phased rollout worked when competitor was at price parity",
        verification_status="confirmed",
    )
    pipeline.memory_manager.retain_verified_episode(ep)

    session_id = "call_session_999"
    pipeline.live_service.start_session(session_id, deal_id="D-LIVE-1", customer_context="Acme Logistics")

    # Ingest live transcript chunk with price objection & competitor mention
    chunk1 = LiveTranscriptChunk(
        session_id=session_id,
        speaker="Customer",
        text="Competitor X is offering a implementation quote. Our implementation cost is too high and Competitor X is cheaper (~30%).",
    )
    res = pipeline.process_live_chunk(chunk1)

    # Check signal extraction
    assert "Competitor X" in res["extracted_signals"]["competitors_mentioned"]
    assert len(res["extracted_signals"]["customer_concerns"]) > 0

    # Check real-time Hindsight recall
    assert len(res["recalled_experiences"]) > 0
    assert res["recalled_experiences"][0]["document_id"] == "episode:ep-hist-1"

    # Check boundary condition recommendation (detecting competitor price shift)
    assert len(res["live_recommendations"]) > 0
    assert "⚠️ CAUTION" in res["live_recommendations"][0]

    # Finalize live call post-meeting
    fin = pipeline.finalize_live_session(session_id)
    cand = fin["episode_candidate"]
    assert cand.verification_status == "pending_review"
    assert cand.extraction_confidence > 0.0


# 2. Mode 02-B Post-Meeting Change Detection (18 Categories)
def test_mode02b_change_detection(pipeline):
    baseline = ClientState(
        deal_id="D-200",
        customer_context="Global Cargo Corp",
        stakeholders=["Alice Smith (VP)"],
        tools_used=["Tool Alpha"],
        competitors=["Competitor X"],
        competitor_pricing={"Competitor X": "₹10L"},
        active_objections=["Initial implementation cost"],
        commitments=["Demo scheduled"],
        pain_points=["Manual data entry"],
    )

    # New interaction with multiple changes
    new_norm = NormalizedRecord(
        source_id="crm_update_99",
        source_type="crm_record",
        deal_id="D-200",
        competitors=["Competitor X", "Competitor Y"],  # New competitor
        pricing_information=["Customer reports Competitor X is now ₹7L"],  # Price change
        stakeholders=["Alice Smith (VP)", "Bob Jones (CTO)"],  # New stakeholder
        objections=["Data security compliance"],  # New objection
        raw_text_content="Customer left company sponsor. Mentioned reorganization and new use case. Agreed to pilot phase.",
    )

    changes = pipeline.detect_changes(baseline, new_norm)
    assert len(changes) >= 4

    categories = [c.category for c in changes]
    assert ChangeCategory.NEW_COMPETITOR in categories
    assert ChangeCategory.COMPETITOR_PRICE_CHANGE in categories
    assert ChangeCategory.NEW_STAKEHOLDER in categories
    assert ChangeCategory.NEW_OBJECTION in categories

    # All change events MUST start as pending_review
    for c in changes:
        assert c.status == "pending_review"


# 3. Common Hindsight Input Contract Verification
def test_common_hindsight_contract(pipeline, tmp_path):
    # Mode 01 post-meeting candidate
    pipeline.live_service.start_session("s_contract", deal_id="D-C1", customer_context="Test Co")
    c = LiveTranscriptChunk(session_id="s_contract", speaker="Rep", text="Rep: Offered pilot option.")
    pipeline.process_live_chunk(c)
    mode01_res = pipeline.finalize_live_session("s_contract")
    ep_mode01 = mode01_res["episode_candidate"]

    # Mode 02 historical file ingestion candidate
    json_path = tmp_path / "hist.json"
    json_path.write_text(json.dumps({
        "source_id": "hist-10",
        "source_type": "crm_record",
        "deal_id": "D-C2",
        "raw_content": {"client_name": "Hist Co", "pushback": ["Price"], "deal_result": "won"}
    }))
    mode02_res = pipeline.process_file(json_path)

    pending = pipeline.verification_service.get_pending_episodes()
    assert len(pending) >= 2

    # Verify both Mode 01 and Mode 02 conform to identical Episode contract fields
    for ep in pending:
        assert hasattr(ep, "episode_id")
        assert hasattr(ep, "situation")
        assert hasattr(ep, "objection")
        assert hasattr(ep, "tactic")
        assert hasattr(ep, "customer_reaction")
        assert hasattr(ep, "outcome")
        assert hasattr(ep, "why")
        assert hasattr(ep, "lesson")
        assert hasattr(ep, "applies_when")
        assert hasattr(ep, "did_not_hold_when")
        assert hasattr(ep, "extraction_confidence")
        assert hasattr(ep, "causal_confidence")
        assert hasattr(ep, "verification_status")


# 4. Five Report Types Generation
def test_report_generation(pipeline):
    state = ClientState(
        deal_id="D-REP-1",
        customer_context="Apex Tech",
        stakeholders=["Sam VP"],
        tools_used=["Salesforce"],
        competitors=["Comp Alpha"],
        active_objections=["Price too high"],
    )
    ep = Episode(
        deal_id="D-REP-1",
        situation="Initial pitch",
        objection="Price too high",
        tactic="Offered modular pricing",
        customer_reaction="Agreed",
        outcome="won",
        why="Modular pricing reduced initial friction",
        lesson="Offer modular pricing when price resistance occurs",
        verification_status="confirmed",
    )

    # 1. Client Relationship Report
    r1 = pipeline.generate_report("client_relationship", {"client_state": state})
    assert "CLIENT RELATIONSHIP REPORT" in r1
    assert "Apex Tech" in r1

    # 2. Live Interaction Report
    sess = pipeline.live_service.start_session("s_rep", deal_id="D-REP-1")
    r2 = pipeline.generate_report("live_interaction", {"live_state": sess})
    assert "LIVE INTERACTION REPORT" in r2

    # 3. Post-Meeting Change Report
    r3 = pipeline.generate_report("post_meeting_change", {"changes": []})
    assert "POST-MEETING CHANGE REPORT" in r3

    # 4. Deal Intelligence Report
    r4 = pipeline.generate_report("deal_intelligence", {"client_state": state, "episodes": [ep]})
    assert "DEAL INTELLIGENCE REPORT" in r4

    # 5. Memory Update Report
    r5 = pipeline.generate_report("memory_update", {"episodes": [ep]})
    assert "HINDSIGHT MEMORY UPDATE REPORT" in r5
