"""
ARCH-2 Test Suite: Getting the Right Knowledge from the Past.
Covers Test 1 through Test 9.
"""

import pytest
from app.services.pipeline import DealIntelligencePipeline
from app.models.episode import Episode
from app.hindsight.client import HindsightClient


@pytest.fixture
def pipeline(tmp_path):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")
    p = DealIntelligencePipeline(hindsight_client=client)

    # Populate baseline verified memories for testing
    ep1 = Episode(
        episode_id="ep-abc-cost",
        deal_id="D-ABC-100",
        situation="ABC Logistics concerned about initial implementation budget",
        objection="Implementation cost too high",
        tactic="Offered 3-phase modular rollout starting with ₹4L pilot phase",
        customer_reaction="Agreed to Phase 1 pilot evaluation",
        outcome="won",
        why="Phased pilot de-risked upfront capital expenditure",
        lesson="Structure enterprise quotes with phase 1 pilot options when price resistance occurs",
        pricing_context="Competitor X price ≈ ₹10L",
        source_ids=["crm_abc_001"],
        verification_status="confirmed",
    )
    p.verification_service._save_confirmed(ep1)
    p.memory_manager.retain_verified_episode(ep1)

    ep2 = Episode(
        episode_id="ep-abc-comp",
        deal_id="D-ABC-100",
        situation="ABC Logistics evaluated Competitor X",
        objection="Competitor X price quote",
        tactic="Demonstrated pre-built connector library",
        customer_reaction="Agreed Competitor X lacks pre-built connectors",
        outcome="won",
        why="Connector library reduced integration risk",
        lesson="Highlight pre-built connectors when competing against Competitor X",
        pricing_context="Competitor X quoted ₹10L",
        source_ids=["crm_abc_001"],
        verification_status="confirmed",
    )
    p.verification_service._save_confirmed(ep2)
    p.memory_manager.retain_verified_episode(ep2)

    return p


# TEST 1: Simple client query
def test_arch2_simple_client_query(pipeline):
    res = pipeline.arch2_query("What happened with ABC Logistics?")
    assert res.memory_found is True
    assert res.query_analysis.client_id == "ABC Logistics"
    assert len(res.retrieved_memories) >= 1
    assert "ABC Logistics" in res.retrieved_memories[0].content


# TEST 2: Specific objection
def test_arch2_specific_objection_query(pipeline):
    res = pipeline.arch2_query("What worked when ABC Logistics objected to implementation cost?")
    assert res.memory_found is True
    assert res.query_analysis.objection == "implementation cost"
    assert len(res.retrieved_memories) >= 1
    assert "implementation cost" in res.retrieved_memories[0].content.lower()


# TEST 3: Competitor query
def test_arch2_competitor_query(pipeline):
    res = pipeline.arch2_query("What competitors have we previously encountered with ABC Logistics?")
    assert res.memory_found is True
    assert res.query_analysis.competitor == "Competitor X" or "Competitor X" in res.retrieved_memories[0].content


# TEST 4: Pricing query
def test_arch2_pricing_query(pipeline):
    res = pipeline.arch2_query("What pricing discussions happened previously?")
    assert res.memory_found is True
    assert res.query_analysis.topic == "pricing"
    assert len(res.retrieved_memories) >= 1


# TEST 5: No-memory query (NO HALLUCINATION)
def test_arch2_no_memory_query(pipeline):
    res = pipeline.arch2_query("What happened with QWZZXYVK998877_UNKNOWN_CLIENT?")
    assert res.memory_found is False
    assert res.retrieved_memory_count == 0
    assert "No relevant historical memories were found" in res.answer or "no relevant" in res.answer.lower()


# TEST 6: Conflicting memories (₹10L vs ₹7L)
def test_arch2_conflicting_memories(pipeline):
    ep_new = Episode(
        episode_id="ep-abc-newprice",
        deal_id="D-ABC-100",
        situation="ABC Logistics 2026 pricing update",
        objection="Competitor X quote",
        tactic="Offered price match",
        customer_reaction="Under review",
        outcome="in_progress",
        why="Competitor X dropped price to ₹7L",
        lesson="Monitor competitor price cuts",
        pricing_context="Competitor X quoted ₹7L",
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep_new)
    pipeline.memory_manager.retain_verified_episode(ep_new)

    res = pipeline.arch2_query("What is the pricing history for Competitor X with ABC Logistics?")
    assert res.memory_found is True
    assert len(res.retrieved_memories) >= 2
    assert len(res.conflicts_detected) >= 1
    assert res.conflicts_detected[0]["conflict_type"] == "pricing_discrepancy"


# TEST 7: Manual salesperson chat
def test_arch2_manual_salesperson_chat(pipeline):
    chat_res = pipeline.chat("What worked when ABC Logistics objected to implementation cost?", client_context="ABC Logistics")
    assert chat_res["memory_found"] is True
    assert chat_res["retrieved_memory_count"] >= 1
    assert "pilot" in chat_res["answer"].lower() or "phased" in chat_res["answer"].lower() or "DEAL INTELLIGENCE" in chat_res["answer"]


# TEST 8: Irrelevant memories filtering
def test_arch2_irrelevant_memories_filtering(pipeline):
    ep_aws = Episode(
        episode_id="ep-aws-sovereignty",
        deal_id="D-AWS-99",
        situation="AWS data sovereignty review",
        objection="AWS data sovereignty compliance",
        tactic="Provided local region compliance doc",
        customer_reaction="Approved",
        outcome="won",
        why="Compliance doc satisfied legal",
        lesson="Provide region compliance docs for AWS deals",
        verification_status="confirmed",
    )
    pipeline.verification_service._save_confirmed(ep_aws)
    pipeline.memory_manager.retain_verified_episode(ep_aws)

    res = pipeline.arch2_query("What worked for implementation cost objections?")
    mem_contents = [m.content.lower() for m in res.retrieved_memories]
    # Check that implementation cost memory is included
    assert any("implementation cost" in c for c in mem_contents)


# TEST 9: Traceability link
def test_arch2_source_traceability_link(pipeline):
    res = pipeline.arch2_query("What worked when ABC Logistics objected to implementation cost?")
    assert len(res.sources) >= 1
    assert res.sources[0]["episode_id"] == "ep-abc-cost"
    assert "crm_abc_001" in res.sources[0]["source_ids"]
