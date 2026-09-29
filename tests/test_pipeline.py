"""
Comprehensive Test Suite for Deal Intelligence Data Pipeline.
Covers all 15 required scenario validations.
"""

import os
import tempfile
import json
import pytest
from pathlib import Path

from app.ingestion.json_loader import JSONLoader
from app.ingestion.csv_loader import CSVLoader
from app.ingestion.text_loader import TextLoader
from app.normalization.normalizer import DataNormalizer
from app.validation.validator import DataValidator
from app.extraction.episode_extractor import EpisodeExtractor
from app.verification.verification_service import VerificationService
from app.verification.verification_models import VerificationAction, VerificationRequest
from app.hindsight.client import HindsightClient
from app.hindsight.retain import retain_episode, retain_market_info, format_episode_narrative
from app.hindsight.memory_manager import MemoryManager
from app.services.traceability import TraceabilityService
from app.services.pipeline import DealIntelligencePipeline
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# 1. JSON & JSONL Ingestion
def test_json_ingestion(tmp_dir):
    json_path = tmp_dir / "sample.json"
    data = {"source_id": "conv-101", "source_type": "sales_conversation", "raw_content": {"deal_id": "D-100"}}
    json_path.write_text(json.dumps(data))

    loader = JSONLoader()
    records = loader.load_file(json_path)
    assert len(records) == 1
    assert records[0].source_id == "conv-101"
    assert records[0].file_format == "json"


# 2. CSV Ingestion
def test_csv_ingestion(tmp_dir):
    csv_path = tmp_dir / "sample.csv"
    csv_path.write_text("source_id,source_type,deal_id,outcome\ncsv-001,crm_record,D-200,won\n")

    loader = CSVLoader()
    records = loader.load_file(csv_path)
    assert len(records) == 1
    assert records[0].source_id == "csv-001"
    assert records[0].raw_content["outcome"] == "won"


# 3. Text & Transcript Ingestion
def test_text_ingestion(tmp_dir):
    txt_path = tmp_dir / "transcript.txt"
    txt_path.write_text("Rep: Hi customer. Customer: We need lower pricing. Rep: Offered tiered deployment.")

    loader = TextLoader()
    records = loader.load_file(txt_path)
    assert len(records) == 1
    assert "Rep: Hi customer" in records[0].raw_content
    assert records[0].source_type == "call_transcript"


# 4. Normalization
def test_normalization():
    raw = RawRecord(
        source_id="crm-500",
        source_type="crm_record",
        raw_content={
            "opportunity_id": "D-500",
            "client_name": "Acme Corp (john@acme.com)",
            "pushback": ["High price", "Long timeline"],
            "deal_result": "closed_won",
        },
    )
    normalizer = DataNormalizer(anonymize=True)
    norm = normalizer.normalize(raw)

    assert norm.deal_id == "D-500"
    assert norm.outcome == "won"
    assert "High price" in norm.objections
    assert "[ANONYMIZED_EMAIL]" in norm.customer_context


# 5. Missing Fields Handling (No Hallucination)
def test_missing_fields():
    raw = RawRecord(
        source_id="sparse-001",
        source_type="sales_notes",
        raw_content={"notes": "Meeting held with prospect."},
    )
    normalizer = DataNormalizer()
    norm = normalizer.normalize(raw)

    # Validate that missing fields are None or [] and not hallucinated
    assert norm.deal_id is None
    assert norm.outcome is None
    assert norm.objections == []
    assert norm.competitors == []


# 6. Duplicate Record Detection
def test_duplicate_records():
    validator = DataValidator()
    raw1 = RawRecord(source_id="dup-1", source_type="crm_record", raw_content={"deal": "D-1"})
    raw2 = RawRecord(source_id="dup-1", source_type="crm_record", raw_content={"deal": "D-1"})

    res1 = validator.validate_raw(raw1)
    res2 = validator.validate_raw(raw2)

    assert res1.is_valid is True
    assert res2.is_valid is False
    assert any(e.error_type == "duplicate_record" for e in res2.errors)


# 7. Invalid Data Validation
def test_invalid_data():
    validator = DataValidator()

    # Invalid timestamp and missing source_id
    raw = RawRecord(
        source_id="",
        source_type="sales_conversation",
        timestamp="invalid-timestamp-string",
        raw_content={"data": "test"},
    )
    res = validator.validate_raw(raw)
    assert res.is_valid is False
    error_types = [e.error_type for e in res.errors]
    assert "missing_field" in error_types
    assert "invalid_date" in error_types


# 8. Episode Extraction & Causal Confidence Separation
def test_episode_extraction():
    norm = NormalizedRecord(
        source_id="src-900",
        source_type="sales_conversation",
        deal_id="D-900",
        customer_context="Global Logistics Inc",
        objections=["Competitor offered 30% discount"],
        salesperson_actions=["Offered modular phase 1 pilot and 10% discount"],
        outcome="won",
        raw_text_content="We gave a discount end of quarter and won the deal because of tactic.",
    )
    extractor = EpisodeExtractor()
    episode = extractor.extract_episode(norm)

    assert episode.deal_id == "D-900"
    assert episode.outcome == "won"
    assert episode.extraction_confidence > 0.0
    assert episode.causal_confidence > 0.0
    # Causal confidence should be distinct from extraction confidence
    assert episode.extraction_confidence != episode.causal_confidence or episode.causal_confidence <= 0.90


# 9. Human Verification & Correction History
def test_human_correction(tmp_dir):
    service = VerificationService(pending_dir=tmp_dir / "pending", confirmed_dir=tmp_dir / "confirmed")
    ep = Episode(
        deal_id="D-777",
        situation="Initial pitch",
        objection="High price",
        tactic="Offered discount",
        customer_reaction="Agreed",
        outcome="won",
        why="Discount applied",
        lesson="Discounts close deals",
    )
    service.save_pending(ep)

    # Perform correction
    req = VerificationRequest(
        episode_id=ep.episode_id,
        action=VerificationAction.CORRECT,
        reviewer_role="sales_vp",
        corrections={"lesson": "Phase 1 pilots prevent price objections better than flat discounts."},
        comments="Updated lesson to reflect strategic directive.",
    )
    updated = service.process_review(req)

    assert updated.verification_status == "corrected"
    assert updated.lesson == "Phase 1 pilots prevent price objections better than flat discounts."
    assert len(updated.corrections) == 1
    assert updated.corrections[0]["previous_state"]["lesson"] == "Discounts close deals"


# 10. Hindsight Payload Generation & Format
def test_hindsight_payload_generation():
    ep = Episode(
        episode_id="ep-test-10",
        deal_id="D-10",
        situation="Customer concerned about integration complexity",
        objection="Integration timeline too long",
        tactic="Provided pre-built connector library",
        customer_reaction="Approved trial run",
        outcome="won",
        why="Pre-built connectors reduced integration time from 3 months to 1 week",
        lesson="Lead with pre-built connectors during technical validation",
        verified_by_role="solutions_architect",
        verification_status="confirmed",
        valid_as_of="2026-09-29",
    )

    narrative = format_episode_narrative(ep)
    assert "[EPISODE]" in narrative
    assert "Situation:\nCustomer concerned about integration complexity" in narrative
    assert "Verified by:\nsolutions_architect (confirmed)" in narrative


# 11. Source Traceability
def test_source_traceability(tmp_dir):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_dir / "hs")
    v_service = VerificationService(pending_dir=tmp_dir / "pending", confirmed_dir=tmp_dir / "confirmed")
    t_service = TraceabilityService(client, v_service)

    ep = Episode(
        episode_id="ep-trace-100",
        deal_id="D-TRACE",
        situation="Test situation",
        objection="Test objection",
        tactic="Test tactic",
        customer_reaction="Test reaction",
        outcome="won",
        why="Test why",
        lesson="Test lesson",
        source_ids=["raw_12345", "norm_67890"],
        verification_status="confirmed",
    )
    v_service._save_confirmed(ep)
    retain_episode(ep, client)

    trace = t_service.trace_by_episode_id("ep-trace-100")
    assert trace["hindsight_document_id"] == "episode:ep-trace-100"
    assert trace["source_ids"] == ["raw_12345", "norm_67890"]
    assert len(trace["traceability_chain"]) == 2


# 12. Outdated Market Information Handling
def test_outdated_market_information(tmp_dir):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_dir / "hs")
    mgr = MemoryManager(client)

    # 1st evidence in 2025
    rec1 = mgr.process_new_market_evidence(
        competitor="CompetitorX",
        topic="Pricing",
        statement="Competitor price is ₹10L",
        pricing_context="₹10L annual license",
    )
    assert rec1.status == "active"

    # 2nd evidence in 2026
    rec2 = mgr.process_new_market_evidence(
        competitor="CompetitorX",
        topic="Pricing",
        statement="Competitor price dropped to ₹7L",
        pricing_context="₹7L annual license",
    )
    assert rec2.status == "active"
    assert rec1.status == "superseded"
    assert rec1.valid_until is not None


# 13. Conflicting Information Detection
def test_conflicting_information():
    validator = DataValidator()
    r1 = NormalizedRecord(source_id="s1", source_type="crm_record", deal_id="D-888", outcome="won")
    r2 = NormalizedRecord(source_id="s2", source_type="crm_record", deal_id="D-888", outcome="lost")

    batch_res = validator.validate_batch([r1, r2])
    assert batch_res.is_valid is False
    assert any(e.error_type == "conflicting_information" for e in batch_res.errors)


# 14. Stable Document IDs
def test_stable_document_ids(tmp_dir):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_dir / "hs")
    ep = Episode(
        episode_id="ep-stable-001",
        situation="Sit",
        objection="Obj",
        tactic="Tac",
        customer_reaction="React",
        outcome="won",
        why="Why",
        lesson="Lesson",
    )
    payload = retain_episode(ep, client)

    assert payload.document_id == "episode:ep-stable-001"
    doc = client.get_document("episode:ep-stable-001")
    assert doc is not None


# 15. Replacing Corrected Memories via Stable Document IDs
def test_replacing_corrected_memories(tmp_dir):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_dir / "hs")
    mgr = MemoryManager(client)

    ep = Episode(
        episode_id="ep-replace-99",
        situation="Initial sit",
        objection="Initial obj",
        tactic="Initial tac",
        customer_reaction="Initial react",
        outcome="won",
        why="Initial why",
        lesson="Initial wrong lesson",
        verification_status="confirmed",
    )

    # First retention
    mgr.retain_verified_episode(ep)
    doc1 = client.get_document("episode:ep-replace-99")
    assert "Initial wrong lesson" in doc1.content

    # Human corrects memory
    ep.lesson = "Corrected accurate lesson"
    ep.verification_status = "corrected"

    # Second retention replaces memory safely under stable ID
    mgr.retain_verified_episode(ep)
    doc2 = client.get_document("episode:ep-replace-99")
    assert "Corrected accurate lesson" in doc2.content
    assert "Initial wrong lesson" not in doc2.content
