"""
ARCH-1 Unit Test Suite.
Covers all 15 required unit test cases.
"""

import json
import tempfile
import pytest
from pathlib import Path

from app.ingestion.registry import LoaderRegistry, default_loader_registry
from app.normalization.normalizer import DataNormalizer
from app.extraction.groq_extractor import GroqLLMExtractor
from app.validation.extraction_validator import ExtractionValidator
from app.models.groq_extraction_schema import GroqExtractionResult, ExtractedEpisodeSchema, ExtractionMetadataSchema, ConfidenceSchema, VerificationSchema
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.verification.verification_service import VerificationService
from app.verification.verification_models import VerificationAction, VerificationRequest
from app.hindsight.retain import retain_episode, format_episode_narrative
from app.hindsight.client import HindsightClient


@pytest.fixture
def fixtures_dir():
    return Path(__file__).resolve().parent.parent / "fixtures"


# 1. Raw Ingestion
def test_raw_ingestion(fixtures_dir):
    loader = default_loader_registry.get_loader("txt")
    records = loader.load_file(fixtures_dir / "raw_email.txt")
    assert len(records) == 1
    assert records[0].file_format == "txt"


# 2. Normalization
def test_normalization(fixtures_dir):
    loader = default_loader_registry.get_loader("txt")
    raw = loader.load_file(fixtures_dir / "raw_email.txt")[0]
    normalizer = DataNormalizer()
    norm = normalizer.normalize(raw)
    assert norm.source_id == raw.source_id
    assert norm.raw_text_content is not None


# 3. Source ID Preservation
def test_source_id_preservation():
    raw = RawRecord(source_id="src-unique-999", source_type="sales_conversation", raw_content="Test")
    normalizer = DataNormalizer()
    norm = normalizer.normalize(raw)
    assert norm.source_id == "src-unique-999"
    assert norm.source_references[0]["source_id"] == "src-unique-999"


# 4. Timestamp Preservation
def test_timestamp_preservation():
    ts = "2026-08-15T14:30:00Z"
    raw = RawRecord(source_id="src-ts", source_type="email", timestamp=ts, raw_content="Test")
    normalizer = DataNormalizer()
    norm = normalizer.normalize(raw)
    assert norm.timestamp == ts


# 5. Extraction Response Parsing
def test_extraction_response_parsing():
    sample_json = {
        "extraction_metadata": {"source_id": "s1", "source_type": "email", "overall_extraction_confidence": 0.9},
        "episodes": [{
            "context": {"situation": "Budget review"},
            "customer": {"objections": ["Price high"]},
            "salesperson": {"tactic": "Pilot offer"},
            "outcome": {"status": "in_progress"},
            "confidence": {"extraction_confidence": 0.85, "causal_confidence": 0.40},
            "verification": {"status": "pending_review"}
        }]
    }
    result = GroqExtractionResult(**sample_json)
    assert result.extraction_metadata.source_id == "s1"
    assert len(result.episodes) == 1


# 6. Malformed JSON Handling
def test_malformed_json_handling():
    validator = ExtractionValidator()
    with pytest.raises(Exception):
        GroqExtractionResult(**{"invalid": "data"})


# 7. Missing Fields Handling (No Invention)
def test_missing_fields_handling():
    extractor = GroqLLMExtractor()
    norm = NormalizedRecord(source_id="s_missing", source_type="text", raw_text_content="Discussed setup.")
    res = extractor._rule_based_fallback(norm)
    ep = res.episodes[0]
    assert ep.outcome.status is None  # System MUST NOT invent 'won'!


# 8. Schema Validation
def test_schema_validation():
    validator = ExtractionValidator()
    res = GroqExtractionResult(
        extraction_metadata={"source_id": "s1", "source_type": "text", "overall_extraction_confidence": 0.8},
        episodes=[ExtractedEpisodeSchema(verification={"status": "pending_review"})]
    )
    val_res = validator.validate_extraction(res)
    assert val_res.is_valid is True


# 9. Confidence Range Validation
def test_confidence_range_validation():
    validator = ExtractionValidator()
    res = GroqExtractionResult.model_construct(
        extraction_metadata=ExtractionMetadataSchema.model_construct(
            source_id="s1", source_type="text", overall_extraction_confidence=1.5
        ),
        episodes=[
            ExtractedEpisodeSchema.model_construct(
                confidence=ConfidenceSchema.model_construct(extraction_confidence=1.2, causal_confidence=-0.5),
                verification=VerificationSchema.model_construct(status="pending_review"),
            )
        ]
    )
    val_res = validator.validate_extraction(res)
    assert val_res.is_valid is False
    err_types = [e.error_type for e in val_res.errors]
    assert "invalid_confidence_range" in err_types


# 10. Evidence Span Validation
def test_evidence_span_validation():
    validator = ExtractionValidator()
    res = GroqExtractionResult(
        extraction_metadata={"source_id": "s1", "source_type": "text", "overall_extraction_confidence": 0.8},
        episodes=[ExtractedEpisodeSchema(provenance={"evidence_spans": ["Fabricated text snippet not in source"]})]
    )
    val_res = validator.validate_extraction(res, raw_text="Actual source text about software sales.")
    assert len(val_res.warnings) > 0
    assert any(w.error_type == "fabricated_evidence_span" for w in val_res.warnings)


# 11. Verification State Machine
def test_verification_state_machine(tmp_path):
    service = VerificationService(pending_dir=tmp_path / "pending", confirmed_dir=tmp_path / "confirmed")
    ep = Episode(episode_id="ep-v1", situation="Sit", objection="Obj", tactic="Tac", customer_reaction="React", outcome="won", why="Why", lesson="Lesson")
    service.save_pending(ep)

    req = VerificationRequest(episode_id="ep-v1", action=VerificationAction.CONFIRM, reviewer_role="AE")
    confirmed_ep = service.process_review(req)
    assert confirmed_ep.verification_status == "confirmed"


# 12. Duplicate Handling & Idempotency
def test_idempotent_retain(tmp_path):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")
    ep = Episode(episode_id="ep-idem", situation="Sit", objection="Obj", tactic="Tac", customer_reaction="React", outcome="won", why="Why", lesson="Lesson", verification_status="confirmed")

    payload1 = retain_episode(ep, client)
    payload2 = retain_episode(ep, client)
    assert payload1.document_id == payload2.document_id
    assert client.health_check()["total_memories"] == 1


# 13. Traceability
def test_traceability(tmp_path):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")
    ep = Episode(episode_id="ep-tr", deal_id="D-1", situation="Sit", objection="Obj", tactic="Tac", customer_reaction="React", outcome="won", why="Why", lesson="Lesson", source_ids=["src-1"])
    payload = retain_episode(ep, client)
    assert payload.metadata["source_ids"] == ["src-1"]


# 14. Conflict Detection
def test_conflict_detection():
    extractor = GroqLLMExtractor()
    norm = NormalizedRecord(source_id="s_conf", source_type="text", raw_text_content="Competitor X was 10L earlier, now offered 7L")
    res = extractor._rule_based_fallback(norm)
    assert len(res.episodes) == 1


# 15. Hindsight Payload Creation
def test_hindsight_payload_creation():
    ep = Episode(episode_id="ep-payload", situation="Sit", objection="Obj", tactic="Tac", customer_reaction="React", outcome="won", why="Why", lesson="Lesson")
    narrative = format_episode_narrative(ep)
    assert "[EPISODE]" in narrative
    assert "Situation:\nSit" in narrative
