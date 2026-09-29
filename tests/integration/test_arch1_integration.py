"""
ARCH-1 Integration Test Suite.
Covers TEST A, TEST B, TEST C, and TEST D.
"""

import tempfile
import pytest
from pathlib import Path

from app.ingestion.registry import default_loader_registry
from app.normalization.normalizer import DataNormalizer
from app.extraction.groq_extractor import GroqLLMExtractor
from app.validation.extraction_validator import ExtractionValidator
from app.verification.verification_service import VerificationService
from app.verification.verification_models import VerificationAction, VerificationRequest
from app.hindsight.client import HindsightClient
from app.hindsight.retain import retain_episode
from app.services.traceability import TraceabilityService


@pytest.fixture
def fixtures_dir():
    return Path(__file__).resolve().parent.parent / "fixtures"


# TEST A: Raw text -> Groq -> structured extraction -> Python validation -> pending review
def test_integration_a_raw_to_pending_review(fixtures_dir, tmp_path):
    loader = default_loader_registry.get_loader("txt")
    raw = loader.load_file(fixtures_dir / "raw_email.txt")[0]
    
    normalizer = DataNormalizer()
    norm = normalizer.normalize(raw)

    extractor = GroqLLMExtractor()
    extraction_result = extractor.extract_from_record(norm)

    validator = ExtractionValidator(error_dir=tmp_path / "errors")
    val_res = validator.validate_extraction(extraction_result, raw_text=norm.raw_text_content)
    assert val_res.is_valid is True

    ep_cand = extractor.convert_to_episode(extraction_result.episodes[0], norm)
    v_service = VerificationService(pending_dir=tmp_path / "pending", confirmed_dir=tmp_path / "confirmed")
    v_service.save_pending(ep_cand)

    pending_list = v_service.get_pending_episodes()
    assert len(pending_list) == 1
    assert pending_list[0].verification_status == "pending_review"


# TEST B: Confirmed episode -> Hindsight RETAIN -> verify successful storage
def test_integration_b_confirmed_to_hindsight_retain(fixtures_dir, tmp_path):
    v_service = VerificationService(pending_dir=tmp_path / "pending", confirmed_dir=tmp_path / "confirmed")
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")

    # Ingest and convert
    loader = default_loader_registry.get_loader("txt")
    raw = loader.load_file(fixtures_dir / "raw_meeting_note.md")[0]
    norm = DataNormalizer().normalize(raw)
    extractor = GroqLLMExtractor()
    ext_res = extractor.extract_from_record(norm)
    episode = extractor.convert_to_episode(ext_res.episodes[0], norm)

    v_service.save_pending(episode)
    req = VerificationRequest(episode_id=episode.episode_id, action=VerificationAction.CONFIRM, reviewer_role="account_executive")
    confirmed_ep = v_service.process_review(req)

    payload = retain_episode(confirmed_ep, client)
    assert payload.document_id == f"episode:{episode.episode_id}"

    stored = client.get_document(payload.document_id)
    assert stored is not None
    assert "[EPISODE]" in stored.content


# TEST C: Duplicate confirmed episode -> verify idempotent behavior
def test_integration_c_duplicate_idempotent_retain(tmp_path):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")
    v_service = VerificationService(pending_dir=tmp_path / "pending", confirmed_dir=tmp_path / "confirmed")

    loader = default_loader_registry.get_loader("txt")
    raw = loader.load_file(Path(__file__).resolve().parent.parent / "fixtures" / "raw_transcript.txt")[0]
    norm = DataNormalizer().normalize(raw)
    extractor = GroqLLMExtractor()
    ext_res = extractor.extract_from_record(norm)
    episode = extractor.convert_to_episode(ext_res.episodes[0], norm)

    v_service.save_pending(episode)
    req = VerificationRequest(episode_id=episode.episode_id, action=VerificationAction.CONFIRM)
    confirmed_ep = v_service.process_review(req)

    res1 = retain_episode(confirmed_ep, client)
    res2 = retain_episode(confirmed_ep, client)

    assert res1.document_id == res2.document_id
    assert client.health_check()["total_memories"] == 1


# TEST D: Raw source -> episode -> source_id -> Hindsight document -> verify complete traceability
def test_integration_d_complete_traceability(tmp_path):
    client = HindsightClient(use_local_mock=True, storage_dir=tmp_path / "hs")
    v_service = VerificationService(pending_dir=tmp_path / "pending", confirmed_dir=tmp_path / "confirmed")
    t_service = TraceabilityService(client, v_service)

    loader = default_loader_registry.get_loader("txt")
    raw = loader.load_file(Path(__file__).resolve().parent.parent / "fixtures" / "raw_crm_note.txt")[0]
    norm = DataNormalizer().normalize(raw)
    extractor = GroqLLMExtractor()
    ext_res = extractor.extract_from_record(norm)
    episode = extractor.convert_to_episode(ext_res.episodes[0], norm)

    v_service.save_pending(episode)
    req = VerificationRequest(episode_id=episode.episode_id, action=VerificationAction.CONFIRM)
    confirmed_ep = v_service.process_review(req)
    retain_episode(confirmed_ep, client)

    trace = t_service.trace_by_episode_id(episode.episode_id)
    assert trace["hindsight_document_id"] == f"episode:{episode.episode_id}"
    assert raw.source_id in trace["source_ids"]
