"""
Comprehensive Unit Test Suite for Deal Intelligence Input & Ingestion Layer.
Covers all 20 required ingestion test cases:
1. CSV ingestion
2. JSON ingestion
3. JSONL ingestion
4. TXT ingestion
5. Markdown ingestion
6. PDF ingestion
7. DOCX ingestion
8. Email ingestion
9. Audio transcription adapter
10. Image ingestion
11. Handwritten note OCR adapter
12. OCR failure handling
13. Low OCR confidence handling
14. Missing timestamps handling
15. Missing deal_id handling
16. Missing customer_id handling
17. Invalid files handling
18. Duplicate files detection
19. Unsupported formats error handling
20. Traceability from processed record -> original source
"""

import tempfile
import json
from pathlib import Path
import pytest

from app.ingestion.registry import IngestionRegistry
from app.ingestion.file_detector import FileDetector
from app.ingestion.base import OCRExtractionResult, STTTranscriptionResult, BaseOCRAdapter, BaseSTTAdapter
from app.ingestion.ocr import MockOCRAdapter, get_ocr_adapter, set_ocr_adapter
from app.ingestion.transcription import MockSTTAdapter, get_stt_adapter, set_stt_adapter
from app.models.raw_data import RawRecord
from app.services.pipeline import DealIntelligencePipeline


@pytest.fixture
def registry():
    reg = IngestionRegistry()
    reg.clear_duplicate_cache()
    return reg


@pytest.fixture
def pipeline():
    p = DealIntelligencePipeline()
    p.loader_registry.clear_duplicate_cache()
    return p


# 1. CSV Ingestion
def test_csv_ingestion(registry, tmp_path):
    csv_file = tmp_path / "deals.csv"
    csv_file.write_text(
        "source_id,deal_id,customer_id,salesperson_id,timestamp,situation,objection\n"
        "crm_001,D-ABC-100,ABC Logistics,rep_01,2026-09-01T10:00:00Z,Initial pitch,Price too high\n"
        "crm_002,D-ABC-101,Acme Corp,rep_02,2026-09-02T11:00:00Z,Follow up,Implementation budget\n"
    )

    recs = registry.ingest_file(csv_file, source_type="crm_record")
    assert len(recs) == 2
    assert recs[0].source_id == "crm_001"
    assert recs[0].deal_id == "D-ABC-100"
    assert recs[0].customer_id == "ABC Logistics"
    assert recs[0].file_format == "csv"
    assert recs[0].extraction_method == "direct"


# 2. JSON Ingestion
def test_json_ingestion(registry, tmp_path):
    json_file = tmp_path / "deals.json"
    data = [
        {"source_id": "json_001", "deal_id": "D-100", "customer_id": "Cust A", "content": "Deal details"},
        {"source_id": "json_002", "deal_id": "D-101", "customer_id": "Cust B", "content": "More details"},
    ]
    json_file.write_text(json.dumps(data))

    recs = registry.ingest_file(json_file, source_type="crm_record")
    assert len(recs) == 2
    assert recs[0].source_id == "json_001"
    assert recs[0].deal_id == "D-100"
    assert recs[0].customer_id == "Cust A"
    assert recs[0].file_format == "json"


# 3. JSONL Ingestion
def test_jsonl_ingestion(registry, tmp_path):
    jsonl_file = tmp_path / "episodes.jsonl"
    jsonl_file.write_text(
        json.dumps({"source_id": "line_01", "deal_id": "D-201", "content": "Line 1"}) + "\n" +
        json.dumps({"source_id": "line_02", "deal_id": "D-202", "content": "Line 2"}) + "\n"
    )

    recs = registry.ingest_file(jsonl_file, source_type="sales_conversation")
    assert len(recs) == 2
    assert recs[0].source_id == "line_01"
    assert recs[1].source_id == "line_02"
    assert recs[0].file_format == "jsonl"


# 4. TXT Ingestion
def test_txt_ingestion(registry, tmp_path):
    txt_file = tmp_path / "sales_note.txt"
    txt_file.write_text("deal_id: D-TXT-99\ncustomer_id: Acme Logistics\nCustomer accepted modular pricing.")

    recs = registry.ingest_file(txt_file, source_type="sales_notes")
    assert len(recs) == 1
    assert recs[0].file_format == "txt"
    assert recs[0].deal_id == "D-TXT-99"
    assert "Acme Logistics" in recs[0].customer_id
    assert "modular pricing" in recs[0].raw_content


# 5. Markdown Ingestion
def test_markdown_ingestion(registry, tmp_path):
    md_file = tmp_path / "meeting_notes.md"
    md_file.write_text(
        "---\n"
        "deal_id: D-MD-300\n"
        "customer_id: Globex Corp\n"
        "salesperson_id: rep_alice\n"
        "---\n"
        "# Executive Meeting Notes\n"
        "Discussed 3-phase rollout."
    )

    recs = registry.ingest_file(md_file, source_type="meeting_note")
    assert len(recs) == 1
    assert recs[0].file_format == "md"
    assert recs[0].deal_id == "D-MD-300"
    assert recs[0].customer_id == "Globex Corp"
    assert recs[0].salesperson_id == "rep_alice"


# 6. PDF Ingestion
def test_pdf_ingestion(registry, tmp_path):
    pdf_file = tmp_path / "market_report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 Market facts: Competitor X lowered price to 7L.")

    recs = registry.ingest_file(pdf_file, source_type="market_info")
    assert len(recs) == 1
    assert recs[0].file_format == "pdf"
    assert recs[0].source_type == "market_info"
    assert recs[0].extraction_method in ["pdf_text", "ocr"]


# 7. DOCX Ingestion
def test_docx_ingestion(registry, tmp_path):
    docx_file = tmp_path / "strategy.docx"
    docx_file.write_bytes(b"PK\x03\x04 Document content: deal_id: D-DOCX-1")

    recs = registry.ingest_file(docx_file, source_type="sales_notes")
    assert len(recs) == 1
    assert recs[0].file_format == "docx"


# 8. Email Ingestion
def test_email_ingestion(registry, tmp_path):
    eml_file = tmp_path / "customer_email.eml"
    eml_file.write_text(
        "From: buyer@acme.com\n"
        "To: rep@ourcompany.com\n"
        "Subject: Budget Objection - D-EML-500\n"
        "Date: Tue, 29 Sep 2026 10:00:00 +0000\n"
        "\n"
        "We are concerned about initial implementation cost for deal_id: D-EML-500."
    )

    recs = registry.ingest_file(eml_file, source_type="email")
    assert len(recs) == 1
    assert recs[0].file_format == "eml"
    assert recs[0].metadata["sender"] == "buyer@acme.com"
    assert recs[0].metadata["subject"] == "Budget Objection - D-EML-500"
    assert recs[0].deal_id == "D-EML-500"


# 9. Audio Transcription Adapter
def test_audio_transcription_adapter(registry, tmp_path):
    audio_file = tmp_path / "sales_call.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    recs = registry.ingest_file(audio_file, source_type="audio_call")
    assert len(recs) == 1
    assert recs[0].file_format == "wav"
    assert recs[0].extraction_method == "stt"
    assert recs[0].metadata["stt_engine"] == "MockSTTAdapter"
    assert recs[0].extraction_confidence > 0.0
    assert len(recs[0].metadata["speaker_segments"]) > 0


# 10. Image Ingestion
def test_image_ingestion(registry, tmp_path):
    img_file = tmp_path / "whiteboard_notes.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

    recs = registry.ingest_file(img_file, source_type="handwritten_note")
    assert len(recs) == 1
    assert recs[0].file_format == "jpg"
    assert recs[0].extraction_method == "ocr"
    assert "OCR Extracted Text" in recs[0].raw_content


# 11. Handwritten Note OCR Adapter
def test_handwritten_note_ocr_adapter(registry, tmp_path):
    class CustomOCRAdapter(BaseOCRAdapter):
        def extract_text(self, file_path_or_bytes, **kwargs):
            return OCRExtractionResult(
                extracted_text="Handwritten: Customer agreed to phase 1 pilot.",
                confidence=0.95,
                page_count=1,
                metadata={"ocr_engine": "CustomOCRAdapter"},
            )

    set_ocr_adapter(CustomOCRAdapter())
    img_file = tmp_path / "note.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    recs = registry.ingest_file(img_file, source_type="handwritten_note")
    assert len(recs) == 1
    assert recs[0].raw_content == "Handwritten: Customer agreed to phase 1 pilot."
    assert recs[0].extraction_confidence == 0.95

    # Reset OCR adapter
    set_ocr_adapter(MockOCRAdapter())


# 12. OCR Failure Handling
def test_ocr_failure_handling(registry, tmp_path):
    class FailingOCRAdapter(BaseOCRAdapter):
        def extract_text(self, file_path_or_bytes, **kwargs):
            return OCRExtractionResult(
                extracted_text="",
                confidence=0.0,
                page_count=1,
                metadata={"ocr_engine": "FailingOCRAdapter"},
                is_low_confidence=True,
                error="Corrupted image file unreadable by OCR engine",
            )

    set_ocr_adapter(FailingOCRAdapter())
    img_file = tmp_path / "corrupt_scan.jpg"
    img_file.write_bytes(b"CORRUPTED_BYTES")

    recs = registry.ingest_file(img_file, source_type="handwritten_note")
    assert len(recs) == 1
    assert recs[0].metadata["status"] == "failed"
    assert recs[0].metadata["flagged_for_review"] is True
    assert recs[0].extraction_confidence == 0.0

    set_ocr_adapter(MockOCRAdapter())


# 13. Low OCR Confidence (Flagged for Review)
def test_low_ocr_confidence_flagging(registry, tmp_path):
    class LowConfidenceOCRAdapter(BaseOCRAdapter):
        def extract_text(self, file_path_or_bytes, **kwargs):
            return OCRExtractionResult(
                extracted_text="Unclear text: Discount ... [unreadable]",
                confidence=0.45,
                page_count=1,
                metadata={"ocr_engine": "LowConfidenceOCRAdapter"},
                is_low_confidence=True,
            )

    set_ocr_adapter(LowConfidenceOCRAdapter())
    img_file = tmp_path / "blurry_note.jpg"
    img_file.write_bytes(b"BLURRY")

    recs = registry.ingest_file(img_file, source_type="handwritten_note")
    assert len(recs) == 1
    assert recs[0].metadata["flagged_for_review"] is True
    assert recs[0].extraction_confidence == 0.45

    set_ocr_adapter(MockOCRAdapter())


# 14. Missing Timestamps Handling
def test_missing_timestamps_handling(registry, tmp_path):
    json_file = tmp_path / "no_timestamp.json"
    json_file.write_text(json.dumps({"source_id": "no_ts_1", "deal_id": "D-999", "content": "No timestamp provided"}))

    recs = registry.ingest_file(json_file, source_type="crm_record")
    assert len(recs) == 1
    assert recs[0].timestamp is not None  # Auto-generated current ISO timestamp


# 15. Missing deal_id Handling
def test_missing_deal_id_handling(registry, tmp_path):
    json_file = tmp_path / "no_deal.json"
    json_file.write_text(json.dumps({"source_id": "no_deal_1", "customer_id": "Acme Corp", "content": "General account note"}))

    recs = registry.ingest_file(json_file, source_type="sales_notes")
    assert len(recs) == 1
    assert recs[0].deal_id is None  # Must remain None, not hallucinated


# 16. Missing customer_id Handling
def test_missing_customer_id_handling(registry, tmp_path):
    json_file = tmp_path / "no_customer.json"
    json_file.write_text(json.dumps({"source_id": "no_cust_1", "deal_id": "D-111", "content": "Market intelligence report"}))

    recs = registry.ingest_file(json_file, source_type="market_info")
    assert len(recs) == 1
    assert recs[0].customer_id is None  # Must remain None, not hallucinated


# 17. Invalid Files Handling
def test_invalid_files_handling(registry, tmp_path):
    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("{invalid_json_syntax:")

    with pytest.raises(ValueError, match="Malformed JSON"):
        registry.ingest_file(invalid_json, source_type="crm_record")


# 18. Duplicate Files Detection
def test_duplicate_files_detection(registry, tmp_path):
    f1 = tmp_path / "file1.txt"
    f1.write_text("Identical content for duplicate check")

    f2 = tmp_path / "file2.txt"
    f2.write_text("Identical content for duplicate check")

    recs1 = registry.ingest_file(f1, source_type="sales_notes")
    assert len(recs1) == 1

    recs2 = registry.ingest_file(f2, source_type="sales_notes")
    assert len(recs2) == 0  # Skipped as duplicate


# 19. Unsupported Formats Error Handling
def test_unsupported_formats_handling(registry, tmp_path):
    unsupported = tmp_path / "data.xyz_unknown"
    unsupported.write_text("some content")

    with pytest.raises(ValueError, match="Unsupported file format"):
        registry.ingest_file(unsupported)


# 20. Traceability from Processed Record -> Original Source
def test_traceability_from_processed_record_to_source(pipeline, tmp_path):
    csv_file = tmp_path / "crm_trace.csv"
    csv_file.write_text(
        "source_id,deal_id,customer_id,salesperson_id,situation,objection,tactic,customer_reaction,outcome,why,lesson\n"
        "crm_trace_999,D-TRACE-9,Acme Logistics,rep_01,Budget negotiation,Implementation cost too high,Offered 3-phase rollout starting with 4L pilot,Agreed to Phase 1 pilot,won,Pilot de-risked upfront capital,Offer phase 1 pilot when price resistance occurs\n"
    )

    res = pipeline.process_file(csv_file, source_type="crm_record")
    assert len(res["pending_review_episodes"]) > 0
    ep_id = res["pending_review_episodes"][0]

    # Verify human review & retention
    v_res = pipeline.verify_and_retain(ep_id, action="confirm", role="account_executive")
    assert v_res["verification_status"] == "confirmed"

    # Trace back to original source ID
    trace_info = pipeline.traceability_service.trace_by_episode_id(ep_id)
    assert trace_info["hindsight_memory_present"] is True
    assert len(trace_info["source_ids"]) > 0
    assert "crm_trace_999" in trace_info["source_ids"]
