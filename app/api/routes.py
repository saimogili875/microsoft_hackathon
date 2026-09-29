"""
FastAPI Route Definitions for Deal Intelligence Data Pipeline.
Supports Mode 01 Live Calls, Mode 02 File Ingestion & Manual Chat, Change Detection, and Reports.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    class BaseModel: pass

from app.services.pipeline import DealIntelligencePipeline
from app.models.live_interaction import LiveTranscriptChunk
from app.verification.verification_models import VerificationAction

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Deal Intelligence Data Pipeline API",
        description="Data ingestion, Mode 01 Live Call Intelligence, Mode 02 Manual Chat & Historical Data, Mode 02-B Change Detection, Groq reasoning, and Hindsight Memory retention.",
        version="2.1.0",
    )
    pipeline = DealIntelligencePipeline()

    class ChatRequestPayload(BaseModel):
        user_query: str
        client_context: Optional[str] = None

    class ReviewPayload(BaseModel):
        episode_id: str
        action: str
        role: Optional[str] = "account_executive"
        corrections: Optional[Dict[str, Any]] = None
        comments: Optional[str] = None

    class RecallPayload(BaseModel):
        query: str
        top_k: Optional[int] = 5

    class ReportRequestPayload(BaseModel):
        report_type: str
        context_data: Dict[str, Any]

    @app.get("/api/v1/health")
    def health_check():
        return {
            "status": "online",
            "version": "2.1.0",
            "hindsight": pipeline.hindsight_client.health_check(),
        }

    # --- Mode 02 Ingestion & Manual Chat ---
    @app.post("/api/v1/ingest/file")
    async def ingest_file(file: UploadFile = File(...), source_type: str = Form("auto")):
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = pipeline.process_file(tmp_path, source_type=source_type)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/v1/chat")
    def manual_chat(payload: ChatRequestPayload):
        try:
            res = pipeline.chat(payload.user_query, client_context=payload.client_context)
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # --- Mode 01 Live Call Intelligence ---
    @app.post("/api/v1/live/chunk")
    def process_live_chunk(chunk: LiveTranscriptChunk):
        try:
            res = pipeline.process_live_chunk(chunk)
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/v1/live/finalize/{session_id}")
    def finalize_live_session(session_id: str):
        try:
            res = pipeline.finalize_live_session(session_id)
            return {
                "status": "session_finalized",
                "session_id": session_id,
                "episode_id": res["episode_candidate"].episode_id,
                "verification_status": "pending_review",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # --- Report Generation ---
    @app.post("/api/v1/reports/generate")
    def generate_report(payload: ReportRequestPayload):
        try:
            report_text = pipeline.generate_report(payload.report_type, payload.context_data)
            return {
                "report_type": payload.report_type,
                "report": report_text,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # --- Verification & Retention ---
    @app.get("/api/v1/verification/pending")
    def get_pending_episodes():
        episodes = pipeline.verification_service.get_pending_episodes()
        return [ep.to_dict() for ep in episodes]

    @app.post("/api/v1/verification/review")
    def review_episode(payload: ReviewPayload):
        try:
            res = pipeline.verify_and_retain(
                episode_id=payload.episode_id,
                action=payload.action,
                role=payload.role or "account_executive",
                corrections=payload.corrections,
            )
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/v1/hindsight/recall")
    def recall_memories(payload: RecallPayload):
        res = pipeline.hindsight_client.recall(payload.query, top_k=payload.top_k or 5)
        return res

    @app.get("/api/v1/trace/{episode_id}")
    def trace_episode(episode_id: str):
        res = pipeline.traceability_service.trace_by_episode_id(episode_id)
        return res
else:
    app = None
