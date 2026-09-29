"""
Deal Intelligence Pipeline Master Orchestrator supporting ARCH-1 (Knowledge Retention) & ARCH-2 (Information Retrieval & Reasoning).
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from app.ingestion.registry import IngestionRegistry
from app.normalization.normalizer import DataNormalizer
from app.validation.validator import DataValidator
from app.validation.extraction_validator import ExtractionValidator
from app.extraction.episode_extractor import EpisodeExtractor
from app.extraction.groq_extractor import GroqLLMExtractor
from app.verification.verification_service import VerificationService
from app.verification.verification_models import VerificationRequest, VerificationAction
from app.hindsight.client import HindsightClient
from app.hindsight.memory_manager import MemoryManager
from app.services.traceability import TraceabilityService
from app.services.live_intelligence_service import LiveIntelligenceService
from app.services.change_detection_service import ChangeDetectionService
from app.services.report_generator import ReportGenerator
from app.services.groq_service import GroqService
from app.services.chat_service import SalespersonChatService
from app.services.arch2_service import Arch2Service
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.models.change_event import ChangeEvent, ClientState
from app.models.live_interaction import LiveTranscriptChunk
from app.models.retrieved_memory import Arch2ContextObject


class DealIntelligencePipeline:
    """
    Complete end-to-end data pipeline master orchestrator:
    - ARCH-1: Knowledge Saving to Past (Ingestion -> Groq Extraction -> Validation -> Verification -> Hindsight Retain)
    - ARCH-2: Getting Right Knowledge from Past (Query Analysis -> Hindsight Recall -> Filtering/Conflicts -> Groq -> Answer & Arch-3 Context Object)
    """

    def __init__(
        self,
        loader_registry: Optional[Any] = None,
        anonymize: bool = True,
        hindsight_client: Optional[HindsightClient] = None,
        groq_service: Optional[GroqService] = None,
    ):
        self.loader_registry = loader_registry or IngestionRegistry()
        self.normalizer = DataNormalizer(anonymize=anonymize)
        self.validator = DataValidator()
        self.extraction_validator = ExtractionValidator()
        self.groq_extractor = GroqLLMExtractor()
        self.episode_extractor = EpisodeExtractor()
        self.verification_service = VerificationService()
        self.hindsight_client = hindsight_client or HindsightClient()
        self.memory_manager = MemoryManager(self.hindsight_client)
        self.traceability_service = TraceabilityService(self.hindsight_client, self.verification_service)
        
        # Groq Reasoning & Arch-2 Services
        self.groq_service = groq_service or GroqService()
        self.arch2_service = Arch2Service(self.hindsight_client, self.groq_service, self.verification_service)
        self.chat_service = SalespersonChatService(self.hindsight_client, self.groq_service, self.verification_service)
        
        # Extended Services
        self.live_service = LiveIntelligenceService(self.hindsight_client)
        self.change_service = ChangeDetectionService()
        self.report_generator = ReportGenerator()

    # --- ARCH-1: Ingestion, Groq Extraction & Retention ---
    def process_file(self, file_path: Union[str, Path], source_type: str = "auto") -> Dict[str, Any]:
        raw_records = self.loader_registry.load(file_path, source_type=source_type)
        processed_episodes = []
        validation_errors = []

        for raw_rec in raw_records:
            val_res = self.validator.validate_raw(raw_rec)
            if not val_res.is_valid:
                validation_errors.extend([e.model_dump() for e in val_res.errors])
                continue

            norm_rec = self.normalizer.normalize(raw_rec)
            
            # Run Groq LLM Semantic Extraction
            ext_result = self.groq_extractor.extract_from_record(norm_rec)
            ext_val_res = self.extraction_validator.validate_extraction(ext_result, raw_text=norm_rec.raw_text_content)
            
            if not ext_val_res.is_valid:
                validation_errors.extend([e.model_dump() for e in ext_val_res.errors])
                continue

            for ext_ep in ext_result.episodes:
                episode = self.groq_extractor.convert_to_episode(ext_ep, norm_rec)
                self.verification_service.save_pending(episode)
                processed_episodes.append(episode)

        return {
            "status": "success",
            "file_processed": str(file_path),
            "raw_records_count": len(raw_records),
            "episodes_extracted": len(processed_episodes),
            "pending_review_episodes": [ep.episode_id for ep in processed_episodes],
            "validation_errors": validation_errors,
        }

    # --- ARCH-2: Information Retrieval & Reasoning ---
    def arch2_query(
        self,
        query: str,
        client_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        stage: Optional[str] = None,
        current_context: Optional[Dict[str, Any]] = None,
    ) -> Arch2ContextObject:
        """
        Executes ARCH-2 information retrieval pipeline.
        READ-ONLY.
        """
        return self.arch2_service.process_query(
            query=query, client_id=client_id, deal_id=deal_id, stage=stage, current_context=current_context
        )

    # --- Mode 02 Manual Chat ---
    def chat(self, user_query: str, client_context: Optional[str] = None) -> Dict[str, Any]:
        arch2_obj = self.arch2_query(query=user_query, client_id=client_context)
        memories_used = []
        for m in arch2_obj.retrieved_memories:
            d = m.to_dict()
            d["document_id"] = m.memory_id
            memories_used.append(d)
        return {
            "user_query": user_query,
            "answer": arch2_obj.answer,
            "hindsight_memories_used": memories_used,
            "source_excerpts": arch2_obj.sources,
            "memory_found": arch2_obj.memory_found,
            "retrieved_memory_count": arch2_obj.retrieved_memory_count,
            "conflicts_detected": arch2_obj.conflicts_detected,
        }

    # --- Mode 01 Live Stream ---
    def process_live_chunk(self, chunk: LiveTranscriptChunk) -> Dict[str, Any]:
        res = self.live_service.process_chunk(chunk)
        if res.get("recalled_experiences"):
            groq_payload = {
                "task": "Live Call Intelligence",
                "user_query": f"Live call signal: {chunk.text}",
                "current_interaction": res["extracted_signals"],
                "hindsight_memories": res["recalled_experiences"],
                "detected_changes": [],
            }
            groq_insight = self.groq_service.generate_response(groq_payload)
            res["groq_reasoning"] = groq_insight
        return res

    def finalize_live_session(self, session_id: str) -> Dict[str, Any]:
        res = self.live_service.finalize_meeting(session_id)
        episode = res["episode_candidate"]
        self.verification_service.save_pending(episode)
        return res

    # --- Mode 02-B Change Detection ---
    def detect_changes(self, baseline: ClientState, new_interaction: NormalizedRecord) -> List[ChangeEvent]:
        return self.change_service.detect_changes(baseline, new_interaction)

    # --- Report Generation ---
    def generate_report(self, report_type: str, context_data: Dict[str, Any]) -> str:
        if report_type == "live_interaction":
            return self.report_generator.generate_live_interaction_report(
                context_data.get("live_state")
            )
        elif report_type in ["change_detection", "post_meeting_change"]:
            return self.report_generator.generate_post_meeting_change_report(
                context_data.get("changes", [])
            )
        elif report_type in ["deal_strategy", "deal_intelligence"]:
            return self.report_generator.generate_deal_intelligence_report(
                context_data.get("client_state", ClientState(deal_id="unknown", customer_context="Unknown Account")),
                context_data.get("episodes", []),
                context_data.get("recalls", []),
                context_data.get("changes", []),
            )
        elif report_type in ["hindsight_learning", "memory_update"]:
            return self.report_generator.generate_memory_update_report(
                context_data.get("episodes", [])
            )
        else:
            return self.report_generator.generate_client_relationship_report(
                context_data.get("client_state"), context_data.get("historical_records", [])
            )

    # --- Human Verification Workflow ---
    def verify_and_retain(self, episode_id: str, action: str, role: str = "account_executive", corrections: dict = None) -> Dict[str, Any]:
        v_action = VerificationAction(action.lower())
        req = VerificationRequest(
            episode_id=episode_id,
            action=v_action,
            reviewer_role=role,
            corrections=corrections,
        )
        episode = self.verification_service.process_review(req)

        retain_result = None
        if episode.verification_status in ["confirmed", "corrected"]:
            retain_result = self.memory_manager.retain_verified_episode(episode)

        return {
            "episode_id": episode.episode_id,
            "verification_status": episode.verification_status,
            "retain_result": retain_result,
        }
