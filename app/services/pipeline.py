"""
Deal Intelligence Pipeline Orchestrator executing full ETL, Mode 01 Live Calls, Mode 02-B Change Detection, and Report Generation.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from app.ingestion.registry import LoaderRegistry, default_loader_registry
from app.normalization.normalizer import DataNormalizer
from app.validation.validator import DataValidator
from app.extraction.episode_extractor import EpisodeExtractor
from app.verification.verification_service import VerificationService
from app.verification.verification_models import VerificationRequest, VerificationAction
from app.hindsight.client import HindsightClient
from app.hindsight.memory_manager import MemoryManager
from app.services.traceability import TraceabilityService
from app.services.live_intelligence_service import LiveIntelligenceService
from app.services.change_detection_service import ChangeDetectionService
from app.services.report_generator import ReportGenerator
from app.models.raw_data import RawRecord
from app.models.normalized_data import NormalizedRecord
from app.models.episode import Episode
from app.models.change_event import ChangeEvent, ClientState
from app.models.live_interaction import LiveTranscriptChunk


class DealIntelligencePipeline:
    """
    Complete end-to-end data pipeline orchestrator supporting:
    - Mode 01: Live Interaction Intelligence
    - Mode 02: Provided / Historical Data Ingestion
    - Mode 02-B: Post-Meeting Change Detection
    - 5 Report Types Generation
    - Common Hindsight Input Contract & Verification
    """

    def __init__(
        self,
        loader_registry: Optional[LoaderRegistry] = None,
        anonymize: bool = True,
        hindsight_client: Optional[HindsightClient] = None,
    ):
        self.loader_registry = loader_registry or default_loader_registry
        self.normalizer = DataNormalizer(anonymize=anonymize)
        self.validator = DataValidator()
        self.episode_extractor = EpisodeExtractor()
        self.verification_service = VerificationService()
        self.hindsight_client = hindsight_client or HindsightClient(use_local_mock=True)
        self.memory_manager = MemoryManager(self.hindsight_client)
        self.traceability_service = TraceabilityService(self.hindsight_client, self.verification_service)
        
        # Extended Services
        self.live_service = LiveIntelligenceService(self.hindsight_client)
        self.change_service = ChangeDetectionService()
        self.report_generator = ReportGenerator()

    # --- Mode 02: Historical / Provided Data File Ingestion ---
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
            norm_val_res = self.validator.validate_normalized(norm_rec)
            if not norm_val_res.is_valid:
                validation_errors.extend([e.model_dump() for e in norm_val_res.errors])
                continue

            episode = self.episode_extractor.extract_episode(norm_rec)
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

    # --- Mode 01: Live Call Stream Processing ---
    def process_live_chunk(self, chunk: LiveTranscriptChunk) -> Dict[str, Any]:
        return self.live_service.process_chunk(chunk)

    def finalize_live_session(self, session_id: str) -> Dict[str, Any]:
        res = self.live_service.finalize_meeting(session_id)
        episode = res["episode_candidate"]
        self.verification_service.save_pending(episode)
        return res

    # --- Mode 02-B: Post-Meeting Change Detection ---
    def detect_changes(self, baseline: ClientState, new_interaction: NormalizedRecord) -> List[ChangeEvent]:
        return self.change_service.detect_changes(baseline, new_interaction)

    # --- Report Generation ---
    def generate_report(self, report_type: str, context_data: Dict[str, Any]) -> str:
        r_type = report_type.lower().strip()
        if r_type == "client_relationship":
            return self.report_generator.generate_client_relationship_report(
                context_data["client_state"], context_data.get("historical_records", [])
            )
        elif r_type == "live_interaction":
            return self.report_generator.generate_live_interaction_report(context_data["live_state"])
        elif r_type == "post_meeting_change":
            return self.report_generator.generate_post_meeting_change_report(context_data["changes"])
        elif r_type == "deal_intelligence":
            return self.report_generator.generate_deal_intelligence_report(
                context_data["client_state"],
                context_data.get("episodes", []),
                context_data.get("recalls", []),
                context_data.get("changes", []),
            )
        elif r_type == "memory_update":
            return self.report_generator.generate_memory_update_report(context_data["episodes"])
        else:
            raise ValueError(f"Unknown report type: '{report_type}'")

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
