"""
Verification Service managing episode review workflows and file storage transitions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from app.models.episode import Episode
from app.verification.verification_models import VerificationAction, VerificationRequest
from config.settings import settings


class VerificationService:
    """
    Manages pending episodes, human verification actions (CONFIRM, CORRECT, REJECT),
    and preserves original extraction audit logs when corrected.
    """

    def __init__(
        self,
        pending_dir: Optional[Path] = None,
        confirmed_dir: Optional[Path] = None,
    ):
        self.pending_dir = pending_dir or settings.pending_review_dir
        self.confirmed_dir = confirmed_dir or settings.confirmed_dir
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.confirmed_dir.mkdir(parents=True, exist_ok=True)

    def save_pending(self, episode: Episode) -> Path:
        episode.verification_status = "pending_review"
        file_path = self.pending_dir / f"{episode.episode_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(episode.to_dict(), f, indent=2)
        return file_path

    def get_pending_episodes(self) -> List[Episode]:
        episodes = []
        for file_path in self.pending_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    episodes.append(Episode(**data))
            except Exception:
                continue
        return episodes

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        # Check pending first
        p_path = self.pending_dir / f"{episode_id}.json"
        if p_path.exists():
            with open(p_path, "r", encoding="utf-8") as f:
                return Episode(**json.load(f))
        # Check confirmed next
        c_path = self.confirmed_dir / f"{episode_id}.json"
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                return Episode(**json.load(f))
        return None

    def process_review(self, request: VerificationRequest) -> Episode:
        episode = self.get_episode_by_id(request.episode_id)
        if not episode:
            raise FileNotFoundError(f"Episode not found: {request.episode_id}")

        now_iso = datetime.now(timezone.utc).isoformat()
        episode.verified_by_role = request.reviewer_role
        episode.verified_at = now_iso

        pending_file = self.pending_dir / f"{episode.episode_id}.json"

        if request.action == VerificationAction.CONFIRM:
            episode.verification_status = "confirmed"
            # Move to confirmed
            if pending_file.exists():
                pending_file.unlink()
            self._save_confirmed(episode)

        elif request.action == VerificationAction.CORRECT:
            episode.verification_status = "corrected"
            
            # Record original extraction snapshot into audit log
            original_snapshot = {
                "corrected_at": now_iso,
                "corrected_by": request.reviewer_role,
                "comments": request.comments,
                "previous_state": {
                    "situation": episode.situation,
                    "objection": episode.objection,
                    "tactic": episode.tactic,
                    "customer_reaction": episode.customer_reaction,
                    "outcome": episode.outcome,
                    "why": episode.why,
                    "lesson": episode.lesson,
                    "pricing_context": episode.pricing_context,
                },
            }
            episode.corrections.append(original_snapshot)

            # Apply corrections if provided
            if request.corrections:
                for k, v in request.corrections.items():
                    if hasattr(episode, k) and v is not None:
                        setattr(episode, k, v)

            # Human verified corrections increase extraction confidence
            episode.extraction_confidence = 1.0

            if pending_file.exists():
                pending_file.unlink()
            self._save_confirmed(episode)

        elif request.action == VerificationAction.REJECT:
            episode.verification_status = "rejected"
            if pending_file.exists():
                pending_file.unlink()

        return episode

    def _save_confirmed(self, episode: Episode) -> Path:
        file_path = self.confirmed_dir / f"{episode.episode_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(episode.to_dict(), f, indent=2)
        return file_path
