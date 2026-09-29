"""
Audio Loader for call recordings (WAV, MP3, M4A, MP4).
Converts audio recordings to structured RawRecords using a pluggable BaseSTTAdapter.
"""

from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import logging
from app.ingestion.base import BaseLoader, BaseSTTAdapter
from app.models.raw_data import RawRecord
from app.ingestion.file_detector import FileDetector
from app.ingestion.transcription import get_stt_adapter

logger = logging.getLogger("ingestion.audio")


class AudioLoader(BaseLoader):
    """
    Loader for audio call recordings.
    Delegates speech transcription to pluggable BaseSTTAdapter.
    """

    def __init__(self, stt_adapter: Optional[BaseSTTAdapter] = None):
        self.stt_adapter = stt_adapter

    def load_file(self, file_path: Union[str, Path], source_type: str = "auto") -> List[RawRecord]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        adapter = self.stt_adapter or get_stt_adapter()
        stype = "audio_call" if source_type == "auto" else source_type
        source_id = f"audio_{FileDetector.calculate_hash(path)[:12]}"
        audio_format = path.suffix.lstrip(".").lower() or "wav"

        # Execute transcription via pluggable adapter
        stt_res = adapter.transcribe(path)

        metadata = {
            "audio_format": audio_format,
            "stt_engine": stt_res.metadata.get("stt_engine", "MockSTTAdapter"),
            "transcription_confidence": stt_res.confidence,
            "duration_seconds": stt_res.duration_seconds,
            "speaker_segments": stt_res.speaker_segments,
            "speaker_count": stt_res.metadata.get("speaker_count", len(stt_res.speaker_segments)),
            "status": "success" if not stt_res.error else "failed",
            "stt_error": stt_res.error,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=stype,
            file_name=path.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_format=audio_format,
            raw_content=stt_res.transcript or f"[Audio Call Recording: {path.name}]",
            metadata=metadata,
            source_location=str(path.resolve()),
            extraction_method="stt",
            extraction_confidence=stt_res.confidence,
        )

        return [record]

    def load_raw_content(
        self,
        content: Union[str, Dict[str, Any], List[Any]],
        source_type: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RawRecord]:
        meta = metadata or {}
        adapter = self.stt_adapter or get_stt_adapter()

        if isinstance(content, (bytes, bytearray)):
            stt_res = adapter.transcribe(content)
            transcript = stt_res.transcript
            confidence = stt_res.confidence
            speaker_segments = stt_res.speaker_segments
        else:
            transcript = str(content)
            confidence = meta.get("transcription_confidence", 1.0)
            speaker_segments = meta.get("speaker_segments", [])

        full_meta = {
            "transcription_confidence": confidence,
            "speaker_segments": speaker_segments,
            **meta,
        }

        record = RawRecord(
            source_id=source_id,
            source_type=source_type or "audio_call",
            file_name=meta.get("file_name"),
            timestamp=meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            file_format=meta.get("audio_format", "wav"),
            raw_content=transcript,
            metadata=full_meta,
            deal_id=meta.get("deal_id"),
            customer_id=meta.get("customer_id") or meta.get("client_id"),
            salesperson_id=meta.get("salesperson_id"),
            source_location=meta.get("source_location"),
            extraction_method="stt",
            extraction_confidence=confidence,
        )

        return [record]
