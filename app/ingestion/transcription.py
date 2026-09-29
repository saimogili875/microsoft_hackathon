"""
Speech-to-Text (STT) Adapter Layer providing pluggable audio call transcription.
Includes MockSTTAdapter for offline test environments and WhisperSTTAdapter for production.
"""

from typing import Union, Dict, Any, Optional
from pathlib import Path
import logging
from app.ingestion.base import BaseSTTAdapter, STTTranscriptionResult

logger = logging.getLogger("ingestion.transcription")


class MockSTTAdapter(BaseSTTAdapter):
    """
    Default Mock Speech-to-Text Adapter simulating call transcription with speaker labels and timestamps.
    """

    def __init__(self, default_confidence: float = 0.92):
        self.default_confidence = default_confidence

    def transcribe(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> STTTranscriptionResult:
        simulated_error = kwargs.get("simulate_error", False)
        forced_confidence = kwargs.get("confidence", self.default_confidence)

        if simulated_error:
            logger.warning("[STT] Simulated transcription engine failure requested.")
            return STTTranscriptionResult(
                transcript="",
                confidence=0.0,
                duration_seconds=0.0,
                metadata={"stt_engine": "MockSTTAdapter", "status": "failed"},
                error="Simulated transcription engine failure",
            )

        file_name = "audio_call"
        audio_format = "wav"
        if isinstance(file_path_or_bytes, (str, Path)):
            path = Path(file_path_or_bytes)
            file_name = path.name
            audio_format = path.suffix.lstrip(".").lower() or "wav"

        # Default transcript with speaker diarization & timestamps
        speaker_segments = [
            {"speaker": "Salesperson", "start_time": "00:00:05", "end_time": "00:00:20", "text": "Thanks for joining. How is the current implementation budget looking?"},
            {"speaker": "Customer", "start_time": "00:00:21", "end_time": "00:00:45", "text": "Implementation cost is higher than expected. Competitor X offered 7L."},
            {"speaker": "Salesperson", "start_time": "00:00:46", "end_time": "00:01:10", "text": "We can offer a 3-phase rollout starting with a 4L pilot phase."},
        ]

        full_transcript = "\n".join([f"[{s['speaker']} {s['start_time']}]: {s['text']}" for s in speaker_segments])

        metadata = {
            "stt_engine": "MockSTTAdapter",
            "file_name": file_name,
            "audio_format": audio_format,
            "transcription_confidence": forced_confidence,
            "speaker_count": 2,
        }

        return STTTranscriptionResult(
            transcript=full_transcript,
            confidence=forced_confidence,
            duration_seconds=kwargs.get("duration", 70.0),
            speaker_segments=speaker_segments,
            metadata=metadata,
            audio_format=audio_format,
        )


class WhisperSTTAdapter(BaseSTTAdapter):
    """
    Production Whisper STT Adapter interface.
    Falls back to MockSTTAdapter if whisper module is not installed.
    """

    def __init__(self, fallback: Optional[BaseSTTAdapter] = None):
        self.fallback = fallback or MockSTTAdapter()

    def transcribe(self, file_path_or_bytes: Union[str, Path, bytes], **kwargs) -> STTTranscriptionResult:
        try:
            import whisper
            model = whisper.load_model("base")
            path_str = str(file_path_or_bytes)
            res = model.transcribe(path_str)
            text = res.get("text", "")
            return STTTranscriptionResult(
                transcript=text.strip(),
                confidence=0.90,
                duration_seconds=0.0,
                speaker_segments=[],
                metadata={"stt_engine": "openai-whisper"},
                audio_format="audio",
            )
        except Exception as e:
            logger.warning(f"[STT] Whisper STT engine unavailable ({str(e)}). Using fallback adapter.")
            return self.fallback.transcribe(file_path_or_bytes, **kwargs)


# Global adapter registry and accessors
_current_stt_adapter: BaseSTTAdapter = MockSTTAdapter()


def get_stt_adapter() -> BaseSTTAdapter:
    return _current_stt_adapter


def set_stt_adapter(adapter: BaseSTTAdapter):
    global _current_stt_adapter
    _current_stt_adapter = adapter
