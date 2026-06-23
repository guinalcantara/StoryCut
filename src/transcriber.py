from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import normalize_whitespace, write_json

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _make_model(model_name: str, device: str = "cpu") -> Any:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install the project dependencies first."
        )
    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_media(
    media_path: Path,
    model_name: str = "medium",
    language: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    model = _make_model(model_name, device=device)
    segments_iter, info = model.transcribe(
        str(media_path),
        language=language,
        vad_filter=True,
    )
    segments: list[TranscriptSegment] = []
    for segment in segments_iter:
        text = normalize_whitespace(segment.text or "")
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
            )
        )
    return {
        "media_path": str(media_path),
        "language": getattr(info, "language", language),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "segments": segments,
        "model_name": model_name,
        "device": device,
    }


def save_transcript(path: Path, transcript: dict[str, Any]) -> Path:
    payload = dict(transcript)
    payload["segments"] = [asdict(segment) for segment in transcript.get("segments", [])]
    return write_json(path, payload)


def load_transcript(path: Path) -> dict[str, Any]:
    data = path.read_text(encoding="utf-8")
    import json

    payload = json.loads(data)
    payload["segments"] = [TranscriptSegment(**segment) for segment in payload.get("segments", [])]
    return payload

