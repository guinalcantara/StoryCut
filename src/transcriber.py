from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import warnings

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


ProgressCallback = Callable[[str, int, int], None]


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, step: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(stage, step, total)


def _make_model(model_name: str, device: str = "cpu") -> tuple[Any, str]:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Install the project dependencies first."
        )

    def _load(target_device: str) -> Any:
        compute_type = "int8" if target_device == "cpu" else "float16"
        return WhisperModel(model_name, device=target_device, compute_type=compute_type)

    if device == "cuda":
        try:
            return _load("cuda"), "cuda"
        except Exception as exc:
            message = str(exc).lower()
            if "cuda" in message or "driver" in message or "nvidia" in message:
                warnings.warn(
                    "CUDA failed in this environment, falling back to CPU. "
                    "Check your GPU driver and Docker GPU configuration if you want to use CUDA."
                )
                return _load("cpu"), "cpu"
            raise

    return _load("cpu"), "cpu"


def transcribe_media(
    media_path: Path,
    model_name: str = "medium",
    language: str | None = None,
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _emit_progress(progress_callback, "loading_model", 0, 2)
    model, effective_device = _make_model(model_name, device=device)
    _emit_progress(progress_callback, "transcribing", 1, 2)
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
    _emit_progress(progress_callback, "transcription_done", 2, 2)
    return {
        "media_path": str(media_path),
        "language": getattr(info, "language", language),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "segments": segments,
        "model_name": model_name,
        "device": effective_device,
        "requested_device": device,
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

