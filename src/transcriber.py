from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import warnings

from .config import TRANSCRIPTIONS_DIR
from .utils import normalize_whitespace, write_json

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


ProgressCallback = Callable[[str, int, int], None]
TRANSCRIPT_CACHE_VERSION = "v1"


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, step: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(stage, step, total)


def _hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _transcript_cache_key(media_path: Path, model_name: str, language: str | None, device: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(TRANSCRIPT_CACHE_VERSION.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(_hash_file(media_path).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update((language or "").encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(device.encode("utf-8"))
    hasher.update(b"\0word_timestamps=1\0vad_filter=1")
    return hasher.hexdigest()


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
            if any(token in message for token in ("cuda", "driver", "nvidia", "cublas", "cudnn")):
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
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _transcript_cache_key(media_path, model_name, language, device)
    cache_path = TRANSCRIPTIONS_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            transcript = load_transcript(cache_path)
            transcript.update(
                {
                    "media_path": str(media_path),
                    "model_name": model_name,
                    "requested_device": device,
                    "cache_key": cache_key,
                    "cache_path": str(cache_path),
                    "cache_hit": True,
                }
            )
            _emit_progress(progress_callback, "transcription_cached", 1, 6)
            return transcript
        except Exception:
            pass

    _emit_progress(progress_callback, "loading_model", 0, 2)
    model, effective_device = _make_model(model_name, device=device)
    _emit_progress(progress_callback, "transcribing", 1, 2)
    segments_iter, info = model.transcribe(
        str(media_path),
        language=language,
        vad_filter=True,
        word_timestamps=True,
    )
    segments: list[TranscriptSegment] = []
    for segment in segments_iter:
        text = normalize_whitespace(segment.text or "")
        if not text:
            continue
        words: list[TranscriptWord] = []
        for word in getattr(segment, "words", None) or []:
            word_text = normalize_whitespace(getattr(word, "word", "") or "")
            if not word_text:
                continue
            words.append(
                TranscriptWord(
                    start=float(getattr(word, "start", segment.start)),
                    end=float(getattr(word, "end", segment.end)),
                    word=word_text,
                )
            )
        if words:
            text = normalize_whitespace(" ".join(item.word for item in words))
        segments.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
                words=words,
            )
        )
    transcript = {
        "media_path": str(media_path),
        "language": getattr(info, "language", language),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "segments": segments,
        "model_name": model_name,
        "device": effective_device,
        "requested_device": device,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "cache_hit": False,
    }
    _emit_progress(progress_callback, "transcription_done", 2, 2)
    save_transcript(cache_path, transcript)
    return transcript


def save_transcript(path: Path, transcript: dict[str, Any]) -> Path:
    payload = dict(transcript)
    payload["segments"] = [asdict(segment) for segment in transcript.get("segments", [])]
    return write_json(path, payload)


def load_transcript(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["segments"] = [
        TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=segment.get("text", ""),
            words=[TranscriptWord(**word) for word in segment.get("words", [])],
        )
        for segment in payload.get("segments", [])
    ]
    return payload

