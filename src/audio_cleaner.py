from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
from pathlib import Path
from typing import Any

from .config import (
    DAVINCI_DIR,
    OUTPUTS_DIR,
    TRANSCRIPTIONS_DIR,
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_POST_SPEECH_PADDING_MS,
    DEFAULT_PRE_SPEECH_PADDING_MS,
    DEFAULT_SILENCE_THRESHOLD_DB,
    DEFAULT_WHISPER_MODEL,
)
from .duplicate_detector import find_duplicate_matches
from .subtitle_generator import build_events, write_ass, write_srt
from .text_matcher import transcript_text
from .transcriber import TranscriptSegment, save_transcript, transcribe_media
from .utils import (
    complement_intervals,
    ensure_parent,
    find_executable,
    merge_intervals,
    normalize_whitespace,
    run_command,
    subtract_intervals,
    write_csv,
    write_json,
)


@dataclass
class SilenceRange:
    start: float
    end: float


def probe_duration(media_path: Path) -> float:
    ffprobe = find_executable("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH.")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    return float((result.stdout or "0").strip() or 0.0)


def detect_silences(
    media_path: Path,
    silence_threshold_db: int = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
) -> list[SilenceRange]:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    noise = f"{silence_threshold_db}dB"
    duration = f"{min_silence_duration_ms / 1000:.3f}"
    result = run_command(
        [
            ffmpeg,
            "-i",
            str(media_path),
            "-af",
            f"silencedetect=noise={noise}:d={duration}",
            "-f",
            "null",
            os.devnull,
        ]
    )
    text = "\n".join(part for part in (result.stderr, result.stdout) if part)
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    duration = probe_duration(media_path)
    ranges: list[SilenceRange] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        if end > start:
            ranges.append(SilenceRange(start=start, end=end))
    return ranges


def _apply_padding(
    ranges: list[tuple[float, float]],
    pre_padding_ms: int,
    post_padding_ms: int,
    duration: float,
) -> list[tuple[float, float]]:
    pre = pre_padding_ms / 1000.0
    post = post_padding_ms / 1000.0
    padded = []
    for start, end in ranges:
        padded.append((max(0.0, start - pre), min(duration, end + post)))
    return merge_intervals(padded)


def _concat_audio_segments(
    media_path: Path,
    ranges: list[tuple[float, float]],
    output_path: Path,
) -> Path:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    ensure_parent(output_path)
    if not ranges:
        run_command([ffmpeg, "-y", "-i", str(media_path), "-vn", "-acodec", "pcm_s16le", str(output_path)])
        return output_path
    filter_parts: list[str] = []
    for index, (start, end) in enumerate(ranges):
        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]"
        )
    concat_inputs = "".join(f"[a{index}]" for index in range(len(ranges)))
    filter_parts.append(f"{concat_inputs}concat=n={len(ranges)}:v=0:a=1[outa]")
    filter_complex = ";".join(filter_parts)
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(media_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return output_path


def transcribe_and_analyze(
    media_path: Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    transcript = transcribe_media(media_path, model_name=model_name, language=language, device=device)
    return transcript


def clean_audio_pipeline(
    media_path: Path,
    script_text: str,
    output_root: Path = OUTPUTS_DIR,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str | None = None,
    device: str = "cpu",
    silence_threshold_db: int = DEFAULT_SILENCE_THRESHOLD_DB,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    pre_speech_padding_ms: int = DEFAULT_PRE_SPEECH_PADDING_MS,
    post_speech_padding_ms: int = DEFAULT_POST_SPEECH_PADDING_MS,
    duplicate_threshold: int = DEFAULT_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    transcript = transcribe_and_analyze(media_path, model_name=model_name, language=language, device=device)
    segments: list[TranscriptSegment] = transcript["segments"]
    duration = transcript.get("duration") or probe_duration(media_path)

    silence_ranges = detect_silences(
        media_path,
        silence_threshold_db=silence_threshold_db,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    silence_tuples = [(item.start, item.end) for item in silence_ranges]
    speech_ranges = complement_intervals(duration, silence_tuples)
    speech_ranges = _apply_padding(
        speech_ranges,
        pre_padding_ms=pre_speech_padding_ms,
        post_padding_ms=post_speech_padding_ms,
        duration=duration,
    )

    duplicate_matches = find_duplicate_matches(segments, threshold=duplicate_threshold)
    duplicate_ranges = [(match.removed_start, match.removed_end) for match in duplicate_matches]
    removal_ranges = merge_intervals([*silence_tuples, *duplicate_ranges])
    kept_ranges = subtract_intervals(speech_ranges, duplicate_ranges)

    cleaned_wav = output_root / "audio_limpo.wav"
    _concat_audio_segments(media_path, kept_ranges, cleaned_wav)

    cleaned_mp3 = output_root / "audio_limpo.mp3"
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    run_command([ffmpeg, "-y", "-i", str(cleaned_wav), str(cleaned_mp3)])

    transcript_path = TRANSCRIPTIONS_DIR / f"{media_path.stem}_transcript.json"
    save_transcript(transcript_path, transcript)

    script_similarity = 0
    if script_text.strip():
        from difflib import SequenceMatcher

        script_similarity = int(
            round(
                SequenceMatcher(
                    None,
                    normalize_whitespace(script_text).lower(),
                    transcript_text(segments).lower(),
                ).ratio()
                * 100
            )
        )

    report = {
        "input_file": str(media_path),
        "output_file": str(cleaned_wav),
        "model_name": model_name,
        "language": transcript.get("language"),
        "script_similarity": script_similarity,
        "removed_silences": [asdict(item) for item in silence_ranges],
        "removed_duplicates": [asdict(item) for item in duplicate_matches],
        "kept_ranges": [{"start": start, "end": end} for start, end in kept_ranges],
        "removed_ranges": [{"start": start, "end": end} for start, end in removal_ranges],
    }

    write_json(output_root / "relatorio_limpeza.json", report)
    rows: list[dict[str, Any]] = []
    for item in silence_ranges:
        rows.append({"type": "silence", "start": item.start, "end": item.end, "action": "removed", "reason": "silence"})
    for item in duplicate_matches:
        rows.append(
            {
                "type": "duplicate",
                "start": item.removed_start,
                "end": item.removed_end,
                "action": "removed",
                "reason": "repeated_take",
            }
        )
        rows.append(
            {
                "type": "speech",
                "start": item.kept_start,
                "end": item.kept_end,
                "action": "kept",
                "reason": "best_take",
            }
        )
    write_csv(
        output_root / "relatorio_limpeza.csv",
        rows,
        fieldnames=["type", "start", "end", "action", "reason"],
    )

    davinci_wav = DAVINCI_DIR / "audio_limpo.wav"
    davinci_csv = DAVINCI_DIR / "cortes_audio.csv"
    ensure_parent(davinci_wav)
    davinci_wav.write_bytes(cleaned_wav.read_bytes())
    write_csv(
        davinci_csv,
        rows,
        fieldnames=["type", "start", "end", "action", "reason"],
    )

    return {
        "cleaned_wav": cleaned_wav,
        "cleaned_mp3": cleaned_mp3,
        "report_json": output_root / "relatorio_limpeza.json",
        "report_csv": output_root / "relatorio_limpeza.csv",
        "davinci_wav": davinci_wav,
        "davinci_csv": davinci_csv,
        "transcript_path": transcript_path,
        "transcript": transcript,
        "speech_ranges": speech_ranges,
        "removed_ranges": removal_ranges,
    }
