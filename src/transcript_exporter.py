from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .transcriber import TranscriptSegment
from .utils import ensure_parent, normalize_whitespace, seconds_to_timestamp, write_csv, write_json


def transcript_rows(segments: Iterable[TranscriptSegment]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        start = float(segment.start)
        end = float(segment.end)
        text = normalize_whitespace(segment.text)
        rows.append(
            {
                "index": index,
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
                "duration_seconds": round(max(end - start, 0.0), 3),
                "word_count": len(segment.words) if segment.words else len(text.split()),
                "text": text,
            }
        )
    return rows


def format_timestamped_transcript(segments: Iterable[TranscriptSegment]) -> str:
    lines: list[str] = []
    for row in transcript_rows(segments):
        lines.append(
            f"{row['index']:03d} | {row['start']} --> {row['end']} | {row['text']}"
        )
    return "\n".join(lines)


def write_timestamped_txt(path: Path, segments: Iterable[TranscriptSegment]) -> Path:
    ensure_parent(path)
    path.write_text(format_timestamped_transcript(segments), encoding="utf-8")
    return path


def write_timestamped_csv(path: Path, segments: Iterable[TranscriptSegment]) -> Path:
    rows = transcript_rows(segments)
    return write_csv(
        path,
        rows,
        fieldnames=["index", "start", "end", "duration_seconds", "word_count", "text"],
    )


def write_timestamped_srt(path: Path, segments: Iterable[TranscriptSegment]) -> Path:
    ensure_parent(path)
    lines: list[str] = []
    for index, row in enumerate(transcript_rows(segments), start=1):
        lines.append(str(index))
        lines.append(f"{row['start']} --> {row['end']}")
        lines.append(str(row["text"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_transcript_json(path: Path, transcript: dict[str, object]) -> Path:
    payload = dict(transcript)
    payload["segments"] = [asdict(segment) for segment in transcript.get("segments", [])]
    return write_json(path, payload)
