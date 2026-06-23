from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .transcriber import TranscriptSegment
from .utils import ass_timestamp, ensure_parent, seconds_to_timestamp


@dataclass
class SubtitleEvent:
    start: float
    end: float
    text: str


def _chunk_text(text: str, max_words: int = 6) -> list[str]:
    words = [word for word in text.split() if word]
    if not words:
        return []
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def build_events(
    segments: Iterable[TranscriptSegment],
    start: float,
    end: float,
    max_words: int = 6,
) -> list[SubtitleEvent]:
    events: list[SubtitleEvent] = []
    for segment in segments:
        segment_start = max(segment.start, start)
        segment_end = min(segment.end, end)
        if segment_end <= segment_start:
            continue
        chunks = _chunk_text(segment.text, max_words=max_words)
        if not chunks:
            continue
        duration = max(segment_end - segment_start, 0.25)
        chunk_duration = duration / len(chunks)
        cursor = segment_start
        for index, chunk in enumerate(chunks):
            chunk_end = segment_end if index == len(chunks) - 1 else min(segment_end, cursor + chunk_duration)
            events.append(SubtitleEvent(start=cursor, end=chunk_end, text=chunk))
            cursor = chunk_end
    return events


def write_srt(path: Path, events: list[SubtitleEvent]) -> Path:
    ensure_parent(path)
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        lines.append(str(index))
        lines.append(f"{seconds_to_timestamp(event.start)} --> {seconds_to_timestamp(event.end)}")
        lines.append(event.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _ass_text(text: str) -> str:
    return text.replace("\n", r"\N").replace("{", r"\{").replace("}", r"\}")


def write_ass(path: Path, events: list[SubtitleEvent]) -> Path:
    ensure_parent(path)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,82,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,5,2,2,60,60,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip()]
    for event in events:
        lines.append(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
                start=ass_timestamp(event.start),
                end=ass_timestamp(event.end),
                text=_ass_text(event.text),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
