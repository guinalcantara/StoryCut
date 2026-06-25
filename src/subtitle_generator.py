from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import (
    DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
    DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT,
    DEFAULT_SHORTS_SUBTITLE_BASE_FONT,
    DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
    DEFAULT_SHORTS_SUBTITLE_COLOR,
    DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
    DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
    DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
    DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
    DEFAULT_SHORTS_SUBTITLE_SPACING,
)
from .transcriber import TranscriptSegment, TranscriptWord
from .utils import ass_timestamp, ensure_parent, normalize_whitespace, seconds_to_timestamp


@dataclass
class SubtitleEvent:
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)


def _segment_words(segment: TranscriptSegment) -> list[TranscriptWord]:
    if segment.words:
        return [
            TranscriptWord(start=float(word.start), end=float(word.end), word=normalize_whitespace(word.word))
            for word in segment.words
            if normalize_whitespace(word.word)
        ]

    words = [word for word in normalize_whitespace(segment.text).split() if word]
    if not words:
        return []

    duration = max(segment.end - segment.start, 0.2)
    step = duration / max(len(words), 1)
    fallback_words: list[TranscriptWord] = []
    for index, word in enumerate(words):
        word_start = segment.start + index * step
        word_end = segment.end if index == len(words) - 1 else min(segment.end, word_start + step)
        fallback_words.append(
            TranscriptWord(
                start=float(word_start),
                end=float(max(word_end, word_start + 0.05)),
                word=word,
            )
        )
    return fallback_words


def build_events(
    segments: Iterable[TranscriptSegment],
    start: float,
    end: float,
    max_words: int = 6,
    max_duration_seconds: float = 4.0,
    max_gap_seconds: float = 0.45,
) -> list[SubtitleEvent]:
    words: list[TranscriptWord] = []
    for segment in segments:
        for word in _segment_words(segment):
            clipped_start = max(start, float(word.start))
            clipped_end = min(end, float(word.end))
            if clipped_end <= clipped_start:
                continue
            words.append(
                TranscriptWord(
                    start=clipped_start,
                    end=clipped_end,
                    word=normalize_whitespace(word.word),
                )
            )

    words.sort(key=lambda item: (item.start, item.end))

    events: list[SubtitleEvent] = []
    current_words: list[TranscriptWord] = []
    for word in words:
        if not current_words:
            current_words.append(word)
            continue

        previous_word = current_words[-1]
        current_start = current_words[0].start
        gap = word.start - previous_word.end
        duration = word.end - current_start

        if gap > max_gap_seconds or len(current_words) >= max_words or duration > max_duration_seconds:
            text = normalize_whitespace(" ".join(item.word for item in current_words))
            events.append(
                SubtitleEvent(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=text,
                    words=list(current_words),
                )
            )
            current_words = [word]
            continue

        current_words.append(word)

    if current_words:
        text = normalize_whitespace(" ".join(item.word for item in current_words))
        events.append(
            SubtitleEvent(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=list(current_words),
            )
        )

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


def _hex_to_ass_color(hex_color: str) -> str:
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid subtitle color: {hex_color!r}")
    red = cleaned[0:2]
    green = cleaned[2:4]
    blue = cleaned[4:6]
    return f"&H00{blue}{green}{red}&"


def _word_font_tags(is_last_word: bool, base_font: str, accent_font: str, base_size: int, accent_size: int) -> str:
    if is_last_word:
        return rf"{{\fn{accent_font}\fs{accent_size}\b0\i1}}"
    return rf"{{\fn{base_font}\fs{base_size}\b0\i0}}"


_SHORT_FUNCTION_WORDS = {
    "a",
    "as",
    "e",
    "o",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "pra",
    "pro",
    "com",
    "sem",
    "se",
    "me",
    "te",
    "lhe",
    "que",
    "ou",
    "ja",
    "eh",
    "é",
}


def _is_meaningful_accent_word(word: str) -> bool:
    normalized = normalize_whitespace(word).strip().strip(".,;:!?\"'()[]{}")
    if len(normalized) < 3:
        return False
    lowered = normalized.casefold()
    if lowered in _SHORT_FUNCTION_WORDS:
        return False
    return any(char.isalpha() for char in normalized)


def _styled_karaoke_text(
    event: SubtitleEvent,
    base_font: str,
    accent_font: str,
    base_size: int,
    accent_size: int,
    accent_last_word: bool,
) -> str:
    if not event.words:
        return _ass_text(event.text)

    parts: list[str] = []
    for index, word in enumerate(event.words):
        is_last_word = index == len(event.words) - 1
        use_accent_font = accent_last_word and is_last_word and _is_meaningful_accent_word(word.word)
        word_start = float(word.start)
        word_end = float(word.end)
        if index + 1 < len(event.words):
            next_start = float(event.words[index + 1].start)
            duration_seconds = max(next_start - word_start, 0.01)
        else:
            duration_seconds = max(event.end - word_start, word_end - word_start, 0.01)
        duration_cs = max(1, int(round(duration_seconds * 100)))
        font_tag = _word_font_tags(use_accent_font, base_font, accent_font, base_size, accent_size)
        parts.append(r"{\k%d}%s%s" % (duration_cs, font_tag, _ass_text(word.word)))
        if index + 1 < len(event.words):
            parts.append(" ")
    return "".join(parts)


def write_ass(
    path: Path,
    events: list[SubtitleEvent],
    subtitle_color: str = DEFAULT_SHORTS_SUBTITLE_COLOR,
    highlight_color: str = DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
    font_size: int = DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
    outline_size: int = DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
    shadow_size: int = DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
    margin_v: int = DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
    spacing: float = DEFAULT_SHORTS_SUBTITLE_SPACING,
    base_font: str = DEFAULT_SHORTS_SUBTITLE_BASE_FONT,
    accent_font: str = DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT,
    accent_font_size: int | None = None,
    accent_last_word: bool = DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
) -> Path:
    ensure_parent(path)
    ass_color = _hex_to_ass_color(subtitle_color)
    ass_highlight = _hex_to_ass_color(highlight_color)
    accent_font_size = accent_font_size or int(round(font_size * 1.08))
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{base_font},{font_size},{color},{highlight},&H00000000,&H64000000,0,0,0,0,100,100,{spacing},0,1,{outline},{shadow},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        base_font=base_font,
        font_size=font_size,
        color=ass_color,
        highlight=ass_highlight,
        spacing=spacing,
        outline=outline_size,
        shadow=shadow_size,
        margin_v=margin_v,
    )
    lines = [header.rstrip()]
    for event in events:
        lines.append(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}".format(
                start=ass_timestamp(event.start),
                end=ass_timestamp(event.end),
                text=_styled_karaoke_text(
                    event,
                    base_font=base_font,
                    accent_font=accent_font,
                    base_size=font_size,
                    accent_size=accent_font_size,
                    accent_last_word=accent_last_word,
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
