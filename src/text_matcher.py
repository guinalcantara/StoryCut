from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable

from .transcriber import TranscriptSegment
from .utils import normalize_whitespace

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


@dataclass
class ClipMatch:
    start: float
    end: float
    start_score: int
    end_score: int
    start_phrase: str
    end_phrase: str


def _ratio(left: str, right: str) -> int:
    left = normalize_whitespace(left).lower()
    right = normalize_whitespace(right).lower()
    if fuzz is not None:
        return int(round(fuzz.partial_ratio(left, right)))
    return int(round(SequenceMatcher(None, left, right).ratio() * 100))


def split_target_phrases(text: str) -> tuple[str, str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "", ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        return cleaned, cleaned
    return sentences[0], sentences[-1]


def _window_text(segments: list[TranscriptSegment], start_index: int, window_size: int) -> str:
    window = segments[start_index : start_index + window_size]
    return normalize_whitespace(" ".join(segment.text for segment in window))


def _find_best_match_start(
    segments: list[TranscriptSegment],
    phrase: str,
    window_size: int = 4,
    threshold: int = 80,
) -> tuple[float, int]:
    if not phrase:
        return 0.0, 0
    best_score = -1
    best_time = 0.0
    for index in range(len(segments)):
        window = _window_text(segments, index, window_size)
        if not window:
            continue
        score = _ratio(phrase, window)
        if score > best_score:
            best_score = score
            best_time = segments[index].start
    if best_score < threshold:
        return 0.0, max(best_score, 0)
    return best_time, best_score


def _find_best_match_end(
    segments: list[TranscriptSegment],
    phrase: str,
    window_size: int = 4,
    threshold: int = 80,
) -> tuple[float, int]:
    if not phrase:
        return 0.0, 0
    best_score = -1
    best_time = 0.0
    for index in range(len(segments)):
        window = _window_text(segments, index, window_size)
        if not window:
            continue
        score = _ratio(phrase, window)
        if score > best_score:
            best_score = score
            window_end_index = min(len(segments) - 1, index + window_size - 1)
            best_time = segments[window_end_index].end
    if best_score < threshold:
        return 0.0, max(best_score, 0)
    return best_time, best_score


def find_clip_match(
    segments: list[TranscriptSegment],
    target_text: str,
    start_threshold: int = 80,
    end_threshold: int = 80,
) -> ClipMatch:
    start_phrase, end_phrase = split_target_phrases(target_text)
    if not start_phrase and not end_phrase:
        raise RuntimeError("The target excerpt is empty.")
    start_time, start_score = _find_best_match_start(segments, start_phrase, threshold=start_threshold)
    end_time, end_score = _find_best_match_end(segments, end_phrase, threshold=end_threshold)
    if start_score < start_threshold:
        raise RuntimeError(
            f"Could not confidently locate the beginning of the excerpt. Best score: {start_score}."
        )
    if end_score < end_threshold:
        raise RuntimeError(
            f"Could not confidently locate the end of the excerpt. Best score: {end_score}."
        )
    if end_time <= start_time:
        end_time = max(start_time + 1.0, segments[-1].end if segments else start_time + 1.0)
    return ClipMatch(
        start=start_time,
        end=end_time,
        start_score=start_score,
        end_score=end_score,
        start_phrase=start_phrase,
        end_phrase=end_phrase,
    )


def transcript_text(segments: Iterable[TranscriptSegment]) -> str:
    return normalize_whitespace(" ".join(segment.text for segment in segments))
