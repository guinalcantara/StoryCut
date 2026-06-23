from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from .transcriber import TranscriptSegment
from .utils import normalize_whitespace

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


@dataclass
class DuplicateMatch:
    removed_start: float
    removed_end: float
    kept_start: float
    kept_end: float
    similarity: int
    removed_text: str
    kept_text: str


def _ratio(left: str, right: str) -> int:
    if fuzz is not None:
        return int(round(fuzz.ratio(left, right)))
    return int(round(SequenceMatcher(None, left, right).ratio() * 100))


def _normalize(text: str) -> str:
    return normalize_whitespace(text).lower()


def find_duplicate_matches(
    segments: list[TranscriptSegment],
    threshold: int = 87,
    lookback: int = 3,
) -> list[DuplicateMatch]:
    matches: list[DuplicateMatch] = []
    removed_indices: set[int] = set()
    for current_index, current in enumerate(segments):
        if current_index in removed_indices:
            continue
        current_text = _normalize(current.text)
        if not current_text:
            continue
        start_index = max(0, current_index - lookback)
        for previous_index in range(start_index, current_index):
            if previous_index in removed_indices:
                continue
            previous = segments[previous_index]
            previous_text = _normalize(previous.text)
            if not previous_text:
                continue
            score = _ratio(previous_text, current_text)
            if score >= threshold:
                removed_indices.add(previous_index)
                matches.append(
                    DuplicateMatch(
                        removed_start=previous.start,
                        removed_end=previous.end,
                        kept_start=current.start,
                        kept_end=current.end,
                        similarity=score,
                        removed_text=previous.text,
                        kept_text=current.text,
                    )
                )
                break
    return matches


def duplicate_matches_to_rows(matches: list[DuplicateMatch]) -> list[dict[str, Any]]:
    return [asdict(match) for match in matches]

