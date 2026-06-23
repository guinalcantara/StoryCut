from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned.lower() or "item"


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    base = slugify(stem)
    token = timestamp_token()
    candidate = directory / f"{base}_{token}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}_{token}_{counter}{suffix}"
        counter += 1
    return candidate


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> Path:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> Path:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def seconds_to_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def ffmpeg_filter_escape_path(path: Path) -> str:
    text = path.as_posix().replace(":", r"\:")
    text = text.replace("'", r"\'")
    return text


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or "Unknown command failure"
        raise RuntimeError(message)
    return result


def find_executable(name: str) -> str | None:
    from shutil import which

    return which(name)


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    filtered = sorted((start, end) for start, end in intervals if end > start)
    if not filtered:
        return []
    merged: list[list[float]] = [[filtered[0][0], filtered[0][1]]]
    for start, end in filtered[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def complement_intervals(duration: float, intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = merge_intervals(intervals)
    if duration <= 0:
        return []
    if not merged:
        return [(0.0, duration)]
    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            kept.append((cursor, min(start, duration)))
        cursor = max(cursor, end)
    if cursor < duration:
        kept.append((cursor, duration))
    return [(start, end) for start, end in kept if end > start]


def subtract_intervals(
    base_intervals: Iterable[tuple[float, float]],
    removal_intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    base = merge_intervals(base_intervals)
    removals = merge_intervals(removal_intervals)
    if not base:
        return []
    if not removals:
        return base
    kept: list[tuple[float, float]] = []
    for start, end in base:
        cursor = start
        for remove_start, remove_end in removals:
            if remove_end <= cursor:
                continue
            if remove_start >= end:
                break
            if remove_start > cursor:
                kept.append((cursor, min(remove_start, end)))
            cursor = max(cursor, remove_end)
            if cursor >= end:
                break
        if cursor < end:
            kept.append((cursor, end))
    return [(start, end) for start, end in kept if end > start]
