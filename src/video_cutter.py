from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable

from .config import OUTPUTS_DIR, SHORTS_DIR, DEFAULT_END_PADDING_SECONDS, DEFAULT_MATCH_THRESHOLD, DEFAULT_START_PADDING_SECONDS
from .subtitle_generator import build_events, write_ass, write_srt
from .text_matcher import ClipMatch, find_clip_match
from .transcriber import TranscriptSegment, save_transcript, transcribe_media
from .utils import ensure_parent, ffmpeg_filter_escape_path, find_executable, run_command, unique_path, write_json


ProgressCallback = Callable[[str, int, int], None]


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, step: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(stage, step, total)


def _render_thumbnail(media_path: Path, output_path: Path, timestamp: float) -> Path:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    ensure_parent(output_path)
    run_command(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(max(0.0, timestamp)),
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            str(output_path),
        ]
    )
    return output_path


def _render_short_video(
    media_path: Path,
    output_path: Path,
    start: float,
    end: float,
    subtitle_path: Path | None = None,
    fps: int = 30,
) -> Path:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    ensure_parent(output_path)
    filters = [
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
    ]
    if subtitle_path is not None:
        filters.append(f"ass='{ffmpeg_filter_escape_path(subtitle_path)}'")
    filter_chain = ",".join(filters)
    run_command(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(max(0.0, start)),
            "-t",
            str(max(1.0, end - start)),
            "-i",
            str(media_path),
            "-vf",
            filter_chain,
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return output_path


def generate_short_from_text(
    media_path: Path,
    target_text: str,
    title: str,
    description: str,
    output_dir: Path = SHORTS_DIR,
    model_name: str = "medium",
    language: str | None = None,
    device: str = "cpu",
    start_padding_seconds: float = DEFAULT_START_PADDING_SECONDS,
    end_padding_seconds: float = DEFAULT_END_PADDING_SECONDS,
    start_threshold: int = DEFAULT_MATCH_THRESHOLD,
    end_threshold: int = DEFAULT_MATCH_THRESHOLD,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, "transcribing", 1, 6)
    transcript = transcribe_media(media_path, model_name=model_name, language=language, device=device)
    segments: list[TranscriptSegment] = transcript["segments"]

    _emit_progress(progress_callback, "matching_excerpt", 2, 6)
    match: ClipMatch = find_clip_match(
        segments,
        target_text,
        start_threshold=start_threshold,
        end_threshold=end_threshold,
    )

    clip_start = max(0.0, match.start - start_padding_seconds)
    clip_end = max(clip_start + 1.0, match.end + end_padding_seconds)

    short_stem = unique_path(output_dir, "short", ".mp4").stem
    short_mp4 = output_dir / f"{short_stem}.mp4"
    short_ass = output_dir / f"{short_stem}.ass"
    short_srt = output_dir / f"{short_stem}.srt"
    short_meta = output_dir / f"{short_stem}_metadata.json"
    short_thumb = output_dir / f"{short_stem}_thumbnail_frame.jpg"
    transcript_path = output_dir / f"{short_stem}_transcript.json"

    _emit_progress(progress_callback, "building_subtitles", 3, 6)
    events = build_events(segments, clip_start, clip_end, max_words=6)
    write_ass(short_ass, events)
    write_srt(short_srt, events)

    _emit_progress(progress_callback, "rendering_video", 4, 6)
    _render_short_video(media_path, short_mp4, clip_start, clip_end, subtitle_path=short_ass)

    _emit_progress(progress_callback, "building_thumbnail", 5, 6)
    _render_thumbnail(media_path, short_thumb, (clip_start + clip_end) / 2)
    _emit_progress(progress_callback, "saving_metadata", 6, 6)
    save_transcript(transcript_path, transcript)

    payload = {
        "title": title,
        "description": description,
        "source_video": str(media_path),
        "start": clip_start,
        "end": clip_end,
        "device": transcript.get("device"),
        "requested_device": transcript.get("requested_device", device),
        "match_start_score": match.start_score,
        "match_end_score": match.end_score,
        "start_phrase": match.start_phrase,
        "end_phrase": match.end_phrase,
        "output_file": str(short_mp4),
        "subtitle_file": str(short_ass),
        "srt_file": str(short_srt),
        "thumbnail_file": str(short_thumb),
    }
    write_json(short_meta, payload)

    return {
        "video": short_mp4,
        "ass": short_ass,
        "srt": short_srt,
        "metadata": short_meta,
        "thumbnail": short_thumb,
        "transcript": transcript_path,
        "match": match,
        "payload": payload,
    }
