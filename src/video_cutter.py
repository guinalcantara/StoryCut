from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import (
    DEFAULT_END_PADDING_SECONDS,
    DEFAULT_MATCH_THRESHOLD,
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
    DEFAULT_START_PADDING_SECONDS,
    OUTPUTS_DIR,
    SUBTITLED_VIDEOS_DIR,
    SHORTS_DIR,
)
from .subtitle_generator import build_events, write_ass, write_srt
from .text_matcher import ClipMatch, find_clip_match
from .transcriber import TranscriptSegment, transcribe_media
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


def _subtitle_filter(subtitle_path: Path) -> str:
    escaped_path = ffmpeg_filter_escape_path(subtitle_path)
    suffix = subtitle_path.suffix.lower()
    if suffix == ".ass":
        return f"ass='{escaped_path}'"
    return f"subtitles='{escaped_path}'"


def _render_short_video(
    media_path: Path,
    output_path: Path,
    start: float,
    end: float,
    subtitle_path: Path | None = None,
    zoom_out: float = 0.0,
    fps: int = 30,
) -> Path:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    ensure_parent(output_path)
    zoom_out = max(0.0, min(float(zoom_out), 0.9))
    if zoom_out <= 0.0:
        filters = [
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[vbase]",
        ]
        if subtitle_path is not None:
            filters.append(f"[vbase]ass='{ffmpeg_filter_escape_path(subtitle_path)}'[vout]")
        else:
            filters.append("[vbase]null[vout]")
    else:
        foreground_height = max(2, int(round(1920 * (1.0 - zoom_out))))
        if foreground_height % 2 == 1:
            foreground_height -= 1
        filters = [
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,split=2[fgbase][bgsrc]",
            "[bgsrc]boxblur=20:1[bg]",
            f"[fgbase]crop=1080:{foreground_height}:0:(ih-{foreground_height})/2[fg]",
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]",
        ]
        if subtitle_path is not None:
            filters.append(f"[vbase]ass='{ffmpeg_filter_escape_path(subtitle_path)}'[vout]")
        else:
            filters.append("[vbase]null[vout]")
    filter_chain = ";".join(filters)
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
            "-filter_complex",
            filter_chain,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
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


def _render_video_with_subtitles(
    media_path: Path,
    output_path: Path,
    subtitle_path: Path,
    fps: int = 30,
) -> Path:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")
    ensure_parent(output_path)
    filter_chain = _subtitle_filter(subtitle_path)
    run_command(
        [
            ffmpeg,
            "-y",
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
    subtitle_color: str = DEFAULT_SHORTS_SUBTITLE_COLOR,
    subtitle_highlight_color: str = DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
    subtitle_font_size: int = DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
    subtitle_outline_size: int = DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
    subtitle_shadow_size: int = DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
    subtitle_margin_v: int = DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
    subtitle_spacing: float = DEFAULT_SHORTS_SUBTITLE_SPACING,
    subtitle_base_font: str = DEFAULT_SHORTS_SUBTITLE_BASE_FONT,
    subtitle_accent_font: str = DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT,
    subtitle_accent_last_word: bool = DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
    zoom_out: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, "transcribing", 1, 6)
    transcript = transcribe_media(
        media_path,
        model_name=model_name,
        language=language,
        device=device,
        progress_callback=progress_callback,
    )
    segments: list[TranscriptSegment] = transcript["segments"]
    transcript_path = Path(transcript["cache_path"])

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

    _emit_progress(progress_callback, "building_subtitles", 3, 6)
    events = build_events(segments, clip_start, clip_end, max_words=6)
    write_ass(
        short_ass,
        events,
        subtitle_color=subtitle_color,
        highlight_color=subtitle_highlight_color,
        font_size=subtitle_font_size,
        outline_size=subtitle_outline_size,
        shadow_size=subtitle_shadow_size,
        margin_v=subtitle_margin_v,
        spacing=subtitle_spacing,
        base_font=subtitle_base_font,
        accent_font=subtitle_accent_font,
        accent_last_word=subtitle_accent_last_word,
    )
    write_srt(short_srt, events)

    _emit_progress(progress_callback, "rendering_video", 4, 6)
    _render_short_video(
        media_path,
        short_mp4,
        clip_start,
        clip_end,
        subtitle_path=short_ass,
        zoom_out=zoom_out,
    )

    _emit_progress(progress_callback, "building_thumbnail", 5, 6)
    _render_thumbnail(media_path, short_thumb, (clip_start + clip_end) / 2)
    _emit_progress(progress_callback, "saving_metadata", 6, 6)

    payload = {
        "title": title,
        "description": description,
        "source_video": str(media_path),
        "start": clip_start,
        "end": clip_end,
        "device": transcript.get("device"),
        "requested_device": transcript.get("requested_device", device),
        "cache_hit": transcript.get("cache_hit", False),
        "match_start_score": match.start_score,
        "match_end_score": match.end_score,
        "start_phrase": match.start_phrase,
        "end_phrase": match.end_phrase,
        "subtitle_color": subtitle_color,
        "subtitle_highlight_color": subtitle_highlight_color,
        "subtitle_font_size": subtitle_font_size,
        "subtitle_outline_size": subtitle_outline_size,
        "subtitle_shadow_size": subtitle_shadow_size,
        "subtitle_margin_v": subtitle_margin_v,
        "subtitle_spacing": subtitle_spacing,
        "subtitle_base_font": subtitle_base_font,
        "subtitle_accent_font": subtitle_accent_font,
        "subtitle_accent_last_word": subtitle_accent_last_word,
        "zoom_out": zoom_out,
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


def generate_video_with_subtitles(
    media_path: Path,
    output_dir: Path = SUBTITLED_VIDEOS_DIR,
    model_name: str = "medium",
    language: str | None = None,
    device: str = "cpu",
    subtitle_path: Path | None = None,
    subtitle_color: str = DEFAULT_SHORTS_SUBTITLE_COLOR,
    subtitle_highlight_color: str = DEFAULT_SHORTS_SUBTITLE_HIGHLIGHT_COLOR,
    subtitle_font_size: int = DEFAULT_SHORTS_SUBTITLE_FONT_SIZE,
    subtitle_outline_size: int = DEFAULT_SHORTS_SUBTITLE_OUTLINE_SIZE,
    subtitle_shadow_size: int = DEFAULT_SHORTS_SUBTITLE_SHADOW_SIZE,
    subtitle_margin_v: int = DEFAULT_SHORTS_SUBTITLE_MARGIN_V,
    subtitle_spacing: float = DEFAULT_SHORTS_SUBTITLE_SPACING,
    subtitle_base_font: str = DEFAULT_SHORTS_SUBTITLE_BASE_FONT,
    subtitle_accent_font: str = DEFAULT_SHORTS_SUBTITLE_ACCENT_FONT,
    subtitle_accent_last_word: bool = DEFAULT_SHORTS_SUBTITLE_ACCENT_LAST_WORD,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_subtitle_path = subtitle_path
    transcript = None
    auto_generated = source_subtitle_path is None
    output_subtitle_ass = None
    output_subtitle_srt = None

    if auto_generated:
        _emit_progress(progress_callback, "transcribing", 1, 4)
        transcript = transcribe_media(
            media_path,
            model_name=model_name,
            language=language,
            device=device,
            progress_callback=progress_callback,
        )
        segments: list[TranscriptSegment] = transcript["segments"]
        subtitle_stem = unique_path(output_dir, "legenda", ".ass").stem
        output_subtitle_ass = output_dir / f"{subtitle_stem}.ass"
        output_subtitle_srt = output_dir / f"{subtitle_stem}.srt"
        _emit_progress(progress_callback, "building_subtitles", 2, 4)
        clip_end = max((float(segment.end) for segment in segments), default=float(transcript.get("duration") or 0.0))
        events = build_events(segments, 0.0, clip_end if clip_end > 0 else 0.0, max_words=6)
        write_ass(
            output_subtitle_ass,
            events,
            subtitle_color=subtitle_color,
            highlight_color=subtitle_highlight_color,
            font_size=subtitle_font_size,
            outline_size=subtitle_outline_size,
            shadow_size=subtitle_shadow_size,
            margin_v=subtitle_margin_v,
            spacing=subtitle_spacing,
            base_font=subtitle_base_font,
            accent_font=subtitle_accent_font,
            accent_last_word=subtitle_accent_last_word,
        )
        write_srt(output_subtitle_srt, events)
        source_subtitle_path = output_subtitle_ass
    elif source_subtitle_path is None:
        raise RuntimeError("Subtitle path was not provided and auto generation failed.")

    video_stem = unique_path(output_dir, "video_com_legenda", ".mp4").stem
    output_video = output_dir / f"{video_stem}.mp4"
    if not auto_generated:
        _emit_progress(progress_callback, "rendering_video", 1, 2)
    else:
        _emit_progress(progress_callback, "rendering_video", 3, 4)
    _render_video_with_subtitles(media_path, output_video, source_subtitle_path)

    if auto_generated:
        _emit_progress(progress_callback, "saving_metadata", 4, 4)
    else:
        _emit_progress(progress_callback, "saving_metadata", 2, 2)

    payload = {
        "source_video": str(media_path),
        "subtitle_source": "auto" if auto_generated else "uploaded",
        "subtitle_file": str(source_subtitle_path),
        "output_file": str(output_video),
        "subtitle_color": subtitle_color,
        "subtitle_highlight_color": subtitle_highlight_color,
        "subtitle_font_size": subtitle_font_size,
        "subtitle_outline_size": subtitle_outline_size,
        "subtitle_shadow_size": subtitle_shadow_size,
        "subtitle_margin_v": subtitle_margin_v,
        "subtitle_spacing": subtitle_spacing,
        "subtitle_base_font": subtitle_base_font,
        "subtitle_accent_font": subtitle_accent_font,
        "subtitle_accent_last_word": subtitle_accent_last_word,
    }
    if transcript is not None:
        payload["transcript_cache_path"] = transcript.get("cache_path")
        payload["requested_device"] = transcript.get("requested_device", device)
        payload["device"] = transcript.get("device")
        payload["cache_hit"] = transcript.get("cache_hit", False)
        payload["subtitle_srt_file"] = str(output_subtitle_srt) if output_subtitle_srt else None

    metadata_path = output_dir / f"{video_stem}_metadata.json"
    write_json(metadata_path, payload)

    result: dict[str, Any] = {
        "video": output_video,
        "metadata": metadata_path,
        "subtitle_file": source_subtitle_path,
        "payload": payload,
    }
    if output_subtitle_ass is not None:
        result["ass"] = output_subtitle_ass
    if output_subtitle_srt is not None:
        result["srt"] = output_subtitle_srt
    if transcript is not None:
        result["transcript"] = Path(transcript["cache_path"])
    return result
