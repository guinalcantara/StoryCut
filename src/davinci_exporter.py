from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DAVINCI_DIR
from .utils import ensure_parent, write_csv, write_json


def export_davinci_bundle(
    audio_wav: Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Path]:
    DAVINCI_DIR.mkdir(parents=True, exist_ok=True)
    target_wav = DAVINCI_DIR / "audio_limpo.wav"
    target_csv = DAVINCI_DIR / "cortes_audio.csv"
    target_json = DAVINCI_DIR / "relatorio_limpeza.json"
    ensure_parent(target_wav)
    target_wav.write_bytes(audio_wav.read_bytes())
    write_csv(target_csv, rows, fieldnames=["type", "start", "end", "action", "reason"])
    write_json(target_json, report)
    return {
        "wav": target_wav,
        "csv": target_csv,
        "json": target_json,
    }

