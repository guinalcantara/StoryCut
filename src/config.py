from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "StoryCut"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
OUTPUTS_DIR = DATA_DIR / "outputs"
TEMP_DIR = DATA_DIR / "temp"

DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_DUPLICATE_THRESHOLD = 87
DEFAULT_MATCH_THRESHOLD = 80
DEFAULT_SILENCE_THRESHOLD_DB = -40
DEFAULT_MIN_SILENCE_DURATION_MS = 700
DEFAULT_PRE_SPEECH_PADDING_MS = 150
DEFAULT_POST_SPEECH_PADDING_MS = 250
DEFAULT_START_PADDING_SECONDS = 0.35
DEFAULT_END_PADDING_SECONDS = 0.50

SHORTS_DIR = OUTPUTS_DIR / "shorts"
DAVINCI_DIR = OUTPUTS_DIR / "davinci"


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        UPLOADS_DIR,
        TRANSCRIPTIONS_DIR,
        OUTPUTS_DIR,
        TEMP_DIR,
        SHORTS_DIR,
        DAVINCI_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

