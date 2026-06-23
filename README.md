# StoryCut

StoryCut is a local Streamlit app for two editing workflows:

1. Clean narrated audio for DaVinci Resolve.
2. Generate vertical Shorts from a long video and a text excerpt.

## What it does

- Transcribes audio or video with Faster-Whisper.
- Detects silences with FFmpeg.
- Detects simple repeated takes with fuzzy matching.
- Cuts audio and exports reports.
- Finds text passages in a transcript and renders vertical Shorts with subtitles.
- Starts with a presentation home screen.
- Supports Portuguese and English UI labels, with Portuguese as the default.
- Accepts uploads up to 1 GB through Streamlit's upload limit.

## Project layout

```txt
storycut/
  app.py
  src/
  data/
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Run locally

Install the Python dependencies, then run Streamlit:

```bash
streamlit run app.py
```

## Run with Docker

```bash
docker compose up --build
```

## Notes

- Outputs are written under `data/outputs`.
- Temporary uploads and transcriptions are stored under `data/`.
- The first version focuses on a working end-to-end flow, not perfect matching.
- Streamlit upload size is configured in `.streamlit/config.toml`.
