# Reusable Audio Dataset Pipeline — Starter

This starter implements **Module 1: Audio ingestion and source tracking**. It inventories audio files, records technical metadata, computes a SHA-256 hash, and writes a CSV manifest. It does not modify, delete, convert, segment, or automatically label audio.

## Requirements

- Python 3.11+
- FFmpeg (the `ffprobe` command must be available on PATH)

The first module uses only Python's standard library, so no `pip install` is required yet.

## Folder structure

```text
audio_dataset_pipeline_starter/
├── data/
│   ├── incoming/
│   │   ├── genuine/
│   │   ├── synthetic/
│   │   └── source_metadata.csv   # optional
│   └── metadata/
├── reports/
└── src/
    └── ingest.py
```

Put genuine recordings under `data/incoming/genuine/` and AI-generated recordings under `data/incoming/synthetic/`. Subfolders are allowed.

Recognized label folder aliases include `real`/`authentic` and `fake`/`ai_generated`, but using `genuine` and `synthetic` consistently is recommended. The code does not infer labels from the sound itself. Files outside recognized label folders are marked `unknown` for review, not silently assigned a class.

## Run it in VS Code

Open this folder in VS Code, then use Terminal → New Terminal.

Windows:
```powershell
python src/ingest.py
```

macOS/Linux:
```bash
python3 src/ingest.py
```

If `ffprobe` is installed but not on PATH:
```bash
python src/ingest.py --ffprobe "C:\path\to\ffprobe.exe"
```

Outputs:
- `data/metadata/ingestion_manifest.csv`
- `reports/ingestion_errors.csv`

Unsupported file extensions are recorded in the manifest. Files that fail probing are recorded in the error report. No audio is deleted or changed.

## Optional source metadata CSV

Create `data/incoming/source_metadata.csv` with these columns:

```csv
filepath,source_name,speaker_id,language,license,notes
genuine/interview_01.wav,volunteer_01,spk_001,Malayalam,consent_form_01,quiet room
synthetic/tts_01.wav,IndicTTS,tts_voice_01,Malayalam,model_license_checked,TTS generated from approved text
```

`filepath` is relative to `data/incoming/`, and must match the path and spelling exactly. Keep speaker IDs pseudonymous; do not put names or sensitive personal information in this file. Only record a license or consent reference if you have actually checked/obtained it.

## Design notes

- `duration_seconds`, sample rate, channel count, codec and container format come from `ffprobe`.
- `sha256` helps identify exact duplicate source files.
- This module does not decide whether quiet speech is valid. It does not filter by loudness.
- Next module: standardize audio to a consistent WAV format, followed by speech-aware segmentation (3–8 seconds), quality flags, and feature extraction.
