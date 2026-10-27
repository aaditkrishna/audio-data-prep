#!/usr/bin/env python3
"""
Module 1: Audio ingestion and source tracking.

Scans:
  data/incoming/genuine/
  data/incoming/synthetic/

Writes:
  data/metadata/ingestion_manifest.csv
  reports/ingestion_errors.csv

This module inventories files only. It does NOT cut, convert, delete, or relabel audio.
FFmpeg's ffprobe command is used to read technical audio metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"
}
LABEL_FOLDERS = {
    "genuine": "genuine",
    "real": "genuine",
    "authentic": "genuine",
    "synthetic": "synthetic",
    "fake": "synthetic",
    "ai_generated": "synthetic",
}
MANIFEST_FIELDS = [
    "record_id", "relative_path", "absolute_path", "label", "label_source",
    "source_name", "speaker_id", "language", "license", "notes",
    "file_extension", "file_size_bytes", "sha256", "duration_seconds",
    "sample_rate_hz", "channels", "codec_name", "format_name",
    "ingested_at_utc", "status", "error"
]
SIDECAR_FIELDS = ["filepath", "source_name", "speaker_id", "language", "license", "notes"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sidecar(path: Path) -> dict[str, dict[str, str]]:
    """Optional CSV keyed by filepath relative to data/incoming/."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "filepath" not in reader.fieldnames:
            raise ValueError(
                f"{path} must have a 'filepath' column. Optional columns: "
                + ", ".join(SIDECAR_FIELDS[1:])
            )
        for row in reader:
            key = (row.get("filepath") or "").strip().replace("\\", "/")
            if not key:
                continue
            result[key] = {field: (row.get(field) or "").strip() for field in SIDECAR_FIELDS[1:]}
    return result


def probe_audio(path: Path, ffprobe_bin: str) -> dict[str, Any]:
    command = [
        ffprobe_bin, "-v", "error",
        "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path)
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffprobe could not read this file"
        raise RuntimeError(message)
    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    fmt = data.get("format", {})
    audio_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"),
        None
    )
    if not audio_stream:
        raise RuntimeError("No audio stream found")
    try:
        duration = float(fmt["duration"])
    except (KeyError, TypeError, ValueError):
        duration = None

    return {
        "duration_seconds": duration,
        "sample_rate_hz": audio_stream.get("sample_rate", ""),
        "channels": audio_stream.get("channels", ""),
        "codec_name": audio_stream.get("codec_name", ""),
        "format_name": fmt.get("format_name", ""),
    }


def find_label(path: Path, incoming_root: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(incoming_root)
    except ValueError:
        return "unknown", "not_in_input_root"
    if not relative.parts:
        return "unknown", "not_in_label_folder"
    folder = relative.parts[0].casefold()
    if folder in LABEL_FOLDERS:
        return LABEL_FOLDERS[folder], "folder_name"
    return "unknown", "unrecognized_folder"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ingest(project_root: Path, ffprobe_bin: str = "ffprobe") -> tuple[int, int]:
    incoming = project_root / "data" / "incoming"
    manifest_path = project_root / "data" / "metadata" / "ingestion_manifest.csv"
    errors_path = project_root / "reports" / "ingestion_errors.csv"
    sidecar_path = incoming / "source_metadata.csv"

    if not incoming.exists():
        raise FileNotFoundError(f"Input directory does not exist: {incoming}")

    sidecar = load_sidecar(sidecar_path)
    manifest_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    files = sorted(
        p for p in incoming.rglob("*")
        if p.is_file() and p.name.casefold() not in {"source_metadata.csv", "source_metadata_template.csv"}
    )

    for path in files:
        ext = path.suffix.casefold()
        relative = path.relative_to(incoming).as_posix()
        label, label_source = find_label(path, incoming)
        side = sidecar.get(relative, {})
        record_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        row: dict[str, Any] = {
            "record_id": record_id,
            "relative_path": relative,
            "absolute_path": str(path.resolve()),
            "label": label,
            "label_source": label_source,
            "source_name": side.get("source_name", ""),
            "speaker_id": side.get("speaker_id", ""),
            "language": side.get("language", ""),
            "license": side.get("license", ""),
            "notes": side.get("notes", ""),
            "file_extension": ext,
            "file_size_bytes": path.stat().st_size,
            "sha256": "",
            "duration_seconds": "",
            "sample_rate_hz": "",
            "channels": "",
            "codec_name": "",
            "format_name": "",
            "ingested_at_utc": timestamp,
            "status": "pending",
            "error": "",
        }

        if ext not in AUDIO_EXTENSIONS:
            row["status"] = "skipped_unsupported_extension"
            row["error"] = "Extension is not in the supported audio extension list"
            manifest_rows.append(row)
            continue

        try:
            row["sha256"] = sha256_file(path)
            row.update(probe_audio(path, ffprobe_bin))
            row["status"] = "ok" if label != "unknown" else "needs_label_review"
            if label == "unknown":
                row["error"] = "Could not infer label from folder; review and label explicitly"
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            error_rows.append({
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "error": str(exc),
                "checked_at_utc": timestamp,
            })
        manifest_rows.append(row)

    write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    write_csv(errors_path, error_rows, ["relative_path", "absolute_path", "error", "checked_at_utc"])

    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    review_count = sum(1 for row in manifest_rows if row["status"] == "needs_label_review")
    error_count = sum(1 for row in manifest_rows if row["status"] == "error")
    unsupported_count = sum(1 for row in manifest_rows if row["status"] == "skipped_unsupported_extension")

    print(f"Scanned files: {len(manifest_rows)}")
    print(f"Ready with known label: {ok_count}")
    print(f"Needs label review: {review_count}")
    print(f"Read errors: {error_count}")
    print(f"Unsupported extensions: {unsupported_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Errors:   {errors_path}")
    print("No audio files were modified or deleted.")
    return len(manifest_rows), error_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory audio files and preserve source metadata.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root (default: parent of src/)")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable name or path")
    args = parser.parse_args()
    try:
        ingest(args.root.resolve(), args.ffprobe)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
