
#!/usr/bin/env python3
"""Incremental audio ingestion. Never modifies or deletes source audio."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac",
    ".ogg", ".opus", ".wma", ".aiff", ".aif"
}

LABEL_FOLDERS = {
    "genuine": "genuine", "real": "genuine", "authentic": "genuine",
    "synthetic": "synthetic", "fake": "synthetic",
    "ai_generated": "synthetic"
}

FIELDS = [
    "record_id", "relative_path", "absolute_path", "label",
    "label_source", "source_name", "speaker_id", "language",
    "license", "notes", "file_extension", "file_size_bytes",
    "sha256", "duration_seconds", "sample_rate_hz", "channels",
    "codec_name", "format_name", "ingested_at_utc", "status",
    "error", "duplicate_of", "present_in_incoming"
]

TECHNICAL_FIELDS = [
    "duration_seconds", "sample_rate_hz", "channels",
    "codec_name", "format_name"
]

SIDECAR_FIELDS = [
    "source_name", "speaker_id", "language", "license", "notes"
]

SUCCESS = {"ok", "needs_label_review"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probe_audio(path, ffprobe):
    command = [
        ffprobe, "-v", "error",
        "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")

    data = json.loads(result.stdout)
    streams = [
        s for s in data.get("streams", [])
        if s.get("codec_type") == "audio"
    ]
    if not streams:
        raise ValueError("No audio stream found")

    fmt = data.get("format", {})
    return {
        "duration_seconds": fmt.get("duration", ""),
        "sample_rate_hz": streams[0].get("sample_rate", ""),
        "channels": streams[0].get("channels", ""),
        "codec_name": streams[0].get("codec_name", ""),
        "format_name": fmt.get("format_name", "")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    root = args.root.resolve()
    incoming = root / "data" / "incoming"
    manifest = root / "data" / "metadata" / "ingestion_manifest.csv"
    errors_file = root / "reports" / "ingestion_errors.csv"

    if not incoming.is_dir():
        print(f"ERROR: Input directory not found: {incoming}", file=sys.stderr)
        return 1

    if shutil.which(args.ffprobe) is None and not Path(args.ffprobe).is_file():
        print(f"ERROR: ffprobe not found: {args.ffprobe}", file=sys.stderr)
        return 1

    previous_rows = read_csv(manifest)
    previous = {
        r["relative_path"]: r
        for r in previous_rows if r.get("relative_path")
    }

    sidecar_path = incoming / "source_metadata.csv"
    sidecar = {}
    for r in read_csv(sidecar_path):
        key = (r.get("filepath") or "").strip().replace("\\", "/")
        if key:
            sidecar[key] = r

    files = sorted(
        p for p in incoming.rglob("*")
        if p.is_file() and p.name.casefold() not in {
            "source_metadata.csv", "source_metadata_template.csv"
        }
    )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows_by_path = {}
    errors = []
    unchanged = 0
    probed = 0

    for path in files:
        relative = path.relative_to(incoming).as_posix()
        ext = path.suffix.lower()
        old = previous.get(relative, {})
        label_folder = relative.split("/")[0].casefold()
        label = LABEL_FOLDERS.get(label_folder, "unknown")
        metadata = sidecar.get(relative, {})

        row = {
            field: old.get(field, "") for field in FIELDS
        }
        row.update({
            "record_id": hashlib.sha256(relative.encode()).hexdigest()[:16],
            "relative_path": relative,
            "absolute_path": str(path.resolve()),
            "label": label,
            "label_source": "folder_name" if label != "unknown" else "unrecognized_folder",
            "file_extension": ext,
            "file_size_bytes": path.stat().st_size,
            "ingested_at_utc": old.get("ingested_at_utc") or now,
            "source_name": metadata.get("source_name", ""),
            "speaker_id": metadata.get("speaker_id", ""),
            "language": metadata.get("language", ""),
            "license": metadata.get("license", ""),
            "notes": metadata.get("notes", ""),
            "duplicate_of": "",
            "present_in_incoming": "true",
            "error": ""
        })

        if ext not in AUDIO_EXTENSIONS:
            row["status"] = "skipped_unsupported_extension"
            row["error"] = "Unsupported audio extension"
            rows_by_path[relative] = row
            continue

        try:
            # Hash each file so changes are detected even if its name stays the same.
            digest = sha256_file(path)
            row["sha256"] = digest

            same_content = (
                old.get("sha256") == digest
                and old.get("status") in SUCCESS
            )

            if same_content:
                # Reuse technical metadata; skip ffprobe.
                for field in TECHNICAL_FIELDS:
                    row[field] = old.get(field, "")
                unchanged += 1
            else:
                row.update(probe_audio(path, args.ffprobe))
                probed += 1

            row["status"] = (
                "ok" if label != "unknown" else "needs_label_review"
            )
            if label == "unknown":
                row["error"] = "Review label: folder name is not recognized"

        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            errors.append({
                "relative_path": relative,
                "absolute_path": str(path.resolve()),
                "error": str(exc),
                "checked_at_utc": now
            })

        rows_by_path[relative] = row

    current_paths = set(rows_by_path)

    # Retain records for previously known files no longer in the input folder.
    for relative, old in previous.items():
        if relative in current_paths:
            continue
        row = {field: old.get(field, "") for field in FIELDS}
        row["status"] = "not_found"
        row["present_in_incoming"] = "false"
        row["duplicate_of"] = ""
        row["error"] = "File is no longer present in data/incoming"
        rows_by_path[relative] = row

    # Flag duplicate content without deleting either file.
    first_for_hash = {}
    duplicates = 0
    for relative in sorted(rows_by_path):
        row = rows_by_path[relative]
        if row.get("present_in_incoming") != "true":
            continue
        if row.get("status") not in SUCCESS:
            continue
        digest = row.get("sha256", "")
        if not digest:
            continue
        if digest in first_for_hash:
            row["duplicate_of"] = first_for_hash[digest]
            duplicates += 1
        else:
            first_for_hash[digest] = relative

    rows = [rows_by_path[k] for k in sorted(rows_by_path)]
    write_csv(manifest, rows, FIELDS)
    write_csv(
        errors_file, errors,
        ["relative_path", "absolute_path", "error", "checked_at_utc"]
    )

    print(f"Files scanned: {len(files)}")
    print(f"Successful files: {sum(r['status'] == 'ok' for r in rows)}")
    print(f"Needs label review: {sum(r['status'] == 'needs_label_review' for r in rows)}")
    print(f"Errors: {sum(r['status'] == 'error' for r in rows)}")
    print(f"Unchanged files (ffprobe skipped): {unchanged}")
    print(f"Files inspected with ffprobe: {probed}")
    print(f"Duplicate files flagged: {duplicates}")
    print(f"Manifest: {manifest}")
    print(f"Errors report: {errors_file}")
    print("Original audio files were not modified or deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
