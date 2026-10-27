
#!/usr/bin/env python3
"""Incremental audio standardization without overwriting existing outputs."""

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

FIELDS = [
    "source_path",
    "output_path",
    "source_sha256",
    "output_sha256",
    "source_codec",
    "source_sample_rate",
    "source_channels",
    "duration_seconds",
    "output_format",
    "output_sample_rate",
    "output_channels",
    "status",
    "last_processed_utc",
    "present_in_input",
    "error",
]

TECHNICAL_FIELDS = [
    "source_codec",
    "source_sample_rate",
    "source_channels",
    "duration_seconds",
    "output_format",
    "output_sample_rate",
    "output_channels",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_ffprobe(path, ffprobe):
    command = [
        ffprobe, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:"
        "format=duration",
        "-of", "json", str(path),
    ]

    result = subprocess.run(
        command, capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"FFprobe failed for {path}"
        )

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    if not streams:
        raise ValueError(f"No readable audio stream: {path}")

    stream = streams[0]

    return {
        "codec": stream.get("codec_name", ""),
        "sample_rate": str(stream.get("sample_rate", "")),
        "channels": str(stream.get("channels", "")),
        "duration_seconds": data.get("format", {}).get("duration", ""),
    }


def choose_destination(base_path, source_hash):
    """Choose a free output path without overwriting an existing file."""

    if not base_path.exists():
        return base_path

    versioned = base_path.with_name(
        f"{base_path.stem}__{source_hash[:8]}{base_path.suffix}"
    )

    if not versioned.exists():
        return versioned

    counter = 2
    while True:
        candidate = base_path.with_name(
            f"{base_path.stem}__{source_hash[:8]}_{counter}"
            f"{base_path.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1

def find_matching_existing_output(base_path, source_hash, expected_output_hash):
    """Find an existing output matching the previously verified output hash."""

    candidates = [base_path]

    hash_path = base_path.with_name(
        f"{base_path.stem}__{source_hash[:8]}{base_path.suffix}"
    )
    candidates.append(hash_path)

    # Include older versioned files, such as files ending in _2.
    if base_path.parent.exists():
        candidates.extend(sorted(
            base_path.parent.glob(
                f"{base_path.stem}__{source_hash[:8]}*.wav"
            )
        ))

    seen = set()

    for candidate in candidates:
        key = str(candidate.resolve()).lower()

        if key in seen:
            continue
        seen.add(key)

        if not candidate.is_file():
            continue

        if sha256_file(candidate) == expected_output_hash:
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally standardize audio to mono 16 kHz WAV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    root = args.root.resolve()
    input_dir = root / "data" / "incoming"
    output_dir = root / "data" / "processed"
    manifest_path = root / "data" / "metadata" / "standardization_manifest.csv"
    errors_path = root / "reports" / "standardization_errors.csv"

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    for executable in (args.ffmpeg, args.ffprobe):
        if shutil.which(executable) is None and not Path(executable).is_file():
            print(f"ERROR: Executable not found: {executable}", file=sys.stderr)
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    previous_rows = read_csv(manifest_path)
    previous = {
        row["source_path"].replace("\\", "/"): row
        for row in previous_rows
        if row.get("source_path")
    }

    sources = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows_by_source = {}
    errors = []

    converted = 0
    skipped = 0
    failed = 0
    unchanged_source_count = 0

    for source in sources:
        source_path = source.relative_to(root).as_posix()
        relative_input = source.relative_to(input_dir)
        base_output = (output_dir / relative_input).with_suffix(".wav")
        old = previous.get(source_path, {})

        row = {field: old.get(field, "") for field in FIELDS}
        row.update({
            "source_path": source_path,
            "present_in_input": "true",
            "error": "",
        })

        temp_path = None

        try:
            source_hash = sha256_file(source)
            row["source_sha256"] = source_hash

            old_output_text = old.get("output_path", "")
            old_output = (
                root / old_output_text
                if old_output_text else None
            )

            
            # Skip only when the source and a recorded output verify.
            if (
                old.get("source_sha256") == source_hash
                and old.get("status") in {"success", "skipped_unchanged"}
                and old.get("output_sha256")
            ):
                matching_output = find_matching_existing_output(
                    base_output,
                    source_hash,
                    old["output_sha256"],
                )

                if matching_output is not None:
                    for field in TECHNICAL_FIELDS:
                        row[field] = old.get(field, "")

                    # Prefer the canonical filename when its contents match.
                    row["output_path"] = (
                        matching_output.relative_to(root).as_posix()
                    )
                    row["output_sha256"] = old["output_sha256"]
                    row["status"] = "skipped_unchanged"
                    row["last_processed_utc"] = (
                        old.get("last_processed_utc") or now
                    )

                    rows_by_source[source_path] = row
                    skipped += 1
                    unchanged_source_count += 1
                    continue


            source_info = run_ffprobe(source, args.ffprobe)

            # Keep a prior output path if it is missing and the source
            # hash is unchanged. Otherwise choose a new, unused path.
            if (
                old_output is not None
                and not old_output.exists()
                and old.get("source_sha256") == source_hash
            ):
                destination = old_output
            else:
                destination = choose_destination(base_output, source_hash)

            destination.parent.mkdir(parents=True, exist_ok=True)

            # Temporary output is unique and ends in .wav for FFmpeg.
            temp_path = destination.with_name(
                f"{destination.stem}.__temp_{source_hash[:8]}.wav"
            )

            suffix = 2
            while temp_path.exists():
                temp_path = destination.with_name(
                    f"{destination.stem}.__temp_{source_hash[:8]}_{suffix}.wav"
                )
                suffix += 1

            command = [
                args.ffmpeg,
                "-nostdin", "-hide_banner", "-loglevel", "error",
                "-n", "-i", str(source),
                "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(temp_path),
            ]

            result = subprocess.run(
                command, capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr.strip() or "FFmpeg conversion failed"
                )

            output_info = run_ffprobe(temp_path, args.ffprobe)

            if (
                output_info["sample_rate"] != "16000"
                or output_info["channels"] != "1"
                or output_info["codec"] != "pcm_s16le"
            ):
                raise ValueError(
                    f"Output validation failed: {output_info}"
                )

            # Do not overwrite anything that appeared during processing.
            if destination.exists():
                destination = choose_destination(base_output, source_hash)
                while destination.exists():
                    destination = choose_destination(base_output, source_hash)

            # Publish the completed temporary file only after validation.
            temp_path.rename(destination)
            temp_path = None

            row.update({
                "output_path": destination.relative_to(root).as_posix(),
                "source_sha256": source_hash,
                "output_sha256": sha256_file(destination),
                "source_codec": source_info["codec"],
                "source_sample_rate": source_info["sample_rate"],
                "source_channels": source_info["channels"],
                "duration_seconds": output_info["duration_seconds"],
                "output_format": "WAV PCM 16-bit",
                "output_sample_rate": 16000,
                "output_channels": 1,
                "status": "success",
                "last_processed_utc": now,
                "error": "",
            })
            converted += 1

        except Exception as exc:
            row["status"] = "error"
            row["last_processed_utc"] = now
            row["error"] = str(exc)
            errors.append({
                "source_path": source_path,
                "error": str(exc),
                "checked_at_utc": now,
            })
            failed += 1

        finally:
            # Only remove the temporary file created by this run.
            # Never delete a final or pre-existing output here.
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        rows_by_source[source_path] = row

    current_sources = set(rows_by_source)

    # Preserve previous manifest records for sources no longer present.
    for source_path, old in previous.items():
        if source_path in current_sources:
            continue

        row = {field: old.get(field, "") for field in FIELDS}
        row["status"] = "source_missing"
        row["present_in_input"] = "false"
        row["error"] = "Source is no longer present in data/incoming"
        rows_by_source[source_path] = row

    rows = [rows_by_source[key] for key in sorted(rows_by_source)]
    write_csv(manifest_path, rows)

    with errors_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_path", "error", "checked_at_utc"],
        )
        writer.writeheader()
        writer.writerows(errors)

    print(f"Audio files converted: {converted}")
    print(f"Unchanged files skipped: {skipped}")
    print(f"Conversion errors: {failed}")
    print(f"Previously known sources now missing: "
          f"{sum(r['status'] == 'source_missing' for r in rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Errors report: {errors_path}")
    print("Original audio and existing final outputs were not deleted.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
