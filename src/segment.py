
from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


# Project paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_ROOT = ROOT / "data" / "processed"
OUTPUT_ROOT = ROOT / "data" / "segmented"
MANIFEST_PATH = ROOT / "data" / "metadata" / "segmentation_manifest.csv"
STANDARDIZATION_MANIFEST_PATH = (
    ROOT / "data" / "metadata" / "standardization_manifest.csv"
)
ERRORS_PATH = ROOT / "reports" / "segmentation_errors.csv"

SAMPLE_RATE = 16000
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 8.0
MAX_GAP_SECONDS = 1.0

MANIFEST_FIELDS = [
    "source_path",
    "source_sha256",
    "source_id",
    "label",
    "language",
    "clip_id",
    "clip_path",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "sample_rate",
    "channels",
    "output_sha256",
    "status",
    "processed_at",
]

ERROR_FIELDS = ["source_path", "error", "processed_at"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
    ) as temp_file:
        writer = csv.DictWriter(
            temp_file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        temp_name = temp_file.name

    os.replace(temp_name, path)


def infer_label_language(source: Path) -> tuple[str, str]:
    relative = source.relative_to(INPUT_ROOT)
    parts = relative.parts

    if len(parts) < 2:
        raise ValueError(
            "Expected data/processed/<label>/<language>/<file>.wav"
        )

    label = parts[0].lower()

    if label not in {"genuine", "synthetic"}:
        raise ValueError(
            f"Unknown class folder: {parts[0]}"
        )

    language = parts[1].lower()

    return label, language


def make_clip_ranges(
    intervals: list[tuple[int, int]],
    total_samples: int,
) -> list[tuple[int, int]]:
    """Combine nearby speech regions into clips lasting 3–8 seconds."""

    min_samples = int(MIN_CLIP_SECONDS * SAMPLE_RATE)
    max_samples = int(MAX_CLIP_SECONDS * SAMPLE_RATE)
    max_gap = int(MAX_GAP_SECONDS * SAMPLE_RATE)

    # Split any speech interval that exceeds eight seconds.
    pieces = []

    for start, end in intervals:
        start = max(0, min(int(start), total_samples))
        end = max(start, min(int(end), total_samples))

        while end - start > max_samples:
            pieces.append((start, start + max_samples))
            start += max_samples

        if end > start:
            pieces.append((start, end))

    # Merge nearby speech intervals when the resulting clip
    # will not exceed the maximum duration.
    groups = []

    for start, end in pieces:
        if not groups:
            groups.append((start, end))
            continue

        previous_start, previous_end = groups[-1]
        gap = start - previous_end
        combined_duration = end - previous_start

        if (
            0 <= gap <= max_gap
            and combined_duration <= max_samples
        ):
            groups[-1] = (previous_start, end)
        else:
            groups.append((start, end))

    # Merge short fragments with adjacent speech where possible.
    cleaned = []
    i = 0

    while i < len(groups):
        start, end = groups[i]

        if end - start >= min_samples:
            cleaned.append((start, end))
            i += 1
            continue

        if i + 1 < len(groups):
            next_start, next_end = groups[i + 1]

            if (
                next_end - start <= max_samples
                and 0 <= next_start - end <= max_gap
            ):
                groups[i + 1] = (start, next_end)
                i += 1
                continue

        if cleaned:
            previous_start, previous_end = cleaned[-1]

            if (
                end - previous_start <= max_samples
                and 0 <= start - previous_end <= max_gap
            ):
                cleaned[-1] = (previous_start, end)

        # Otherwise, discard the short fragment.
        i += 1

    return [
        (start, end)
        for start, end in cleaned
        if min_samples <= end - start <= max_samples
    ]


def source_is_unchanged(
    previous_rows: list[dict[str, str]],
    source_hash: str,
) -> bool:
    matching = [
        row for row in previous_rows
        if row.get("source_sha256") == source_hash
        and row.get("status") in {"ok", "no_speech"}
    ]

    if not matching:
        return False

    # Verify that all recorded clips still exist and are unchanged.
    for row in matching:
        if row["status"] == "no_speech":
            continue

        clip_path = Path(row.get("clip_path", ""))

        if not clip_path.is_file():
            return False

        expected_hash = row.get("output_sha256", "")

        if not expected_hash or sha256_file(clip_path) != expected_hash:
            return False

    return True


def main() -> None:
    if not INPUT_ROOT.exists():
        raise SystemExit(
            f"Input folder not found: {INPUT_ROOT}\n"
            "Run standardize.py first."
        )

    if not STANDARDIZATION_MANIFEST_PATH.is_file():
        raise SystemExit(
            f"Standardization manifest not found: {STANDARDIZATION_MANIFEST_PATH}\n"
            "Run src/standardize.py first."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Read only the WAV outputs explicitly recorded by standardize.py.
    standardization_rows = read_csv(STANDARDIZATION_MANIFEST_PATH)
    sources = []
    for item in standardization_rows:
        if item.get("status") not in {"success", "skipped_unchanged"}:
            continue

        output_text = item.get("output_path", "").strip()
        if not output_text:
            continue

        source = (ROOT / output_text).resolve()
        if not source.is_file():
            raise SystemExit(
                f"Manifest-listed WAV does not exist: {source}\n"
                "Check standardization_manifest.csv before segmenting."
            )

        expected_hash = item.get("output_sha256", "").strip().lower()
        if expected_hash and sha256_file(source).lower() != expected_hash:
            raise SystemExit(
                f"Manifest hash does not match WAV file: {source}\n"
                "Run src/standardize.py and inspect its manifest first."
            )

        if source.suffix.lower() != ".wav":
            raise SystemExit(f"Manifest output is not a WAV file: {source}")

        sources.append(source)

    sources = sorted(set(sources), key=lambda path: str(path).lower())

    if not sources:
        print(
            "No valid WAV outputs found in standardization_manifest.csv. "
            "Run src/standardize.py first."
        )
        return

    # Keep old segmentation rows only for the canonical source paths above.
    # This removes duplicate-source rows from the new manifest without
    # deleting any existing clip files.
    allowed_source_keys = {normalize_path(path) for path in sources}
    old_rows = read_csv(MANIFEST_PATH)
    previous_by_source = {}

    for row in old_rows:
        source_path = row.get("source_path", "")
        if source_path and normalize_path(source_path) in allowed_source_keys:
            key = normalize_path(source_path)
            previous_by_source.setdefault(key, []).append(row)

    output_rows = [
        row for row in old_rows
        if normalize_path(row.get("source_path", "")) in allowed_source_keys
    ]
    current_errors = []

    print("Loading Silero VAD model...")
    model = load_silero_vad()
    model.eval()

    segmented_count = 0
    skipped_count = 0
    no_speech_count = 0
    error_count = 0

    for source in sources:
        source_key = normalize_path(source)

        try:
            label, language = infer_label_language(source)
            source_hash = sha256_file(source)

            source_id = hashlib.sha256(
                source_key.encode("utf-8")
            ).hexdigest()[:16]

            previous = previous_by_source.get(source_key, [])

            if source_is_unchanged(previous, source_hash):
                skipped_count += 1
                print(f"SKIP unchanged: {source.name}")
                continue

            # Read standardized audio.
            audio, sample_rate = sf.read(
                source,
                dtype="float32",
                always_2d=True,
            )

            if sample_rate != SAMPLE_RATE:
                raise ValueError(
                    f"Expected 16000 Hz; found {sample_rate} Hz."
                )

            if audio.shape[1] != 1:
                raise ValueError(
                    f"Expected mono audio; found {audio.shape[1]} channels."
                )

            mono = np.nan_to_num(
                audio[:, 0],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            waveform = torch.from_numpy(
                np.ascontiguousarray(mono)
            )

            # Detect speech regions.
            timestamps = get_speech_timestamps(
                waveform,
                model,
                sampling_rate=SAMPLE_RATE,
                return_seconds=False,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            )

            intervals = [
                (int(item["start"]), int(item["end"]))
                for item in timestamps
            ]

            clip_ranges = make_clip_ranges(
                intervals,
                len(mono),
            )

            # Replace manifest entries for this source only.
            output_rows = [
                row for row in output_rows
                if normalize_path(row.get("source_path", ""))
                != source_key
            ]

            common = {
                "source_path": str(source.resolve()),
                "source_sha256": source_hash,
                "source_id": source_id,
                "label": label,
                "language": language,
                "sample_rate": str(SAMPLE_RATE),
                "channels": "1",
                "processed_at": now_utc(),
            }

            if not clip_ranges:
                output_rows.append({
                    **common,
                    "clip_id": "",
                    "clip_path": "",
                    "start_seconds": "",
                    "end_seconds": "",
                    "duration_seconds": "",
                    "output_sha256": "",
                    "status": "no_speech",
                })

                no_speech_count += 1
                print(f"NO SUITABLE CLIPS: {source.name}")
                continue

            # Use a source-hash directory so changed inputs
            # do not overwrite outputs from earlier runs.
            safe_stem = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                source.stem,
            ).strip("._") or "audio"

            source_dir = (
                OUTPUT_ROOT
                / label
                / language
                / f"{safe_stem}__{source_hash[:8]}"
            )
            source_dir.mkdir(parents=True, exist_ok=True)

            for number, (start, end) in enumerate(
                clip_ranges,
                start=1,
            ):
                clip_id = f"{source_id}_clip_{number:05d}"
                clip_path = source_dir / f"clip_{number:05d}.wav"

                clip_audio = mono[start:end]

                # Do not overwrite an unexpected existing clip.
                if clip_path.exists():
                    existing, existing_rate = sf.read(
                        clip_path,
                        dtype="float32",
                        always_2d=True,
                    )

                    matches = (
                        existing_rate == SAMPLE_RATE
                        and existing.shape == (len(clip_audio), 1)
                        and np.array_equal(existing[:, 0], clip_audio)
                    )

                    if not matches:
                        raise FileExistsError(
                            "Existing clip differs from expected audio; "
                            f"refusing to overwrite: {clip_path}"
                        )
                else:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False,
                        dir=source_dir,
                    ) as temp_file:
                        temp_path = Path(temp_file.name)

                    try:
                        sf.write(
                            temp_path,
                            clip_audio,
                            SAMPLE_RATE,
                            subtype="PCM_16",
                        )
                        os.replace(temp_path, clip_path)
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

                duration = (end - start) / SAMPLE_RATE

                output_rows.append({
                    **common,
                    "clip_id": clip_id,
                    "clip_path": str(clip_path.resolve()),
                    "start_seconds": f"{start / SAMPLE_RATE:.3f}",
                    "end_seconds": f"{end / SAMPLE_RATE:.3f}",
                    "duration_seconds": f"{duration:.3f}",
                    "output_sha256": sha256_file(clip_path),
                    "status": "ok",
                })

            segmented_count += 1
            print(
                f"SEGMENTED: {source.name} "
                f"-> {len(clip_ranges)} clips"
            )

        except Exception as exc:
            error_count += 1

            current_errors.append({
                "source_path": str(source.resolve()),
                "error": f"{type(exc).__name__}: {exc}",
                "processed_at": now_utc(),
            })

            print(f"ERROR: {source.name}: {type(exc).__name__}: {exc}")

    write_csv(MANIFEST_PATH, MANIFEST_FIELDS, output_rows)
    write_csv(ERRORS_PATH, ERROR_FIELDS, current_errors)

    print("\nSegmentation summary")
    print(f"Standardized WAV files found: {len(sources)}")
    print(f"Sources segmented this run: {segmented_count}")
    print(f"Unchanged sources skipped: {skipped_count}")
    print(f"Sources with no suitable clips: {no_speech_count}")
    print(f"Source errors: {error_count}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Errors report: {ERRORS_PATH}")
    print("Input audio files were not modified or deleted.")


if __name__ == "__main__":
    main()
