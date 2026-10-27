
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent

SEGMENT_MANIFEST = ROOT / "data" / "metadata" / "segmentation_manifest.csv"
QUALITY_REPORT = ROOT / "reports" / "quality_check_report.csv"

INVENTORY_REPORT = ROOT / "data" / "metadata" / "dataset_inventory.csv"
DUPLICATE_REPORT = ROOT / "reports" / "duplicate_audio_report.csv"
SUMMARY_REPORT = ROOT / "reports" / "dataset_summary.txt"
SPLIT_PLAN = ROOT / "data" / "metadata" / "split_plan.csv"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    clips = read_csv(SEGMENT_MANIFEST)
    quality_rows = read_csv(QUALITY_REPORT)

    # Match quality results to clips using their resolved file paths.
    quality_by_path = {}

    for row in quality_rows:
        try:
            key = str(resolve_path(row.get("clip_path", "")))
            quality_by_path[key] = row
        except (ValueError, OSError):
            continue

    inventory = []
    hash_paths = defaultdict(list)
    label_counts = Counter()
    language_counts = Counter()
    source_by_label = defaultdict(set)
    source_records = {}

    missing_files = 0
    invalid_files = 0
    hash_mismatches = 0

    for row in clips:
        clip_value = row.get("clip_path", "")
        label = row.get("label", "").strip().lower()
        language = row.get("language", "").strip().lower() or "unknown"

        source_path = row.get("source_path", "").strip()
        source_hash = row.get("source_sha256", "").strip().lower()

        # Prefer the source hash as the identity of an original recording.
        source_id = source_hash or source_path

        source_records[source_id] = {
            "source_path": source_path,
            "source_sha256": source_hash,
            "label": label,
        }

        if label:
            label_counts[label] += 1
            source_by_label[label].add(source_id)

        language_counts[language] += 1

        try:
            clip_path = resolve_path(clip_value)
        except (ValueError, OSError):
            clip_path = ROOT / "__invalid_clip_path__"

        quality = quality_by_path.get(str(clip_path), {})
        quality_status = quality.get("status", "missing_quality_result")
        quality_issues = quality.get("issues", "")

        actual_hash = ""
        duration = ""
        sample_rate = ""
        channels = ""
        file_status = "ok"

        if not clip_path.is_file():
            file_status = "missing_file"
            missing_files += 1
        else:
            try:
                info = sf.info(str(clip_path))
                duration = round(info.duration, 3)
                sample_rate = info.samplerate
                channels = info.channels

                actual_hash = sha256_file(clip_path)
                hash_paths[actual_hash].append(str(clip_path))

                expected_hash = row.get("output_sha256", "").strip().lower()

                if expected_hash and actual_hash != expected_hash:
                    file_status = "hash_mismatch"
                    hash_mismatches += 1

                elif (
                    info.format != "WAV"
                    or info.samplerate != 16000
                    or info.channels != 1
                    or info.duration < 3.0
                    or info.duration > 8.0
                ):
                    file_status = "format_or_duration_review"
                    invalid_files += 1

            except Exception as exc:
                file_status = f"read_error: {type(exc).__name__}"
                invalid_files += 1

        inventory.append({
            "clip_path": clip_value,
            "label": label,
            "language": language,
            "source_path": source_path,
            "source_sha256": source_hash,
            "source_id": source_id,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "channels": channels,
            "output_sha256_actual": actual_hash,
            "file_status": file_status,
            "quality_status": quality_status,
            "quality_issues": quality_issues,
        })

    # Report identical file contents. Keep all files untouched.
    duplicate_rows = []

    for digest, paths in hash_paths.items():
        if len(paths) > 1:
            for path in paths:
                duplicate_rows.append({
                    "sha256": digest,
                    "number_of_identical_files": len(paths),
                    "clip_path": path,
                })

    write_csv(
        INVENTORY_REPORT,
        inventory,
        [
            "clip_path", "label", "language", "source_path",
            "source_sha256", "source_id", "duration_seconds",
            "sample_rate", "channels", "output_sha256_actual",
            "file_status", "quality_status", "quality_issues",
        ],
    )

    write_csv(
        DUPLICATE_REPORT,
        duplicate_rows,
        ["sha256", "number_of_identical_files", "clip_path"],
    )

    # This project currently has only two source recordings.
    # Do not invent a train/test split that cannot evaluate generalization.
    split_rows = []
    total_sources = len(source_records)
    enough_sources = (
        total_sources >= 10
        and all(len(source_by_label[label]) >= 3
                for label in ("genuine", "synthetic"))
    )

    for source_id, source in sorted(source_records.items()):
        split_rows.append({
            "source_id": source_id,
            "source_path": source["source_path"],
            "label": source["label"],
            "split": "UNASSIGNED" if not enough_sources else "REQUIRES_SPLIT_DESIGN",
            "reason": (
                "Source-level split design required; verify speaker and "
                "generator independence before assigning splits."
                if enough_sources else
                "Insufficient independent source recordings; collect more "
                "sources before creating train/validation/test splits."
            ),
        })

    write_csv(
        SPLIT_PLAN,
        split_rows,
        ["source_id", "source_path", "label", "split", "reason"],
    )

    quality_flagged = sum(
        row["quality_status"] != "pass" for row in inventory
    )

    with SUMMARY_REPORT.open("w", encoding="utf-8") as file:
        file.write("DATASET PREPARATION SUMMARY\n")
        file.write("=" * 32 + "\n\n")
        file.write(f"Manifest rows: {len(clips)}\n")
        file.write(f"Inventory entries: {len(inventory)}\n")
        file.write(f"Independent source IDs: {total_sources}\n")
        file.write(f"Missing files: {missing_files}\n")
        file.write(f"Hash mismatches: {hash_mismatches}\n")
        file.write(f"Invalid/unreadable files: {invalid_files}\n")
        file.write(f"Quality results not marked pass: {quality_flagged}\n")
        file.write(f"Duplicate-file entries: {len(duplicate_rows)}\n\n")

        file.write("CLIPS BY LABEL\n")
        for label, count in sorted(label_counts.items()):
            file.write(f"{label}: {count}\n")

        file.write("\nCLIPS BY LANGUAGE\n")
        for language, count in sorted(language_counts.items()):
            file.write(f"{language}: {count}\n")

        file.write("\nSOURCE RECORDINGS BY LABEL\n")
        for label in sorted(source_by_label):
            file.write(
                f"{label}: {len(source_by_label[label])} independent source(s)\n"
            )

        file.write("\nSPLIT STATUS\n")
        file.write(
            "Ready for split design.\n" if enough_sources
            else "Not ready for train/validation/test splitting.\n"
        )

    print("\nDataset Preparation Summary")
    print(f"Manifest rows: {len(clips)}")
    print(f"Independent source IDs: {total_sources}")
    print(f"Missing files: {missing_files}")
    print(f"Hash mismatches: {hash_mismatches}")
    print(f"Invalid/unreadable files: {invalid_files}")
    print(f"Quality results not marked pass: {quality_flagged}")
    print(f"Duplicate-file entries: {len(duplicate_rows)}")
    print("\nClips by label:")

    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nSource recordings by label:")

    for label in sorted(source_by_label):
        print(f"  {label}: {len(source_by_label[label])}")

    print("\nSplit design:", "ready for design" if enough_sources
          else "NOT READY — more independent recordings needed")

    print(f"\nInventory: {INVENTORY_REPORT}")
    print(f"Duplicate report: {DUPLICATE_REPORT}")
    print(f"Summary: {SUMMARY_REPORT}")
    print(f"Split plan: {SPLIT_PLAN}")
    print("No audio files were modified or deleted.")


if __name__ == "__main__":
    main()
