
import csv
import hashlib
import math
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "metadata" / "segmentation_manifest.csv"
REPORT = ROOT / "reports" / "quality_check_report.csv"

MIN_DURATION = 3.0
MAX_DURATION = 8.0
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def check_clip(row):
    issues = []
    clip_path = ROOT / row["clip_path"]

    if not clip_path.is_file():
        return ["missing_file"]

    try:
        info = sf.info(str(clip_path))

        if info.format != "WAV":
            issues.append("not_wav")

        if info.samplerate != EXPECTED_SAMPLE_RATE:
            issues.append("wrong_sample_rate")

        if info.channels != EXPECTED_CHANNELS:
            issues.append("wrong_channel_count")

        duration = info.duration

        if duration < MIN_DURATION:
            issues.append("too_short")

        if duration > MAX_DURATION:
            issues.append("too_long")

        audio, sample_rate = sf.read(
            str(clip_path),
            dtype="float32",
            always_2d=True
        )

        if not np.isfinite(audio).all():
            issues.append("invalid_audio_values")
        else:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

            if peak < 1e-6 or rms < 1e-5:
                issues.append("silent_or_very_quiet")

            # Flag possible digital clipping for review.
            if peak >= 0.999:
                issues.append("possible_clipping")

        actual_hash = sha256_file(clip_path)
        expected_hash = row.get("output_sha256", "").strip()

        if expected_hash and actual_hash.lower() != expected_hash.lower():
            issues.append("hash_mismatch")
        elif not expected_hash:
            issues.append("missing_expected_hash")

    except Exception as exc:
        issues.append(f"read_error: {type(exc).__name__}: {exc}")

    return issues

def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Segmentation manifest not found: {MANIFEST}"
        )

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("The segmentation manifest contains no rows.")
        return

    results = []
    for index, row in enumerate(rows, start=1):
        issues = check_clip(row)
        results.append({
            "clip_path": row.get("clip_path", ""),
            "label": row.get("label", ""),
            "source_path": row.get("source_path", ""),
            "duration_seconds": row.get("duration_seconds", ""),
            "status": "pass" if not issues else "review",
            "issues": "; ".join(issues)
        })

        if index % 100 == 0:
            print(f"Checked {index}/{len(rows)} clips")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "clip_path",
                "label",
                "source_path",
                "duration_seconds",
                "status",
                "issues"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    passed = sum(item["status"] == "pass" for item in results)
    flagged = len(results) - passed

    print("\nAudio Quality Check Summary")
    print(f"Clips checked: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Flagged for review: {flagged}")
    print(f"Report: {REPORT}")
    print("No audio files were modified or deleted.")

if __name__ == "__main__":
    main()
