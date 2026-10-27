# TODO — Audio Data Preparation

## Current baseline
- [x] Keep the initial ingestion script in `src/ingest.py`.
- [x] Document FFmpeg/ffprobe as an external requirement.
- [x] Add ignore rules for local audio, generated files, environments, and backups.
- [ ] Verify `ffprobe` is installed and visible to the terminal.
- [ ] Run ingestion and confirm files have `status=ok` in the manifest.

## Before every ingestion run
- [ ] Add new files under `data/incoming/genuine/<language>/` or `data/incoming/synthetic/<language>/`.
- [ ] Confirm the labels and usage permissions are correct.
- [ ] Add source/speaker/language/license details to `data/incoming/source_metadata.csv` when known.
- [ ] Run `python src/ingest.py`.
- [ ] Review `data/metadata/ingestion_manifest.csv` and `reports/ingestion_errors.csv`.
- [ ] Resolve errors and label-review cases before using the files downstream.

## Planned implementation
- [ ] Add a robust incremental ingest strategy and duplicate report.
- [ ] Implement standardization to a single audio format and sample rate.
- [ ] Implement speech segmentation/VAD with a reproducible configuration.
- [ ] Add audio quality validation.
- [ ] Create a dataset inventory and source-level train/validation/test split plan.
- [ ] Add tests for path handling, labels, hashes, duplicate inputs, and repeat runs.
- [ ] Pin/test dependency versions in a clean virtual environment.
- [ ] Document the complete pipeline only after each stage is implemented and tested.

## Before pushing to GitHub
- [ ] Check `git status`.
- [ ] Check `git ls-files` for audio, manifests, reports, logs, and backups.
- [ ] Ensure local data and generated outputs are not tracked.
- [ ] Add a suitable license for the code.
- [ ] Never publish recordings unless their licenses and consent/usage terms permit redistribution.
