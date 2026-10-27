# Audio Data Preparation

A Python project for preparing audio recordings for a later audio deepfake-detection model. The repository is currently at an early stage: the implemented module inventories incoming audio and records source metadata. Later stages such as audio standardization, voice-activity segmentation, quality checks, and dataset split preparation are planned but are not included in this repository snapshot.

## What is implemented now

`src/ingest.py`:
- Scans audio files under `data/incoming/`.
- Infers `genuine` or `synthetic` labels from the first folder under `data/incoming/`.
- Calculates SHA-256 hashes.
- Uses `ffprobe` to read audio metadata such as duration, sample rate, channels, and codec.
- Writes a local ingestion manifest and an error report.
- Does not edit or delete the source audio.

Supported labels/folder names include `genuine`, `real`, `authentic`, `synthetic`, `fake`, and `ai_generated`. Keep recordings in a label folder, for example `data/incoming/genuine/malayalam/` or `data/incoming/synthetic/malayalam/`.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later (use a supported Python version compatible with your environment)
- FFmpeg installed, including `ffprobe`, available on your system `PATH`

The current ingestion module uses only Python's standard library, so `requirements.txt` has no third-party Python packages yet.

### Verify FFmpeg

Open a new terminal and run:

```powershell
ffmpeg -version
ffprobe -version
```

If either command is not recognized, install FFmpeg and add its `bin` directory to your `PATH`, then restart VS Code or the terminal.

## Setup on Windows (PowerShell)

From the folder where you want the repository:

```powershell
git clone https://github.com/aaditkrishna/audio-data-prep.git
cd audio-data-prep
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can run the project without activating it:

```powershell
.\.venv\Scripts\python.exe src/ingest.py
```

If the repository is not public yet, download/extract your repository ZIP instead of using `git clone`.

## Add new audio files

1. Copy original audio into the correct label folder:
   - Genuine recordings: `data/incoming/genuine/<language>/`
   - Synthetic recordings: `data/incoming/synthetic/<language>/`
2. Do not overwrite a different recording with the same filename.
3. If available, fill in `data/incoming/source_metadata.csv` using `data/incoming/source_metadata_template.csv` as a template. The `filepath` value must be relative to `data/incoming/`, such as `genuine/malayalam/recording_01.wav`.
4. Make sure you have permission to use the recordings and record the source/license information where possible.

## Run the current module

From the repository root:

```powershell
python src/ingest.py
```

Optional: if `ffprobe` is not on `PATH`, provide its executable path:

```powershell
python src/ingest.py --ffprobe "C:\path\to\ffprobe.exe"
```

The module writes these local outputs:
- `data/metadata/ingestion_manifest.csv`
- `reports/ingestion_errors.csv`

These files are intentionally excluded from Git tracking. Review the manifest's `status` and `error` columns after each run. A status of `ok` means the file was probed and its label was inferred from the folder; `needs_label_review` means the folder did not provide a recognized label; `error` means the file could not be read/probed successfully.

## Repeatable workflow checklist

Use this checklist whenever you add recordings:

- [ ] Confirm the recording is permitted for your project and note its source/license.
- [ ] Place each recording under the correct `genuine/` or `synthetic/` folder.
- [ ] Check that filenames do not accidentally overwrite other recordings.
- [ ] Verify `ffprobe -version` works in the terminal.
- [ ] Run `python src/ingest.py` from the repository root.
- [ ] Open `data/metadata/ingestion_manifest.csv` and review statuses, labels, hashes, and metadata.
- [ ] Fix any errors before continuing to the next processing stage.
- [ ] Keep original audio and generated outputs backed up locally; do not commit them to Git.
- [ ] Commit only source code and documentation changes.

## Repository hygiene

Audio files, generated manifests, reports, virtual environments, caches, and local backups are excluded by `.gitignore`. Ignore rules do not automatically untrack files that were already committed. Before pushing an existing repository, inspect `git status` and `git ls-files`; if audio/output files are already tracked, remove them from Git's index without deleting local copies:

```powershell
git rm --cached -- data/incoming/genuine/malayalam/"Real 1.mp3"
git rm --cached -- data/incoming/synthetic/malayalam/"Fake 1.mp3"
git rm --cached -- data/metadata/ingestion_manifest.csv
git rm --cached -- reports/ingestion_errors.csv
```

Adjust those paths to match the files that are actually tracked. Do not run these commands blindly; check `git ls-files` first. Then commit the cleanup.

## Planned modules

- [ ] Incremental source inventory and duplicate detection improvements
- [ ] Standardize audio to a consistent format/sample rate/channel count
- [ ] Segment recordings using voice activity detection
- [ ] Validate clip duration, audio format, silence, and clipping
- [ ] Build a source-level dataset inventory and split plan
- [ ] Add automated tests and a small safe test fixture
- [ ] Document reproducible environment versions and end-to-end workflow

## Important dataset note

Many clips cut from one recording are not independent recordings. When training and evaluating a model, split data by original source recording (and consider speaker/generator grouping) to reduce data leakage. A large clip count alone does not establish that the dataset is large or diverse enough.

## License

Choose and add a license before publishing the repository. A code license does not grant permission to redistribute audio files; verify each recording's separate license and usage terms.
