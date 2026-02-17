# nfl-draft-nlp

Text mining project scaffold for NFL Draft analysis.

## Project Structure

- `scripts/data/`: data ingestion scripts (Google Drive pull, preprocessing helpers)
- `scripts/analysis/`: analysis/feature scripts
- `src/nfl_draft_nlp/`: reusable Python package code
- `data/raw/`: downloaded source data (git-ignored)
- `data/interim/`: temporary transformed data (git-ignored)
- `data/processed/`: model-ready data (git-ignored)
- `data/external/`: third-party static data (git-ignored)
- `notebooks/`: exploratory notebooks
- `config/`: non-secret config files
- `.secrets/`: local credentials/tokens (git-ignored)

## Setup With uv

1. Install uv (if needed):
   - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Create virtual env and install deps:
   - `uv sync`
3. Activate env:
   - PowerShell: `.venv\\Scripts\\Activate.ps1`

## Data Setup (Manual Download)

Use this shared Drive folder:
- `https://drive.google.com/drive/u/0/folders/1zqobJD1gXaTZeCOCTtaBFbX-Py-DGY3m`

Each teammate should:
1. Open the link and download the folder contents.
2. Extract/copy files into `data/raw/` in this repo.
3. Keep the same folder structure from Drive inside `data/raw/`.

Notes:
- No API/auth script is required.
- `data/` contents are git-ignored, so large files will stay local.

## Recommended Team Workflow

- Keep all source data in shared Drive.
- Everyone manually refreshes `data/raw/` from Drive when new files are added.
- Git only tracks code + small metadata; data folders stay ignored.
