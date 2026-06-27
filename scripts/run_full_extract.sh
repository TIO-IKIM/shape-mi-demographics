#!/bin/bash
# Download the full TotalSegmentator zip (resumable) and extract all organ shapes.
# Portable: data goes to $SHAPEDEM_ROOT/data (default <repo>/_workspace/data).
# Run with the venv active (or PY=... ). Set JOBS to control parallelism.

# -e = exit immediately if any command fails
# -u = treat unset variables as an error (catches typos)
set -eu

# Resolve the repo root (one level up from scripts/)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# SHAPEDEM_ROOT overrides the workspace location; defaults to <repo>/_workspace
ROOT="${SHAPEDEM_ROOT:-$REPO/_workspace}"
# All raw downloads go here
DATA="$ROOT/data"; mkdir -p "$DATA"
# Path where the TotalSegmentator v2 zip will be saved
ZIP="$DATA/Totalsegmentator_dataset_v2.zip"
# Zenodo direct-download URL for TotalSegmentator v2 (~15 GB)
URL="https://zenodo.org/api/records/8367088/files/Totalsegmentator_dataset_v2.zip/content"
# Make sure Python can find the shapedem package
export PYTHONPATH="$REPO"
# Allow overriding the Python binary (e.g. PY=python3.12 bash run_full_extract.sh)
PY="${PY:-python}"

# Quick integrity check: a complete zip has >140k entries (1,228 subjects x ~117 organs)
valid_zip() { "$PY" -c "import zipfile,sys; sys.exit(0 if len(zipfile.ZipFile('$ZIP').namelist())>140000 else 1)" 2>/dev/null; }

# --- Step 1: Download the dataset zip ---
echo "[1/3] download (resumable) -> $ZIP"
if valid_zip; then
  echo "zip already present & valid"
else
  # -C - = resume a partial download where it left off
  # --retry 8 / --retry-delay 10 = retry up to 8 times with 10s between attempts
  # --retry-all-errors = retry on any error, not just transient HTTP codes
  curl -L -C - --retry 8 --retry-delay 10 --retry-all-errors -o "$ZIP" "$URL"
  valid_zip || { echo "ZIP INVALID"; exit 1; }
fi

# --- Step 2: Extract organ shapes from all subjects ---
# Runs marching cubes + surface sampling on each organ mask, saving point clouds
echo "[2/3] extract organ shapes (jobs=${JOBS:-8})"
"$PY" -m shapedem.cli extract --backend local --subjects all --jobs "${JOBS:-8}"

# --- Step 3: Run the main analyses (attribution, fusion, cross-domain, etc.) ---
echo "[3/3] analyses"
"$PY" -m shapedem.cli analyze
echo "RUN_FULL_DONE"
