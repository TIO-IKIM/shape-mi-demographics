#!/bin/bash
# One-command environment setup: creates a local .venv and installs the package.
# Usage:  bash setup.sh   (then: source .venv/bin/activate)
set -e
cd "$(dirname "$0")"
# Allow override via PYTHON env var (e.g. PYTHON=python3.12 bash setup.sh)
if [ -n "$PYTHON" ]; then
  PYBIN="$PYTHON"
else
  # Auto-detect: try common Python versions, pick the first where
  # venv + ensurepip actually work (avoids broken installs like py3.14 + macOS expat)
  PYBIN=""
  for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import venv, ensurepip" 2>/dev/null; then
      PYBIN="$candidate"
      break
    fi
  done
  if [ -z "$PYBIN" ]; then
    echo "Error: no working Python >=3.10 found. Install one or set PYTHON=<path>." >&2
    exit 1
  fi
fi
echo "Creating virtual environment in .venv (using $PYBIN) ..."
"$PYBIN" -m venv .venv
. .venv/bin/activate
python -m pip install -U pip wheel >/dev/null
echo "Installing shapedem and dependencies ..."
pip install -e .
echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python example/run_example.py        # offline demo (~1 min)"
echo
echo "For the deep models (optional, GPU): pip install torch"
