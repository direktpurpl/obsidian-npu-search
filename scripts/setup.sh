#!/usr/bin/env bash
# Create an isolated Python environment for the plugin.
# Runtime data (venv, model, cache, index) lives outside the git repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${NPU_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/npu-search}"
VENV_DIR="${NPU_VENV:-$DATA_DIR/venv}"
PYTHON="${PYTHON:-python3}"

echo "[setup] repository: $REPO_DIR"
echo "[setup] data dir:   $DATA_DIR"
echo "[setup] venv:       $VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"

echo
echo "[setup] Python dependencies installed."
echo "[setup] Environment check:"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/npu_vault_search.py" --check --vault-path "${OBSIDIAN_VAULT_PATH:-}" || true
echo
echo "Next step: provide the multilingual-e5-base OpenVINO model, for example:"
echo "  $VENV_DIR/bin/python $SCRIPT_DIR/fetch_model.py --from /path/to/Multilingual-E5-Base"
echo "or let optimum export it:"
echo "  $VENV_DIR/bin/python -m pip install -r $REPO_DIR/requirements-model.txt"
echo "  $VENV_DIR/bin/python $SCRIPT_DIR/fetch_model.py --download"
