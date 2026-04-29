#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

cd "${SCRIPT_DIR}"

echo "[INFO] building Python extension..."
"${PYTHON_BIN}" setup.py build_ext --inplace

echo "[INFO] running compare script..."
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${PYTHON_BIN}" run_compare.py
