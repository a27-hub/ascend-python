#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

find_first_file() {
    local root="$1"
    local name="$2"
    find "${root}" -type f -name "${name}" 2>/dev/null | head -n 1
}

TORCH_NPU_ROOT="$("${PYTHON_BIN}" - <<'PY'
import os
import torch_npu
print(os.path.dirname(os.path.abspath(torch_npu.__file__)))
PY
)"
TORCH_NPU_LIB_DIR="${TORCH_NPU_ROOT}/lib"
TORCH_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
from torch.utils.cpp_extension import library_paths
paths = library_paths()
print(paths[0] if paths else "")
PY
)"

LIBOP_PLUGIN_ATB_PATH="${TORCH_NPU_LIB_DIR}/libop_plugin_atb.so"
if [[ ! -f "${LIBOP_PLUGIN_ATB_PATH}" ]]; then
    echo "[ERROR] libop_plugin_atb.so not found under ${TORCH_NPU_LIB_DIR}"
    exit 1
fi

LIBATB_PATH=""
if [[ -n "${ATB_HOME_PATH:-}" && -d "${ATB_HOME_PATH}" ]]; then
    LIBATB_PATH="$(find_first_file "${ATB_HOME_PATH}" "libatb.so")"
fi
if [[ -z "${LIBATB_PATH}" ]]; then
    echo "[ERROR] real libatb.so not found under ATB_HOME_PATH"
    echo "[ERROR] set ATB_HOME_PATH to a directory that contains libatb.so"
    exit 1
fi

RUNTIME_PATHS=(
    "$(dirname "${LIBOP_PLUGIN_ATB_PATH}")"
    "${TORCH_NPU_LIB_DIR}"
    "$(dirname "${LIBATB_PATH}")"
)
if [[ -n "${TORCH_LIB_DIR}" ]]; then
    RUNTIME_PATHS+=("${TORCH_LIB_DIR}")
fi

LD_LIBRARY_PATH_VALUE=""
for path in "${RUNTIME_PATHS[@]}"; do
    if [[ -d "${path}" ]]; then
        if [[ -z "${LD_LIBRARY_PATH_VALUE}" ]]; then
            LD_LIBRARY_PATH_VALUE="${path}"
        else
            LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${path}"
        fi
    fi
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${LD_LIBRARY_PATH}"
fi

cd "${BASE_DIR}"

echo "[INFO] building wheel..."
"${PYTHON_BIN}" setup.py build bdist_wheel

echo "[INFO] installing wheel..."
"${PYTHON_BIN}" -m pip install --force-reinstall dist/*.whl

echo "[INFO] running test..."
cd "${BASE_DIR}/test"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" "${PYTHON_BIN}" test_atb_quantize_per_tensor.py
echo "[INFO] run atb quantize per tensor sample success!"
