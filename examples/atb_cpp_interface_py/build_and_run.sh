#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

LIBATB_PATH="$(find_first_file "${TORCH_NPU_ROOT}" "libatb.so")"
if [[ -z "${LIBATB_PATH}" ]]; then
    LIBATB_PATH="$(find_first_file "${REPO_ROOT}" "libatb.so")"
fi
if [[ -z "${LIBATB_PATH}" && -n "${ATB_HOME_PATH:-}" ]]; then
    LIBATB_PATH="$(find_first_file "${ATB_HOME_PATH}" "libatb.so")"
fi
if [[ -z "${LIBATB_PATH}" && -n "${NNAL_HOME_PATH:-}" ]]; then
    LIBATB_PATH="$(find_first_file "${NNAL_HOME_PATH}" "libatb.so")"
fi

RUNTIME_PATHS=("${TORCH_NPU_LIB_DIR}")
if [[ -n "${TORCH_LIB_DIR}" ]]; then
    RUNTIME_PATHS+=("${TORCH_LIB_DIR}")
fi
if [[ -n "${LIBATB_PATH}" ]]; then
    RUNTIME_PATHS+=("$(dirname "${LIBATB_PATH}")")
fi
if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}/lib64" ]]; then
    RUNTIME_PATHS+=("${ASCEND_HOME_PATH}/lib64")
fi
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}/lib64" ]]; then
    RUNTIME_PATHS+=("${ASCEND_TOOLKIT_HOME}/lib64")
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

cd "${SCRIPT_DIR}"

echo "[INFO] building Python extension..."
"${PYTHON_BIN}" setup.py build_ext --inplace

echo "[INFO] torch_npu lib: ${TORCH_NPU_LIB_DIR}"
if [[ -n "${TORCH_LIB_DIR}" ]]; then
    echo "[INFO] torch lib: ${TORCH_LIB_DIR}"
fi
if [[ -n "${LIBATB_PATH}" ]]; then
    echo "[INFO] libatb.so: ${LIBATB_PATH}"
else
    echo "[WARN] libatb.so not found by the helper; runtime will rely on current environment"
fi
echo "[INFO] running compare script..."
LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" \
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" run_compare.py
