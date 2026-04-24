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

CXX11_ABI="$("${PYTHON_BIN}" - <<'PY'
import torch
print("1" if torch.compiled_with_cxx11_abi() else "0")
PY
)"

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
ATB_ABI_DIR="cxx_abi_${CXX11_ABI}"
ATB_SEARCH_ROOTS=(
    "${ATB_HOME_PATH:-}"
    "${NNAL_HOME_PATH:-}"
    "${ASCEND_NNAL_HOME:-}"
)
for root in "${ATB_SEARCH_ROOTS[@]}"; do
    if [[ -n "${root}" && -d "${root}/${ATB_ABI_DIR}" ]]; then
        LIBATB_PATH="$(find_first_file "${root}/${ATB_ABI_DIR}" "libatb.so")"
        if [[ -n "${LIBATB_PATH}" ]]; then
            break
        fi
    fi
done
if [[ -z "${LIBATB_PATH}" ]]; then
    echo "[ERROR] real libatb.so not found for ABI ${CXX11_ABI}"
    echo "[ERROR] set ATB_HOME_PATH, NNAL_HOME_PATH or ASCEND_NNAL_HOME to your NNAL ATB root"
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

cd "${BASE_DIR}"

echo "[INFO] building wheel..."
"${PYTHON_BIN}" setup.py build bdist_wheel

echo "[INFO] installing wheel..."
"${PYTHON_BIN}" -m pip install --force-reinstall dist/*.whl

echo "[INFO] running test..."
LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" "${PYTHON_BIN}" test/test_atb_quantize_per_tensor.py
echo "[INFO] run atb quantize per tensor sample success!"
