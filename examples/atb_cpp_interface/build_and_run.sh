#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CXX_BIN="${CXX:-g++}"
EXAMPLE_SRC="${SCRIPT_DIR}/atb_quantize_per_tensor_example.cpp"
EXAMPLE_BIN="${SCRIPT_DIR}/atb_quantize_per_tensor_example"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] C++ compiler not found: ${CXX_BIN}"
    exit 1
fi

mapfile -t TORCH_INFO < <(
    "${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch
from torch.utils.cpp_extension import include_paths, library_paths

print(f"CXX11_ABI={'1' if torch.compiled_with_cxx11_abi() else '0'}")
for item in include_paths():
    print(f"INCLUDE={item}")
for item in library_paths():
    print(f"LIB={item}")
PY
)

CXX11_ABI=""
TORCH_INCLUDE_DIRS=()
TORCH_LIBRARY_DIRS=()
for line in "${TORCH_INFO[@]}"; do
    case "${line}" in
        CXX11_ABI=*)
            CXX11_ABI="${line#CXX11_ABI=}"
            ;;
        INCLUDE=*)
            TORCH_INCLUDE_DIRS+=("${line#INCLUDE=}")
            ;;
        LIB=*)
            TORCH_LIBRARY_DIRS+=("${line#LIB=}")
            ;;
    esac
done

if [[ -z "${CXX11_ABI}" ]]; then
    echo "[ERROR] failed to query torch C++ ABI setting"
    exit 1
fi

find_first_file() {
    local root="$1"
    local name="$2"
    find "${root}" -type f -name "${name}" 2>/dev/null | head -n 1
}

GENERATED_HEADER="$(find "${REPO_ROOT}" -type f -path '*/op_plugin/include/AtbOpsInterface.h' 2>/dev/null | head -n 1)"
if [[ -z "${GENERATED_HEADER}" ]]; then
    echo "[ERROR] generated AtbOpsInterface.h not found under ${REPO_ROOT}"
    echo "[ERROR] please run the ATB codegen/build flow first"
    exit 1
fi
GENERATED_INCLUDE_ROOT="${GENERATED_HEADER%/op_plugin/include/AtbOpsInterface.h}"

LIBOP_PLUGIN_ATB="$(find_first_file "${REPO_ROOT}" "libop_plugin_atb.so")"
if [[ -z "${LIBOP_PLUGIN_ATB}" ]]; then
    echo "[ERROR] libop_plugin_atb.so not found under ${REPO_ROOT}"
    echo "[ERROR] please build torch_npu/op-plugin artifacts first"
    exit 1
fi
TORCH_NPU_LIB_DIR="$(dirname "${LIBOP_PLUGIN_ATB}")"

LIBATB_PATH="$(find_first_file "${REPO_ROOT}" "libatb.so")"
LIBATB_DIR=""
if [[ -n "${LIBATB_PATH}" ]]; then
    LIBATB_DIR="$(dirname "${LIBATB_PATH}")"
fi

INCLUDE_DIRS=(
    "${REPO_ROOT}"
    "${GENERATED_INCLUDE_ROOT}"
)
for dir in "${TORCH_INCLUDE_DIRS[@]}"; do
    INCLUDE_DIRS+=("${dir}")
done

LIB_DIRS=(
    "${TORCH_NPU_LIB_DIR}"
)
for dir in "${TORCH_LIBRARY_DIRS[@]}"; do
    LIB_DIRS+=("${dir}")
done
if [[ -n "${LIBATB_DIR}" ]]; then
    LIB_DIRS+=("${LIBATB_DIR}")
fi

if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}/lib64" ]]; then
    LIB_DIRS+=("${ASCEND_HOME_PATH}/lib64")
fi
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}/lib64" ]]; then
    LIB_DIRS+=("${ASCEND_TOOLKIT_HOME}/lib64")
fi

dedupe_array() {
    local -n input_ref=$1
    local -A seen=()
    local output=()
    for item in "${input_ref[@]}"; do
        if [[ -n "${item}" && -d "${item}" && -z "${seen[$item]:-}" ]]; then
            seen["${item}"]=1
            output+=("${item}")
        fi
    done
    input_ref=("${output[@]}")
}

dedupe_array INCLUDE_DIRS
dedupe_array LIB_DIRS

COMPILE_CMD=(
    "${CXX_BIN}"
    -std=c++17
    "-D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI}"
    "${EXAMPLE_SRC}"
    -o "${EXAMPLE_BIN}"
)

for dir in "${INCLUDE_DIRS[@]}"; do
    COMPILE_CMD+=(-I "${dir}")
done
for dir in "${LIB_DIRS[@]}"; do
    COMPILE_CMD+=(-L "${dir}")
    COMPILE_CMD+=(-Wl,-rpath,"${dir}")
    COMPILE_CMD+=(-Wl,-rpath-link,"${dir}")
done

COMPILE_CMD+=(
    -Wl,--allow-shlib-undefined
    "${LIBOP_PLUGIN_ATB}"
    -ltorch_python
    -ltorch
    -ltorch_cpu
    -lc10
)

echo "[INFO] repository root: ${REPO_ROOT}"
echo "[INFO] generated header: ${GENERATED_HEADER}"
echo "[INFO] libop_plugin_atb.so: ${LIBOP_PLUGIN_ATB}"
if [[ -n "${LIBATB_PATH}" ]]; then
    echo "[INFO] libatb.so: ${LIBATB_PATH}"
else
    echo "[WARN] libatb.so not found under repository root; runtime will rely on current environment"
fi
echo "[INFO] compiling example..."
printf '[INFO] command:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'
"${COMPILE_CMD[@]}"

RUNTIME_LD_LIBRARY_PATH=""
for dir in "${LIB_DIRS[@]}"; do
    if [[ -z "${RUNTIME_LD_LIBRARY_PATH}" ]]; then
        RUNTIME_LD_LIBRARY_PATH="${dir}"
    else
        RUNTIME_LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}:${dir}"
    fi
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    RUNTIME_LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

echo "[INFO] running example..."
LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" "${EXAMPLE_BIN}"
