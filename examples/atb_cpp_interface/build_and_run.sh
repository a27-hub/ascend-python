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

mapfile -t ENV_INFO < <(
    "${PYTHON_BIN}" - <<'PY'
import os
import sysconfig
import torch
import torch_npu
from torch.utils.cpp_extension import include_paths, library_paths

print(f"CXX11_ABI={'1' if torch.compiled_with_cxx11_abi() else '0'}")
print(f"TORCH_NPU_PYTHON_ROOT={os.path.dirname(torch_npu.__file__)}")
print(f"PYTHON_INCLUDE={sysconfig.get_path('include') or ''}")
print(f"PYTHON_LIBDIR={sysconfig.get_config_var('LIBDIR') or ''}")
print(f"PYTHON_LIBPL={sysconfig.get_config_var('LIBPL') or ''}")
print(f"PYTHON_LIBRARY={sysconfig.get_config_var('LDLIBRARY') or ''}")
print(f"PYTHON_LINKFORSHARED={(sysconfig.get_config_var('LINKFORSHARED') or '').strip()}")
print(f"PYTHON_LIBS={(sysconfig.get_config_var('LIBS') or '').strip()}")
print(f"PYTHON_SYSLIBS={(sysconfig.get_config_var('SYSLIBS') or '').strip()}")
for item in include_paths():
    print(f"INCLUDE={item}")
for item in library_paths():
    print(f"LIB={item}")
PY
)

CXX11_ABI=""
TORCH_NPU_PYTHON_ROOT=""
PYTHON_INCLUDE_DIR=""
PYTHON_LIBDIR=""
PYTHON_LIBPL=""
PYTHON_LIBRARY=""
PYTHON_LINKFORSHARED=""
PYTHON_LIBS=""
PYTHON_SYSLIBS=""
TORCH_INCLUDE_DIRS=()
TORCH_LIBRARY_DIRS=()

for line in "${ENV_INFO[@]}"; do
    case "${line}" in
        CXX11_ABI=*) CXX11_ABI="${line#CXX11_ABI=}" ;;
        TORCH_NPU_PYTHON_ROOT=*) TORCH_NPU_PYTHON_ROOT="${line#TORCH_NPU_PYTHON_ROOT=}" ;;
        PYTHON_INCLUDE=*) PYTHON_INCLUDE_DIR="${line#PYTHON_INCLUDE=}" ;;
        PYTHON_LIBDIR=*) PYTHON_LIBDIR="${line#PYTHON_LIBDIR=}" ;;
        PYTHON_LIBPL=*) PYTHON_LIBPL="${line#PYTHON_LIBPL=}" ;;
        PYTHON_LIBRARY=*) PYTHON_LIBRARY="${line#PYTHON_LIBRARY=}" ;;
        PYTHON_LINKFORSHARED=*) PYTHON_LINKFORSHARED="${line#PYTHON_LINKFORSHARED=}" ;;
        PYTHON_LIBS=*) PYTHON_LIBS="${line#PYTHON_LIBS=}" ;;
        PYTHON_SYSLIBS=*) PYTHON_SYSLIBS="${line#PYTHON_SYSLIBS=}" ;;
        INCLUDE=*) TORCH_INCLUDE_DIRS+=("${line#INCLUDE=}") ;;
        LIB=*) TORCH_LIBRARY_DIRS+=("${line#LIB=}") ;;
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

find_header_root() {
    local search_root="$1"
    local header_name="$2"
    local suffix="$3"
    local header_path
    header_path="$(find "${search_root}" -type f -name "${header_name}" 2>/dev/null | head -n 1)"
    if [[ -n "${header_path}" && "${header_path}" == *"${suffix}" ]]; then
        echo "${header_path%${suffix}}"
    fi
}

GENERATED_HEADER="$(find "${REPO_ROOT}" -type f -path '*/op_plugin/include/AtbOpsInterface.h' 2>/dev/null | head -n 1)"
if [[ -z "${GENERATED_HEADER}" ]]; then
    echo "[ERROR] generated AtbOpsInterface.h not found under ${REPO_ROOT}"
    echo "[ERROR] please run the ATB codegen/build flow first"
    exit 1
fi
GENERATED_INCLUDE_ROOT="${GENERATED_HEADER%/op_plugin/include/AtbOpsInterface.h}"

TORCH_NPU_INCLUDE_ROOT="$(find_header_root "${REPO_ROOT}" "NPUMacros.h" "/torch_npu/csrc/core/npu/NPUMacros.h")"
if [[ -z "${TORCH_NPU_INCLUDE_ROOT}" && -n "${TORCH_NPU_PYTHON_ROOT}" ]]; then
    TORCH_NPU_INCLUDE_ROOT="$(find_header_root "${TORCH_NPU_PYTHON_ROOT}" "NPUMacros.h" "/torch_npu/csrc/core/npu/NPUMacros.h")"
fi
if [[ -z "${TORCH_NPU_INCLUDE_ROOT}" ]]; then
    echo "[ERROR] torch_npu/csrc/core/npu/NPUMacros.h not found"
    echo "[ERROR] cannot infer the torch_npu header include root"
    exit 1
fi

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

PYTHON_SHARED_LIBRARY_PATH=""
if [[ -n "${PYTHON_LIBRARY}" ]]; then
    if [[ -n "${PYTHON_LIBDIR}" && -f "${PYTHON_LIBDIR}/${PYTHON_LIBRARY}" ]]; then
        PYTHON_SHARED_LIBRARY_PATH="${PYTHON_LIBDIR}/${PYTHON_LIBRARY}"
    elif [[ -n "${PYTHON_LIBPL}" && -f "${PYTHON_LIBPL}/${PYTHON_LIBRARY}" ]]; then
        PYTHON_SHARED_LIBRARY_PATH="${PYTHON_LIBPL}/${PYTHON_LIBRARY}"
    fi
fi

INCLUDE_DIRS=(
    "${REPO_ROOT}"
    "${GENERATED_INCLUDE_ROOT}"
    "${TORCH_NPU_INCLUDE_ROOT}"
    "${PYTHON_INCLUDE_DIR}"
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
if [[ -n "${PYTHON_LIBDIR}" ]]; then
    LIB_DIRS+=("${PYTHON_LIBDIR}")
fi
if [[ -n "${PYTHON_LIBPL}" ]]; then
    LIB_DIRS+=("${PYTHON_LIBPL}")
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

split_flags() {
    local raw="$1"
    local -n out_ref=$2
    if [[ -z "${raw}" ]]; then
        return
    fi
    # shellcheck disable=SC2206
    local parts=( ${raw} )
    for item in "${parts[@]}"; do
        out_ref+=("${item}")
    done
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

if [[ -n "${PYTHON_SHARED_LIBRARY_PATH}" ]]; then
    COMPILE_CMD+=("${PYTHON_SHARED_LIBRARY_PATH}")
elif [[ -n "${PYTHON_LIBRARY}" && "${PYTHON_LIBRARY}" == lib* ]]; then
    PYTHON_LIB_NAME="${PYTHON_LIBRARY#lib}"
    if [[ "${PYTHON_LIB_NAME}" == *.so.* ]]; then
        PYTHON_LIB_NAME="${PYTHON_LIB_NAME%%.so*}"
    elif [[ "${PYTHON_LIB_NAME}" == *.so ]]; then
        PYTHON_LIB_NAME="${PYTHON_LIB_NAME%.so}"
    elif [[ "${PYTHON_LIB_NAME}" == *.a ]]; then
        PYTHON_LIB_NAME="${PYTHON_LIB_NAME%.a}"
    fi
    COMPILE_CMD+=("-l${PYTHON_LIB_NAME}")
fi

PYTHON_EXTRA_LDFLAGS=()
split_flags "${PYTHON_LINKFORSHARED}" PYTHON_EXTRA_LDFLAGS
split_flags "${PYTHON_LIBS}" PYTHON_EXTRA_LDFLAGS
split_flags "${PYTHON_SYSLIBS}" PYTHON_EXTRA_LDFLAGS
if [[ ${#PYTHON_EXTRA_LDFLAGS[@]} -gt 0 ]]; then
    COMPILE_CMD+=("${PYTHON_EXTRA_LDFLAGS[@]}")
fi

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
echo "[INFO] torch_npu include root: ${TORCH_NPU_INCLUDE_ROOT}"
echo "[INFO] libop_plugin_atb.so: ${LIBOP_PLUGIN_ATB}"
if [[ -n "${PYTHON_SHARED_LIBRARY_PATH}" ]]; then
    echo "[INFO] python shared library: ${PYTHON_SHARED_LIBRARY_PATH}"
else
    echo "[INFO] python shared library: using linker name fallback (${PYTHON_LIBRARY})"
fi
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
