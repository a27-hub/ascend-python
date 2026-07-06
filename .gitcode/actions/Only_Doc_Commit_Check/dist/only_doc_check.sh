#!/bin/bash
########
# Only_Doc_Commit_Check
# 检查 PR 是否仅包含文档变更
#
# 调用 cie/plugin/only_doc.py 执行以下逻辑：
#   1. clone 目标仓库 -> 模拟 pre-merge -> 生成 modify_files.txt
#   2. 检查变更文件是否全为文档类型 (.md/.rst/.log/.png/.jpg/.doc)
#   3. 输出 JSON 结果 {"is_only_doc": "yes"/"no"}
########

set -eo pipefail

# 接收外部传入参数
PR_ID="$1"
TARGET_BRANCH="$2"
RESULT_MSG_PATH="$3"
STEP_ID="$4"
REMOTE_URL="$5"

echo "=========================================="
echo "  Only Doc Commit Check"
echo "=========================================="
echo "PR_ID:           ${PR_ID}"
echo "TARGET_BRANCH:   ${TARGET_BRANCH}"
echo "RESULT_MSG_PATH: ${RESULT_MSG_PATH}"
echo "STEP_ID:         ${STEP_ID}"
echo "REMOTE_URL:      ${REMOTE_URL}"

# --- 环境预检查 ---

# 检查 git
if ! command -v git &> /dev/null; then
    echo "ERROR: git 未安装或不在 PATH 中"
    exit 1
fi
echo "git version: $(git --version)"

# 检查 python (兼容 python3 / python)
PYTHON_BIN=""
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo "ERROR: python 未安装或不在 PATH 中"
    exit 1
fi
echo "python binary: ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"

# --- 构建结果文件路径 ---
if [ -n "${RESULT_MSG_PATH}" ] && [ -n "${STEP_ID}" ]; then
    RESULT_FILE="${RESULT_MSG_PATH}/${STEP_ID}_result.json"
else
    RESULT_FILE="./only_doc_result.json"
fi
echo "RESULT_FILE: ${RESULT_FILE}"

# 确保结果目录存在
RESULT_DIR=$(dirname "${RESULT_FILE}")
if [ ! -d "${RESULT_DIR}" ]; then
    mkdir -p "${RESULT_DIR}"
    echo "Created result directory: ${RESULT_DIR}"
fi

# --- 克隆 CI 工具仓库 ---
echo ""
echo ">>> Stage 1: Cloning ci_gitcode repository..."
CI_CLONE_DIR="./cie"
if [ -d "${CI_CLONE_DIR}" ]; then
    echo "Removing existing ${CI_CLONE_DIR} directory..."
    rm -rf "${CI_CLONE_DIR}"
fi
git clone https://codehub.devcloud.cn-north-4.huaweicloud.com/a9b188a62f334fc0963bcfb523426d17/ci_gitcode.git "${CI_CLONE_DIR}"
echo "CI tools cloned successfully."

# --- 执行 only_doc.py 检查 ---
echo ""
echo ">>> Stage 2: Running only_doc.py check..."
ONLY_DOC_SCRIPT="${CI_CLONE_DIR}/plugin/only_doc.py"

if [ ! -f "${ONLY_DOC_SCRIPT}" ]; then
    echo "ERROR: only_doc.py not found at ${ONLY_DOC_SCRIPT}"
    exit 1
fi

${PYTHON_BIN} "${ONLY_DOC_SCRIPT}" "${PR_ID}" "${TARGET_BRANCH}" "${RESULT_FILE}" "${REMOTE_URL}"
EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: only_doc.py 执行失败，exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

# --- 输出结果 ---
echo ""
echo ">>> Stage 3: Check result"
if [ -f "${RESULT_FILE}" ]; then
    echo "Result file: ${RESULT_FILE}"
    echo "----------------------------------------"
    cat "${RESULT_FILE}"
    echo ""
    echo "----------------------------------------"

    # 读取结果并设置输出
    IS_ONLY_DOC=$(python3 -c "import json; print(json.load(open('${RESULT_FILE}')).get('is_only_doc', 'unknown'))" 2>/dev/null || echo "unknown")
    echo ""
    echo "=========================================="
    echo "  Result: is_only_doc = ${IS_ONLY_DOC}"
    echo "=========================================="
else
    echo "ERROR: 结果文件 ${RESULT_FILE} 未生成"
    exit 1
fi

echo ""
echo "Only Doc Commit Check completed successfully."
