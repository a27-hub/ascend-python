#!/usr/bin/env bash
# 从日志文本中提取形如 aclnnXxx 的 token，并按出现次数聚类。
# 源自 solving-skill/scripts/extract_aclnn.sh；改为不使用 grep -P（便于 macOS / BSD）。
#
# 用法:
#   bash tools/op-delimitation-analyze-single-op/scripts/extract_aclnn.sh <log_file>
#
# 说明:
# - 输出仅为「文本命中 + 频次」辅助；单算子/组合类最终以 single-op SKILL 中正反向去重规则为准，
#   不能单凭本脚本最后一行启发式结论定案。
#
set -euo pipefail

if [ "${#}" -lt 1 ]; then
	echo "用法: $0 <log_file>"
	exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
	echo "[ERROR] 文件不存在: $LOG_FILE"
	exit 1
fi

echo "============================================================"
echo "ACLNN token 提取（文本辅助）"
echo "日志文件: $LOG_FILE"
echo "============================================================"

# ERE：aclnn 后跟字母数字下划线（覆盖常见 aclnnXxx / aclnn_xx 风格；可按实际日志微调）
_extract() {
	grep -Eo 'aclnn[A-Za-z0-9_]*' "$LOG_FILE" || true
}

echo ""
echo "--- 顺序列表（grep -Eo）---"
_extract | cat -n

echo ""
echo "--- 频次（降序）---"
_extract | sort | uniq -c | sort -rn

TOTAL=$(_extract | wc -l | tr -d '[:space:]')
UNIQUE=$(_extract | sort -u | wc -l | tr -d '[:space:]')

echo ""
echo "--- 统计 ---"
echo "总命中次数: $TOTAL"
echo "不同 token 数: $UNIQUE"

echo ""
echo "[备注] UNIQUE=0 → 可能在日志片段中无明文 aclnn token，需换含 TORCH_NPU_LOGS=op_plugin 的完整单跑日志再试。"
echo "UNIQUE=1 → 常为单 aclnn token；是否算「单算子」仍须按 single-op SKILL 去重。"
echo "UNIQUE>1 → 常为组合链路；是否与精度 dump 分叉结合见 precision SKILL。"
