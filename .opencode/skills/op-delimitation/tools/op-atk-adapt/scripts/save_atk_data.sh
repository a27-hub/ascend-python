#!/usr/bin/env bash
# 单 case 本地 --save_data 便捷脚本（op-atk-adapt 简化版）
# 去掉 copy 步骤（cwd=pta 目录 → atk_output 就地生成在同目录下）
#
# 用法:
#   bash tools/op-atk-adapt/scripts/save_atk_data.sh \
#       <case_json> <node_yaml> <case_id> <issue_dir>
#
# 要求在已 cd 到 {opName}_pta_atk 目录（含 -c/-n 相对路径能解析）、且 atk 已在 PATH。
#
set -euo pipefail

if [ "${#}" -lt 4 ]; then
	echo "用法: $0 <case_json> <node_yaml> <case_id> <issue_dir>"
	echo ""
	echo "示例:"
	echo "  $0 result/all_function_swiglu_group_quant_backward.json node.yaml 12 workdir/issue-001"
	exit 1
fi

JSON_FILE="$1"
NODE_YAML="$2"
CASE_ID="$3"
ISSUE_DIR="$4"

for f in "$JSON_FILE" "$NODE_YAML"; do
	if [ ! -f "$f" ]; then
		echo "[ERROR] 文件不存在: $f"
		exit 1
	fi
done

mkdir -p "$ISSUE_DIR" "${ISSUE_DIR}/运行日志"

echo "============================================================"
echo "ATK --save_data（单 case · 本地）"
echo "JSON:       $JSON_FILE"
echo "Node YAML:  $NODE_YAML"
echo "case_id:    $CASE_ID"
echo "输出目录:    $ISSUE_DIR"
echo "============================================================"

# op-atk-adapt 权威基线
unset http_proxy https_proxy
export ATK_BIND_CPU_TYPE=2

echo ""
echo "--- 执行 atk task（--save_data input output + tee）---"
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && \
	atk task -c "$JSON_FILE" -n "$NODE_YAML" -wl "[${CASE_ID}]" \
		--save_data input --save_data output \
		2>&1 | tee "${ISSUE_DIR}/运行日志/save_atk_data_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "--- 保存完成 ---"
echo "atk_output/ 已就地生成在当前目录（{opName}_pta_atk/）下"
echo "input/output 素材路径：atk_output/all_*/input/ 和 atk_output/all_*/output/"
ls -la atk_output/ 2>/dev/null || echo "[WARN] atk_output/ 不存在——检查 atk task 是否成功"

echo ""
echo "[DONE] 数据已保存"
