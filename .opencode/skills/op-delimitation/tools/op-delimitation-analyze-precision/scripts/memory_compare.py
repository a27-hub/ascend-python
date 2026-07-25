#!/usr/bin/env python3
"""
内存一致性对比脚本：读取 NPU 运行日志和 GPU Excel 数据，对比内存指标。

用法：
    python3 memory_compare.py --npu-log npu_output.log --gpu-xlsx 内存.xlsx --issue 5
"""

import json
import argparse
import sys
import re
from pathlib import Path


def parse_npu_log(log_path: str) -> list[dict]:
    """从 NPU 日志中提取 JSON 输出行。"""
    results = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and 'target_api' in line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def parse_gpu_xlsx(xlsx_path: str) -> list[dict]:
    """从 GPU Excel 中提取 GPU 内存数据。

    返回列表，每项为 dict，包含 32aligned(bool) 以及 GPU 的三项指标。
    Excel 格式（行首为表头）：
    id, target_api, id, 级别, 32aligned,
    total_driver_GB(NPU/GPU), pta_reserved_GB(NPU/GPU), pta_activated_GB(NPU/GPU),
    ..., GPU_total_driver_GB, GPU_pta_reserved_GB, GPU_pta_activated_GB, ...
    """
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        print("WARNING: Excel 只有表头，无数据", file=sys.stderr)
        return []

    # 表头行：确定 GPU 列位置
    header = rows[0]
    col_idx = {}
    for i, h in enumerate(header):
        if h is not None:
            col_idx[str(h).strip()] = i

    # GPU 数据在"gpu支持情况"标题之后的三个连续列
    # 找到"gpu支持情况"列，其后 1/2/3 列分别是 total_driver_GB, pta_reserved_GB, pta_activated_GB
    # 更稳健的方式：按列名搜索含 "GPU" 或按位置推算
    results = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        aligned = row[4]  # 32aligned 列（第5列，0-based=4）
        total_driver = None
        reserved = None
        activated = None

        for i, h in enumerate(header):
            h_str = str(h).strip() if h else ""
            if h_str == 'total_driver_GB' and 'gpu支持情况' in str(header[i-2] if i >= 2 else ''):
                # 这列在 GPU 区域
                pass
        # Fallback: 用硬编码列位置（根据常见 Excel 格式）
        # 第10列(index 9): GPU total_driver_GB, 第11列(index 10): GPU pta_reserved_GB, 第12列(index 11): GPU pta_activated_GB
        if len(row) > 11:
            total_driver = row[9]
            reserved = row[10]
            activated = row[11]

        if total_driver is not None:
            results.append({
                '32aligned': bool(aligned),
                'gpu_total_driver_GB': float(total_driver) if total_driver else 0.0,
                'gpu_pta_reserved_GB': float(reserved) if reserved else 0.0,
                'gpu_pta_activated_GB': float(activated) if activated else 0.0,
            })
    return results


def compare(npu: dict, gpu: dict, threshold: float = 0.05) -> dict:
    """对比单条 NPU vs GPU 记录，返回差异分析。

    diff% = |npu - gpu| / max(npu, gpu) * 100
    """
    metrics = ['total_driver_GB', 'pta_reserved_GB', 'pta_activated_GB']
    result = {'differences': {}, 'pass': True}

    for m in metrics:
        npu_val = npu.get(m, 0)
        gpu_val = gpu.get('gpu_' + m, 0)
        denominator = max(abs(npu_val), abs(gpu_val))
        if denominator == 0:
            diff_pct = 0.0
        else:
            diff_pct = abs(npu_val - gpu_val) / denominator * 100

        result['differences'][m] = {
            'npu': npu_val,
            'gpu': gpu_val,
            'diff_pct': diff_pct,
            'over_threshold': diff_pct > threshold * 100,
        }
        if diff_pct > threshold * 100:
            result['pass'] = False

    return result


def main():
    parser = argparse.ArgumentParser(description='NPU vs GPU 内存一致性对比')
    parser.add_argument('--npu-log', required=True, help='NPU 运行日志（含 JSON 输出）')
    parser.add_argument('--gpu-xlsx', required=True, help='GPU 基准数据 Excel 文件')
    parser.add_argument('--threshold', type=float, default=5.0, help='阈值百分比（默认5）')
    parser.add_argument('--issue', default='', help='问题单编号')
    args = parser.parse_args()

    npu_data = parse_npu_log(args.npu_log)
    gpu_data = parse_gpu_xlsx(args.gpu_xlsx)

    if not npu_data:
        print("ERROR: 未从 NPU 日志中提取到数据", file=sys.stderr)
        sys.exit(1)
    if not gpu_data:
        print("ERROR: 未从 Excel 中提取到 GPU 数据", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*70}")
    print(f"内存一致性对比分析 (阈值: {args.threshold}%)")
    print(f"{'='*70}")

    all_pass = True
    for npu_item in npu_data:
        aligned = npu_item.get('32aligned', False)
        mode = '32位对齐' if aligned else '普通(非32对齐)'
        gpu_item = next(
            (g for g in gpu_data if g.get('32aligned') == aligned), None
        )
        if gpu_item is None:
            print(f"\n[{mode}] 警告: 未找到对应 GPU 数据，跳过")
            continue

        result = compare(npu_item, gpu_item, args.threshold / 100.0)

        print(f"\n{'─'*70}")
        print(f"模式: {mode}")
        print(f"{'指标':<25} {'NPU':>12} {'GPU':>12} {'差异%':>10} {'判定'}")
        print(f"{'─'*70}")
        for metric, detail in result['differences'].items():
            short_name = metric.replace('gpu_', '')
            flag = '✗ FAIL' if detail['over_threshold'] else '✓ PASS'
            print(f"{short_name:<25} {detail['npu']:>10.4f} GB {detail['gpu']:>10.4f} GB {detail['diff_pct']:>8.2f}% {flag}")

        status = 'PASS' if result['pass'] else 'FAIL'
        if not result['pass']:
            all_pass = False
        print(f"{'─'*70}")
        print(f">> 综合判定: {status}")

    print(f"\n{'='*70}")
    print(f"最终结论: {'ALL PASS (内存一致性达标)' if all_pass else 'FAIL (内存一致性不达标)'}")
    print(f"{'='*70}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
