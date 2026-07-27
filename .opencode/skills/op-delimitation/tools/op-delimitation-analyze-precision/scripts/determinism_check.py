"""
确定性验证模板（50 次循环）。

用法：根据实际 API 修改 run_once()，在 --save_data 产物目录执行：
    python determinism_check.py 2>&1 | tee <工单目录>/运行日志/determinism_check.log

结单时在 分析结论.md「定位过程」记录：总次数 / 不一致次数 / 首次分叉轮次 / Max diff。
"""
import sys
import os
import torch
import torch_npu

torch.npu.set_device(0)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPEAT_COUNT = 50

# ============ 根据实际 API 修改以下部分 ============

INPUT_FILE = "input.bin"


def run_once():
    """
    替换为实际的 API 调用。
    每次调用前重新设 seed；每次重新 load 输入，避免 inplace 污染。
    返回需对比的结果 tensor（建议 .cpu()）。
    """
    torch.manual_seed(2025)
    torch.use_deterministic_algorithms(True)

    inp = torch.load(INPUT_FILE, weights_only=False)

    # 示例：
    # x = inp[0].npu()
    # result = torch.some_api(x)
    # return result.cpu()

    raise NotImplementedError("请替换为实际 API 调用")


# ============ 以下为通用验证逻辑，无需修改 ============

print("=" * 60)
print(f"确定性验证（重复 {REPEAT_COUNT} 次）")
print("=" * 60)

try:
    baseline = run_once()
    print(f"Baseline shape: {baseline.shape}, dtype: {baseline.dtype}")
    print(f"Baseline mean: {baseline.float().mean().item()}")

    inconsistent_count = 0
    first_diff_run = -1
    first_max_diff = None

    for i in range(REPEAT_COUNT - 1):
        current = run_once()
        if not torch.equal(baseline, current):
            inconsistent_count += 1
            if first_diff_run == -1:
                first_diff_run = i + 2
                diff = (baseline.float() - current.float()).abs()
                first_max_diff = diff.max().item()
                print(f"\n第{i + 2}次结果不一致!")
                print(f"  Max diff: {first_max_diff}")
                print(f"  Diff positions: {(diff > 0).sum().item()} / {diff.numel()}")

    print("\n--- 统计结果 ---")
    print(f"总执行次数: {REPEAT_COUNT}")
    print(f"不一致次数: {inconsistent_count}")

    if inconsistent_count == 0:
        print(f"\n[PASS] {REPEAT_COUNT} 次运行结果完全一致，确定性验证通过")
    else:
        print(f"\n[FAIL] 存在 {inconsistent_count} 次不一致，首次出现在第 {first_diff_run} 次")
        if first_max_diff is not None:
            print(f"  首次 Max diff: {first_max_diff}")
        sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] 执行报错：")
    print(f"{type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
