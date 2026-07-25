"""
正向精度 UT 模板。

用法：根据实际 API 修改 target_api_*，将 ATK --save_data 产物置于当前目录后执行：
    python ut_forward.py 2>&1 | tee <工单目录>/运行日志/step3_ut_verify.log
"""
import sys
import torch
import torch_npu

torch.npu.set_device(0)

# ============ 根据实际API修改以下部分 ============

# 加载 ATK 保存的输入
# 常见格式：input.bin 包含 list of tensors，tensor_input.bin 包含单个 tensor
inp = torch.load('input.bin', weights_only=False)

# 如果是 Tensor.xxx 类型的API，还需加载 tensor_input.bin
# tensor_inp = torch.load('tensor_input.bin', weights_only=False)


def target_api_npu(inputs):
    """替换为实际的NPU API调用"""
    # 示例：torch.maximum(inputs[0].npu(), inputs[1].npu())
    raise NotImplementedError("请替换为实际API调用")


def target_api_cpu(inputs):
    """替换为实际的CPU API调用"""
    # 示例：torch.maximum(inputs[0], inputs[1])
    raise NotImplementedError("请替换为实际API调用")

# ============ 以下为通用对比逻辑，无需修改 ============

print("=" * 60)
print("正向精度UT验证")
print("=" * 60)

try:
    res_npu = target_api_npu(inp)
    res_cpu = target_api_cpu(inp)

    if isinstance(res_npu, torch.Tensor):
        res_npu_cpu = res_npu.cpu()
        diff = (res_npu_cpu - res_cpu).abs()
        print(f"NPU shape: {res_npu.shape}, dtype: {res_npu.dtype}")
        print(f"CPU shape: {res_cpu.shape}, dtype: {res_cpu.dtype}")
        print(f"Max abs diff: {diff.max().item()}")
        print(f"Mean abs diff: {diff.mean().item()}")
        print(f"torch.allclose(atol=1e-3): {torch.allclose(res_npu_cpu, res_cpu, atol=1e-3)}")
        print(f"torch.allclose(atol=1e-4): {torch.allclose(res_npu_cpu, res_cpu, atol=1e-4)}")
        print(f"torch.allclose(atol=1e-5): {torch.allclose(res_npu_cpu, res_cpu, atol=1e-5)}")

        if diff.max().item() > 1e-3:
            print("\n[FAIL] 精度差异超过1e-3阈值")
            sys.exit(1)
        else:
            print("\n[PASS] 精度在可接受范围内")
    else:
        print(f"NPU result: {res_npu}")
        print(f"CPU result: {res_cpu}")
        print(f"Equal: {res_npu == res_cpu}")

except Exception as e:
    print(f"\n[ERROR] 执行报错：")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
