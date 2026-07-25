"""
反向梯度 UT 模板。

用法：根据实际 API 修改 target_api 与 requires_grad 开关，将 ATK --save_data 产物置于当前目录后执行：
    python ut_backward.py 2>&1 | tee <工单目录>/运行日志/step3_ut_verify.log
"""
import sys
import torch
import torch_npu

torch.npu.set_device(0)

# ============ 根据实际API修改以下部分 ============

# 加载输入数据
# tensor_input.bin: Tensor.xxx 类型API的self参数
# input.bin: 其他参数
# cpu_output_grad_0.pt / npu_output_grad_0.pt: 反向梯度

TENSOR_INPUT_FILE = 'tensor_input.bin'
INPUT_FILE = 'input.bin'
CPU_GRAD_FILE = 'cpu_output_grad_0.pt'
NPU_GRAD_FILE = 'npu_output_grad_0.pt'


def target_api(tensor_inp, other_inp):
    """替换为实际API调用，返回需要backward的输出tensor"""
    # 示例：return tensor_inp.remainder(other_inp)
    raise NotImplementedError("请替换为实际API调用")


# 哪些输入需要梯度
TENSOR_INP_REQUIRES_GRAD = True
OTHER_INP_REQUIRES_GRAD = True

# ============ 以下为通用对比逻辑，无需修改 ============

print("=" * 60)
print("反向梯度UT验证")
print("=" * 60)

try:
    # === CPU 路径 ===
    print("\n--- CPU 计算 ---")
    inp_cpu = torch.load(TENSOR_INPUT_FILE, weights_only=False)
    if isinstance(inp_cpu, (list, tuple)):
        inp_cpu = inp_cpu[0]
    inp_cpu = inp_cpu.clone().requires_grad_(TENSOR_INP_REQUIRES_GRAD)

    other_cpu = torch.load(INPUT_FILE, weights_only=False)
    if isinstance(other_cpu, (list, tuple)):
        other_cpu = other_cpu[0]
    other_cpu = other_cpu.clone().requires_grad_(OTHER_INP_REQUIRES_GRAD)

    out_cpu = target_api(inp_cpu, other_cpu)
    grad_output_cpu = torch.load(CPU_GRAD_FILE, weights_only=False)
    out_cpu.backward(grad_output_cpu)

    grad_inp_cpu = inp_cpu.grad
    grad_other_cpu = other_cpu.grad
    print(f"inp grad shape: {grad_inp_cpu.shape if grad_inp_cpu is not None else None}")
    print(f"other grad shape: {grad_other_cpu.shape if grad_other_cpu is not None else None}")

    # === NPU 路径 ===
    print("\n--- NPU 计算 ---")
    inp_npu = torch.load(TENSOR_INPUT_FILE, weights_only=False)
    if isinstance(inp_npu, (list, tuple)):
        inp_npu = inp_npu[0]
    inp_npu = inp_npu.to('npu').clone().requires_grad_(TENSOR_INP_REQUIRES_GRAD)

    other_npu = torch.load(INPUT_FILE, weights_only=False)
    if isinstance(other_npu, (list, tuple)):
        other_npu = other_npu[0]
    other_npu = other_npu.to('npu').clone().requires_grad_(OTHER_INP_REQUIRES_GRAD)

    out_npu = target_api(inp_npu, other_npu)
    grad_output_npu = torch.load(NPU_GRAD_FILE, weights_only=False).to('npu')
    out_npu.backward(grad_output_npu)

    grad_inp_npu = inp_npu.grad.cpu()
    grad_other_npu = other_npu.grad.cpu() if other_npu.grad is not None else None

    # === 对比 ===
    print("\n--- 梯度对比 ---")
    if grad_inp_cpu is not None and grad_inp_npu is not None:
        diff_inp = (grad_inp_cpu - grad_inp_npu).abs()
        print(f"Input grad max diff: {diff_inp.max().item()}")
        print(f"Input grad mean diff: {diff_inp.mean().item()}")
        print(f"Input grad allclose(atol=1e-3): {torch.allclose(grad_inp_cpu, grad_inp_npu, atol=1e-3)}")
        print(f"Input grad allclose(atol=1e-5): {torch.allclose(grad_inp_cpu, grad_inp_npu, atol=1e-5)}")

    if grad_other_cpu is not None and grad_other_npu is not None:
        diff_other = (grad_other_cpu - grad_other_npu).abs()
        print(f"Other grad max diff: {diff_other.max().item()}")
        print(f"Other grad mean diff: {diff_other.mean().item()}")
        print(f"Other grad allclose(atol=1e-3): {torch.allclose(grad_other_cpu, grad_other_npu, atol=1e-3)}")
        print(f"Other grad allclose(atol=1e-5): {torch.allclose(grad_other_cpu, grad_other_npu, atol=1e-5)}")

    # 判定
    has_issue = False
    if grad_inp_cpu is not None and grad_inp_npu is not None:
        if not torch.allclose(grad_inp_cpu, grad_inp_npu, atol=1e-3):
            has_issue = True
    if grad_other_cpu is not None and grad_other_npu is not None:
        if not torch.allclose(grad_other_cpu, grad_other_npu, atol=1e-3):
            has_issue = True

    if has_issue:
        print("\n[FAIL] 梯度精度差异超过1e-3阈值")
        sys.exit(1)
    else:
        print("\n[PASS] 梯度精度在可接受范围内")

except Exception as e:
    print(f"\n[ERROR] 执行报错：")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
