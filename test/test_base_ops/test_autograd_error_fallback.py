# -*- coding: utf-8 -*-
# 验证 4 个无反向算子的 AutogradPrivateUse1 拦截是否生效。
# 预期：前向正常执行；对输出调用 .backward() 时抛出
#       RuntimeError: derivative for 'npu::xxx' is not implemented
#
# 注意：下面输入的 shape / dtype 是常见示例值。如果某个算子前向报
#       shape/dtype 不匹配的错误，请用你实际业务里的输入替换即可，
#       不影响对“反向是否抛异常”的验证。

import torch
import torch_npu  # noqa: F401

DEVICE = "npu"


def check(name, forward_fn, grad_index=0):
    """前向应成功；对第 grad_index 个输出反向应抛 'is not implemented'。"""
    print(f"\n===== {name} =====")
    # ---- 前向 ----
    try:
        outs = forward_fn()
    except Exception as e:  # noqa: BLE001
        print(f"[前向] 失败（请检查输入 shape/dtype 是否符合算子要求）:\n  {e}")
        return
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    print(f"[前向] 成功，输出数量 = {len(outs)}")

    # ---- 反向 ----
    target = outs[grad_index]
    if not torch.is_floating_point(target):
        print(f"[反向] 第 {grad_index} 个输出非浮点，无法反传，请换 grad_index")
        return
    try:
        target.float().sum().backward()
        print("[反向] 未抛异常 —— 不符合预期！应当抛出 'is not implemented'")
    except RuntimeError as e:
        first_line = str(e).strip().splitlines()[0]
        if "is not implemented" in str(e):
            print(f"[反向] 符合预期，抛出 RuntimeError:\n  {first_line}")
        else:
            print(f"[反向] 抛出 RuntimeError，但文案不符预期:\n  {str(e)}")
    except Exception as e:  # noqa: BLE001
        print(f"[反向] 抛出非 RuntimeError 异常: {type(e).__name__}: {e}")


# 1. npu_add_rms_norm(Tensor x1, Tensor x2, Tensor gamma, float epsilon) -> (T,T,T)
def t_add_rms_norm():
    x1 = torch.randn(4, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    x2 = torch.randn(4, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    gamma = torch.randn(16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    return torch.ops.npu.npu_add_rms_norm(x1, x2, gamma, 1e-6)


# 2. npu_apply_rotary_pos_emb(query, key, cos, sin, layout, rotary_mode) -> (T,T)
def t_apply_rotary_pos_emb():
    q = torch.randn(1, 4, 2, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    k = torch.randn(1, 4, 2, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    cos = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=DEVICE)
    sin = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=DEVICE)
    return torch.ops.npu.npu_apply_rotary_pos_emb(q, k, cos, sin, "BSND", "half")


# 3. npu_interleave_rope(Tensor x, Tensor cos, Tensor sin) -> Tensor
def t_interleave_rope():
    x = torch.randn(1, 4, 2, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    cos = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=DEVICE)
    sin = torch.randn(1, 4, 1, 16, dtype=torch.float16, device=DEVICE)
    return torch.ops.npu.npu_interleave_rope(x, cos, sin)


# 4. npu_moe_gating_top_k_softmax(Tensor x, Tensor? finished, int k) -> (T,T,T)
def t_moe_gating_top_k_softmax():
    x = torch.randn(8, 16, dtype=torch.float16, device=DEVICE, requires_grad=True)
    return torch.ops.npu.npu_moe_gating_top_k_softmax(x, None, 2)


if __name__ == "__main__":
    check("npu_add_rms_norm", t_add_rms_norm, grad_index=0)
    check("npu_apply_rotary_pos_emb", t_apply_rotary_pos_emb, grad_index=0)
    check("npu_interleave_rope", t_interleave_rope, grad_index=0)
    check("npu_moe_gating_top_k_softmax", t_moe_gating_top_k_softmax, grad_index=0)
    print("\n===== 完成 =====")
