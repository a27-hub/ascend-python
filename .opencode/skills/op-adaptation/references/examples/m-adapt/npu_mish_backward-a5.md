# npu_mish_backward A5 平台适配（M-adapt 案例 · 端到端成功先例）

> 本案例是 op-adaptation skill 首次走通"**生成 + 整编 + A5 实测**"全链路的成功先例。
> 适用于 acl_op-only、签名简单、aclnn 已在 A5 kernel 可用的算子补 op_api 路径。
>
> **补充说明**：`npu_mish` / `npu_mish_backward` 本身已 deprecated（同 `npu_silu`），但作为 skill 自检探针价值满分。本案例的核心价值不是"复用到生产"，而是"skill 全链路可复现"。

**type**: `workflow-review`

**基线环境**：op-plugin 26.0.0 / torch 2.10.0 / py3.12 / CANN 9.0.0-beta.2 / 硬件 `Ascend950PR`

## Background

`npu_mish_backward(grad, input) -> grad_input` 计算 Mish 激活的反向梯度。op-plugin 中：
- 前向 `npu_mish` 已走 `op_api: all_version`（手写 `MishKernelNpuOpApi.cpp`），A5 跑通；
- 但 `npu_mish_backward` 在 yaml 中**只有 `acl_op: all_version`**，**无 `op_api:`**、无 `op_plugin/ops/opapi/MishBackwardKernelNpuOpApi.cpp`；
- `aclops/` 也没有 `MishBackwardKernelNpu.cpp`（acl_op 实际定义在 `MishKernelNpu.cpp` line 23，命名 `acl_op::npu_mish_backward`，是个**孤儿 backward**：forward/op_api 走了，backward 还在老 ACL 路径）；
- `derivatives.yaml` 已绑定 `npu_mish → npu_mish_backward`（line 205-207）→ autograd 链可达。

**A5 上的真实问题**：调 `npu_mish_backward(grad, input)` 在 `Ascend950PR` 上**直接崩溃**：
```
RuntimeError: Current device only support aclnn operator,
but current operator npu_mish_backward do not have aclnn implementation
```
（错误模式与 `npu_silu` 同根因：只有老 ACL 路径，A5 不支持。）

**CANN 侧覆盖度（已 M-coverage 校验）**：
- `aclnnMishBackwardGetWorkspaceSize` 符号在 CANN 9.0.0 的 `aclnn_mish_backward.h` 中；
- ascend950 opp config 登记了 `mishgrad` kernel（注意底层 kernel 名 ≠ aclnn 名：`aclnnMishBackward` → `MishGrad`）；
- → M-coverage 通过，**可直接走硬编 `EXEC_NPU_CMD`**，无需 `check_aclnn_kernel_available` 回退（与 moe_dispatch 的版本分派模式不同）。

## Solution

按 m-adapt 标准三段式（段 1 SoC 路由 / 段 2 aclnn / 段 3 特殊处理），简洁实现：

| 段 | 内容 |
|----|------|
| 段 1 | `if (GetSocVersion() < Ascend950) return acl_op::npu_mish_backward(grad, input);` |
| 段 2.2-2.3 | `at::Tensor grad_input = npu_preparation::apply_tensor(input);`（output 与 input 同 shape、同 dtype）|
| 段 2.4 | `EXEC_NPU_CMD(aclnnMishBackward, grad, input, grad_input);` |
| 段 3 | 无（无 dtype 转换、无 shape 差异、无 transpose）|

## File Changes

| File | Action | Details |
|------|--------|---------|
| `op_plugin/ops/opapi/MishBackwardKernelNpuOpApi.cpp` | 新增 | 38 行（含 license），段 1/段 2 三段式骨架 |
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | `npu_mish_backward` 加 `op_api: all_version` |
| `op_plugin/python/meta/_meta_registrations.py` | 新增 | `npu_mish_backward_meta(grad, input) -> empty_like(input)` 一行 |

### C++ 关键代码（完整 OpApi 文件仅 38 行）

```cpp
// (license header omitted)

#include "op_plugin/OpApiInterface.h"      // ← 必须三件套
#include "op_plugin/AclOpsInterface.h"      // ← 段 1 ACL 兜底要它, 不能省
#include "op_plugin/utils/op_api_common.h" // ← EXEC_NPU_CMD 宏在这里

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_mish_backward(const at::Tensor& grad, const at::Tensor& input)
{
    // 段 1: SoC 路由 — 老平台(A2/A3) 走 ACL 路径, A5 走 ACLNN
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
        return acl_op::npu_mish_backward(grad, input);
    }

    // 段 2.2-2.3: infer output size + pre-allocate
    at::Tensor grad_input = npu_preparation::apply_tensor(input);

    // 段 2.4: invoke aclnn
    EXEC_NPU_CMD(aclnnMishBackward, grad, input, grad_input);

    return grad_input;
}
}  // namespace op_api
```

### YAML 改动

```yaml
- func: npu_mish_backward(Tensor grad, Tensor input) -> Tensor
  acl_op: all_version
  op_api: all_version    # ← 新增 (after)
```

### Meta 改动（单行）

```python
@impl(m, "npu_mish_backward")
def npu_mish_backward_meta(grad, input):
    return torch.empty_like(input)
```

## Pitfalls (从这次端到端踩到的坑)

- **`AclOpsInterface.h` 不能省**：即使 A5 路径只走 aclnn，段 1 的 `acl_op::npu_mish_backward(grad, input)` 老平台兜底需要它。漏 include → 链接报 missing symbol `acl_op::npu_mish_backward`。
- **namespace 必须是 `op_api`、using 必须照搬**：与 opapi 目录其他 cpp 文件不一致会破坏 codegen 的 `gen_opapi` 链接表。
- **opapi 文件名 `*KernelNpuOpApi.cpp` 与 `*KernelOpApi.cpp` 两种命名都存在**（365 vs 28 个），按 cpp 文件名 grep 检测时要用 `grep -rln`，不要写死 `ls *KernelNpuOpApi.cpp`（详见 `m-adapt-guide.md` §3 条件 2）。
- **编译期 OOM 死循环**（**最大教训**）：见下方"端到端 build 过程"。

## A5 Platform Specific Info (M-adapt only)

### A5 adaptation basics

| Field | Value |
|-------|-------|
| Target SoC | Ascend950 (Atlas A5 training series) |
| aclnn operator | `aclnnMishBackward` (底 kernel 名 `MishGrad`) |
| Routing | `GetSocVersion()` (not `DO_COMPATIBILITY`) |
| Related PR | (内部版本, 26.0.0 分支) |

### A5-specific handling

| Aspect | Description | Applies? |
|--------|-------------|----------|
| dtype cast (fp16→fp32→fp16) | aclnnMishBackward 原生支持 fp16/fp32/bf16 | **否** |
| output transpose | aclnn 输出与 user-facing 一致 | 否 |
| backward unsupported | 不适用（本例就是 backward 适配）| N/A |
| multi-output internal API | 无 | 否（单输出）|
| aclnn new params | 无 | 否 |
| A5-specific constraints | 无特殊输入规模约束 | 否 |
| **version dispatch (check_aclnn_kernel_available)** | M-coverage 验证 ascend950 有 `mishgrad` → **不需要** | **否（已确认）** |
| **runtime kernel gating** | 同上 | **否** |

### A5 test design notes

- **平台隔离**：本案例未引入 `op_test` 套件；端到端验证直接用 Python `torch_npu.npu_mish_backward` 调 aclnn 路径（参 `C-build/C-test` 段）。
- **精度断言**：
  - fp32：手算公式 `d/dx mish = tanh(softplus(x)) + x·sech²(softplus(x))·sigmoid(x)` 作参考；误差 ~6e-8。
  - fp16/bf16：与手算公式 max_abs_err ≈ 0（aclnn 内部按 fp32 算再降精度）。
- **autograd 端到端**：`F.mish(x).sum().backward()` 经 `derivatives.yaml` 绑定触发新 `npu_mish_backward` → 验证 `x.grad` shape 正确、finite。

#### 本精度策略**不适用**的算子

"vs 手算公式 max_abs_err" 这种 deterministic 算子的标准精度断言**不适用于**：

| 算子类型 | 例子 | yaml 标识 | 替代策略 |
|----------|------|----------|---------|
| **非确定性算子**（含随机性/seed）| `npu_dropout_do_mask`, `npu_dropout_with_add_softmax` | `tags: nondeterministic_seeded` | 不能用固定期望值比对；用 `cpu_op_exec` 走 CPU 路径作参考（结果会因 seed 漂移但应在容差内）|
| **MoE 路由类**（dispatch/combine）| `npu_moe_distribute_dispatch*` | （无特殊 tag） | 不能手算公式；用 `cpu_op_exec` 或固定 routing map 比对 expert 分配 |
| **多版本 aclnn 分派**（V4/V3/V2）| 见 [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) | yaml 区段 | 按 CANN 版本分别测；不适用单一"手算公式"标准 |

> **识别技巧**：搜 yaml 中 `tags: nondeterministic_seeded`，命中即改用 stochastic 容差策略；或搜 `npu_moe_/npu_routing_/npu_dropout_` 前缀算子——基本都是非确定性，**强制走 cpu_op_exec**。

详细 cpu_op_exec 模式见 [npu_swi_glu_quant-create](../create-mode/npu_swi_glu_quant-create.md) Pitfalls 段（quant-scale 语义对比也用该模式）。

## 端到端 build 过程（重要：M-adapt 首次全链路踩坑记录）

本节专门记录 cgroup/MAX_JOBS 教训，**这个坑 skill 此前完全没提**。

### 第一次 build 失败：OOM 死循环

- 启 `bash ci/build.sh --python=3.12 --pytorch=v2.10.0-26.0.0`（后台）；
- 5 分钟后 build.log 出现 59 条 `c++: fatal error: Killed signal terminated program cc1plus`；
- 0 active 进程，setup.py 死循环重试同一文件后退出。

**根因**：
- `setup.py` line 369：`max_jobs = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))`；
- 本机 `nproc = 128`，cgroup 限 32G（`/sys/fs/cgroup/memory.max`）；
- 单 `cc1plus` 内存峰值 > cgroup limit → OOM killer SIGKILL → setup.py 重试 → 再被杀 → 死循环。

### 第二次 build 成功：显式 MAX_JOBS

```bash
export MAX_JOBS=4
nohup stdbuf -oL -eL bash ci/build.sh --python=3.12 --pytorch=v2.10.0-26.0.0 \
  > build_log/build2.log 2>&1 &
```

进度时间线（4 并发）：
- 7% (aclops 早期) → 36% (aclops 中段) → **62% (`MishBackwardKernelNpuOpApi.cpp` 编过, 0 error)**
- → 69% (进入 opapi/ 段) → 82% (进入 torch_npu/csrc 段) → 89% (开始 linking) → **100% (whl 37MB)**
- 总耗时约 **18 分钟**（首次跑全量，若 `build/pytorch` 已有缓存可复用则更快）

### 装 whl + A5 实测

```bash
pip install --force-reinstall --no-deps \
  build/pytorch/dist/torch_npu-2.10.0.post2-cp312-cp312-linux_x86_64.whl
```

直接调：
```python
import torch, torch_npu
g = torch.randn(4,8,dtype=torch.float16).npu()
x = torch.randn(4,8,dtype=torch.float16).npu()
out = torch_npu.npu_mish_backward(g, x)  # ← 适配前 A5 崩, 适配后跑通✅
```

端到端 autograd：
```python
x = torch.randn(3,5, dtype=torch.float16, device='npu', requires_grad=True)
y = torch.nn.functional.mish(x)
y.sum().backward()  # ← 经 derivatives.yaml 触发新 aclnn 路径
assert torch.isfinite(x.grad).all()
```

**before/after**：
- 适配前：A5 上 `Current device only support aclnn operator, but current operator npu_mish_backward do not have aclnn implementation`
- 适配后：跑通，fp32 误差 6e-8，fp16/bf16 误差 0，端到端 autograd 链正常

### 给后来者的 3 条 build 铁律

1. **先 `cat /sys/fs/cgroup/memory.max` 查 cgroup**（无 cgroup 也无害，但容器内必有）；
2. **导出 `MAX_JOBS = min(nproc, cgroup_mem_GB / 4)`**（4 核~4-8G 内存峰值是稳的）；
3. **看到 `Killed signal terminated program cc1plus` 立刻停并降并发**——已 clone 的 `build/pytorch` 可复用，不必从头重 clone。

## 适用范围与互补案例

**本案例覆盖**：
- acl_op-only 算子（无 op_api、无手写 OpApi.cpp）
- **单一 aclnn、aclnn 在 A5 已有 kernel**（M-coverage 硬编通过）
- 无版本分派、无 dtype 转换、无 shape 差异、无新参数
- 端到端验证链路（整编 + whl 安装 + A5 实测 + autograd 联调）

**不覆盖的请看**：
| 场景 | 推荐看 |
|------|--------|
| aclnn **多版本**(V2..VN) + `check_aclnn_kernel_available` 回退 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |
| 算子**已通过 gen_opapi 适配**（不应触发 M-adapt）| [npu_swiglu-a5](npu_swiglu-a5.md) |
| 简单 aclnn + 输出转置（user-facing shape 差异）| [npu_iou-a5](npu_iou-a5.md) |
| 简单 aclnn + fp16→fp32 cast + backward 硬报错 | [npu_ciou-a5](npu_ciou-a5.md) |
| cgroup/MAX_JOBS 详细推导 | `build-guide.md` 顶部 |
| M-coverage 完整校验流程 | `m-adapt-guide.md` §4 |
| OpApi.cpp boilerplate 必读 | `m-adapt-cpp-patterns-guide.md` §1 |
