# M-adapt Case 索引与选型指南

> A5 平台适配（把已有算子适配到 Ascend950 / Atlas A5 训练系列）的案例库。M-adapt 子模式专用。**M-adapt 触发前必先做 Step 0 状态预分类**（详见 `../../guides/m-adapt-guide.md` §2）。

## Case 索引

| # | Case | 场景 | 影响文件 | 关键词 |
|---|------|------|----------|--------|
| 1 | [npu_ciou-a5](npu_ciou-a5.md) | A5 适配：dtype 转换 + 反向硬报错 | yaml, cpp, meta, test, doc | A5, Ascend950, aclnnCIoU, GetSocVersion, fp16→fp32 |
| 2 | [npu_iou-a5](npu_iou-a5.md) | A5 适配：输出转置 + aclnn 新增参数 | yaml, cpp, meta, test, doc | A5, Ascend950, aclnnIou, GetSocVersion, transpose(0,1) |
| 3 | [npu_mish_backward-a5](npu_mish_backward-a5.md) | A5 适配：acl_op-only 简单 aclnn + 端到端整编/A5 验证 | yaml, cpp, meta, build | A5, Ascend950, aclnnMishBackward, GetSocVersion, cgroup/MAX_JOBS, autograd |
| 4 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) | A5 适配：aclnn 多版本分派 + `check_aclnn_kernel_available` 回退 + 特性 gate（领域参考） | yaml, cpp, meta, test | A5, Ascend950, V2/V3/V4 探测回退, M-coverage, 优雅报错 |
| 5 | [npu_swiglu-a5](npu_swiglu-a5.md) | A5 "已适配" 判别：`gen_opapi:` 结构化适配器（Step 0 状态 B，**不应触发 M-adapt**）（领域参考） | yaml (仅识别) | A5, gen_opapi, structured adapter, out-of-scope |

> 格式参考：`../../templates/case-template.md`。

## M-adapt Case Selection Guide

按 4 维度选最贴近你"要适配的算子"的 case。

### 维度 1：算子当前 op_api 状态（**决定是否进 M-adapt**）

| yaml 状态 | `opapi/` 是否有手写 cpp | 是否含 `gen_opapi:` | **Step 0 状态** | 看哪个 case |
|----------|----------------------|---------------------|----------------|------------|
| `op_api:` + 手写 cpp 已存在 | 有 `*Kernel(Npu)?OpApi.cpp` | — | **A. 已适配 (手写)** → 勿动 | [npu_iou-a5](npu_iou-a5.md), [npu_ciou-a5](npu_ciou-a5.md) |
| `op_api:` + 无手写 cpp | 无 | **有** | **B. 已适配 (gen_opapi)** → 勿动 | [npu_swiglu-a5](npu_swiglu-a5.md) |
| 仅 `acl_op:`, 无 `op_api:` | 无 | 无 | **C. 真候选** → 走 m-adapt | [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| 无 aclop 目录对应 cpp | — | — | **D. 该用 Create 模式** | (参见 [../create-mode/](../create-mode/README.md)) |

### 维度 2：aclnn 复杂度

| aclnn 形态 | 看哪个 case |
|-----------|-------------|
| 单一 aclnn，无版本变体 | [npu_iou-a5](npu_iou-a5.md), [npu_ciou-a5](npu_ciou-a5.md), [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| **多版本变体** (V2..VN)，需运行时探测 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |

### 维度 3：特殊处理需求

| 需要 | 看哪个 case |
|------|------------|
| **acl_op → op_api 全链路**（acl_op-only 算子补 op_api）| [npu_mish_backward-a5](npu_mish_backward-a5.md) |
| aclnn 输出 shape 与 user-facing 不一致（需 transpose）| [npu_iou-a5](npu_iou-a5.md) |
| aclnn 不支持原 dtype（fp16→fp32 转换）| [npu_ciou-a5](npu_ciou-a5.md) |
| aclnn **反向不支持**（硬报错模板）| [npu_ciou-a5](npu_ciou-a5.md) |
| **版本探测** (`check_aclnn_kernel_available`) + 回退 | [npu_moe_distribute_dispatch_v2-a5](npu_moe_distribute_dispatch_v2-a5.md) |
| **按入参特性 gate**（`sink.has_value` 决定 V5/V3）| 参考 `FlashAttentionV2KernelNpuOpApi.cpp`（本目录未单独 case）|

### 维度 4：环境与验证

| 需要 | 看哪个 case / 文档 |
|------|---------------------|
| 整编 torch_npu + 装 whl + A5 实测的**完整链路** | [npu_mish_backward-a5](npu_mish_backward-a5.md)（含 cgroup/MAX_JOBS 教训）|
| 编译期 cgroup/MAX_JOBS 坑 | `../../guides/build-guide.md` 顶部 + mish_backward case |
| M-coverage 完整流程 | `../../guides/m-adapt-guide.md` §4 |
| OpApi.cpp 顶部 boilerplate 必读 | `../../guides/m-adapt-cpp-patterns-guide.md` §1 |
| 端到端 autograd 联调 | [npu_mish_backward-a5](npu_mish_backward-a5.md) |

### 选 case 流程（自检清单）

1. **先看维度 1** → 确定 Step 0 状态；如果不是 C，**停**（改 case 看对应状态的识别细节）。
2. **C 状态下看维度 2** → 选 simple / multi-version case。
3. **按维度 3 加选 case**（如需 fp16→fp32 cast 则看 ciou）。
4. **若要端到端实测**（特别需要验证 aclnn 在 A5 真能跑 / 数值正确）→ 看 [npu_mish_backward-a5](npu_mish_backward-a5.md)。
5. **遇到 cgroup/MAX_JOBS 报错** → 看 mish_backward case 的"端到端 build 过程"段 + `../../guides/build-guide.md` 顶部。
