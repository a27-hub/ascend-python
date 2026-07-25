# Attention 算子 ACLGraph 手工 Update 接入

> **适用**：attention 类（FA/IFA/MLA/paged-attention）带动态 seq-len host 参数的算子，需 ACLGraph 手工 task-group update 支持。由 C-confirm Q2 触发（见 `confirm-guide.md`）。
>
> 本 guide 为 attention 类算子 ACLGraph 手工 update 的流程映射入口（skill 步骤映射 + 仓内实证；规则在各步骤 guide）。provenance：源自 torch_npu 业务专家 2026-07-22 设计文档，基线 `torch_npu@2916e65de` / `op-plugin@29fca88f`，操作性内容已下沉本 guide 与各步骤 guide。

## 适用判定

attention 家族 + 动态 seq-len host 参数（`actual_seq_*` / `context_lens` / `seq_lengths` / `prefix`）+ C-confirm Q2 确认 YES。动态参数形态见 `cpp-guide.md` §host 序列参数处理细则（`SymInt[]?` / CPU `Tensor?` / `int[]?`）。

## 交付物（每逻辑算子）

**只新增 2 个 PTA 接口**：

| 接口 | 作用 |
|---|---|
| `npu_xxx.out` | functional 参数段照抄 + 尾部 `Tensor? workspace=None` + alias 输出 |
| `_npu_xxx_get_max_workspace` | functional 全参数，`-> Tensor`（新接口返回 1D `at::kByte`） |

- 另需算子 C++ 内的**输出构造 helper**（供 functional 与 get-max 复用）。
- **`_infer_output` 不在手工 update 范围**：新算子默认不建；仓内已有（IFA）的保留可用；确需新建须作为单独需求评审。
- **autograd 场景**：forward 和 backward 各为一个逻辑算子，grad 按独立逻辑算子重复同一 2 接口流程。

## 参考实现与覆盖声明

**仓内唯一 conformance 参考 = IFA v1/v2**：

- `op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp`（v1）
- `op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp`（v2）

> ⚠️ **FA v3 覆盖声明**：FA v3（`npu_fusion_attention_v3` / `_grad_v3`）曾列参考实现、其 workspace 路径曾引为正例——经业务确认其**未完成真实 ACLGraph 接入验证**。其具体实现细节（`prefix` 的 SymInt→`int[]` 降级、全位置参数的 `*,` 插入形式、6 输出取 3 的 `_infer_output` 子集、TND 限定）**不作依据**。

## 参数构造要点（仓内实证；权威约束见 `yaml-guide.md` §Attention ACLGraph Interfaces）

- `.out` 在 workspace 之前的参数与 functional 逐字一致（名称/顺序/类型/默认值）；尾部追加 `Tensor? workspace=None` + `Tensor(a!)/(b!)/...` 输出，返回同 alias。
- **SymInt 参数默认照抄保留**（IFA 形态），并同步 `symint:` 分组、C++ 函数名带 `_symint` 后缀。`SymInt[]?`→`int[]?` 降级仓内仅 FA v3 一例且未验证——**不作先例**，确需降级必须用户明确确认并记入 CHECKLIST。
- 全位置参数 functional（无 `*,`）追加形式的推荐值：`Tensor? workspace=None, *, Tensor(a!)...`（alias 输出必须 keyword-only），C-confirm 确认。
- `op_api` / `internal_format_opapi` / `device_check` 按目标算子真实支持范围填，**不得照抄其他 FA 算子**。

## 原十五步 → skill 步骤映射

| 原十五步 | skill 步骤 |
|---|---|
| 1 读取 functional schema / 2 读取 functional C++ | C-parse（Create）/ M-discover（Modify） |
| 3 输出构造抽公共 helper / 7 `.out` 普通路径 / 8 get-max / 9 `.out` workspace 路径 | C-cpp（helper 先行；普通路径 → get-max → workspace 路径顺序实现） |
| 4 `.out` schema / 5 get-max schema / 6 SymInt 分组 | C-yaml（由 `gen-aclgraph` 出草稿，⚠️ codegen-type 逐项确认） |
| 预处理审查 → 限制转 `TORCH_CHECK` | C-cpp 实现 + C-verify 核对项 |
| 10 codegen 产物检查（不手改） | C-build |
| 11 schema 兼容性基线 | C-baseline |
| 12 eager 三路径（functional / out(None) / out(max_ws)） | C-test |
| 13/14 torch_npu task-group、连续 update 测试 | **移交 torch_npu**，C-summary 标注，非本 skill 范围 |
| 15 版本矩阵 | C-verify 人工核对（`verify-guide.md` §ACLGraph Structural Verification 三参数表） |

## Meta 与 Derivatives

- `.out` / `_get_max_workspace` **均不注册 meta**，沿用 `.default` 既有 meta（由 torch_npu handler 直调，不经 Dynamo dispatch）。C-verify 发现其无 meta 属**预期**，不报 BLOCK。
- 不为 `.out` 新增 derivatives 绑定。

## 范围边界

op-plugin 侧交付 = YAML schema + C++ 实现 + schema 兼容性基线 + eager 测试。不得修改：`torch_npu/npu/graphs.py`、`_npugraph_handlers/`、`Graph.cpp` 算子专用 pybind、`update_op_api_common.h` 公共宏。

## 接入 Checklist

```
□ 1. C-confirm Q2=YES，门禁逐条确认（CANN GetMaxWorkspaceSize 存在为硬门）
□ 2. C-yaml：.out + _get_max_workspace 两条目（gen-aclgraph 草稿 + ⚠️ codegen-type 确认）
□ 3. C-cpp：helper → .out 普通路径 → get-max → .out workspace 路径；预处理限制转 TORCH_CHECK
□ 4. C-verify：签名结构检查 + 三条 Host API 参数表人工核对 + 预处理审查
□ 5. Meta 不新增；Derivatives 不为 .out 新增绑定
□ 6. C-test：eager 三路径对齐（参照 test_npu_fused_infer_attention_score*_with_workspace.py）
□ 7. C-baseline：schema 基线只加本次新增条目；C-summary 标注 torch_npu 侧测试移交
```
