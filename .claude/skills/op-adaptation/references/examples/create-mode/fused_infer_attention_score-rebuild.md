# fused_infer_attention_score 全流程重建（Create 模式实测）

**type**: `workflow-review`

> **与 [fused_infer_attention_score-create.md](fused_infer_attention_score-create.md) 的分工**：那篇是 `domain-reference`，沉淀 attention 领域事实（14 layout / valueD 多路径 / dtype 4 路），且描述的是仓内**已存在**的 2 入口 + V1~V5 分派形态。本篇是 `workflow-review`，记录**在算子代码被完全删除后、仅凭 skill + 输入文件从零重建**的一次实测：跑通了哪些环节、哪些环节被卡、skill 的哪些约定真正起了作用。领域事实请读那篇，流程经验读本篇。
>
> **日期**：2026-07-25。**接入形态**：单入口 `npu_fused_infer_attention_score` → 直调 `aclnnFusedInferAttentionScoreV3`（未做 V2 分派，V2 签名未提供）+ ACLGraph 2 辅助接口。
>
> **完整总结**：`notes/summary_2026-07-25.md`。

## Background

用户删除了该算子在 op-plugin 的全部实现，要求**禁止从 git 恢复、禁止读 commit 反查删了什么**，纯粹按 skill 流程从 `aclnn_input/aclnnFusedInferAttentionScoreV3.md`（1502 行）+ `_def.cpp` 重新生成，用以实测 skill 的端到端能力。

这个约束让本次运行成为一次**对 skill 本身的验证**：任何"靠记忆/靠仓内残留补齐"的路径都被切断，产物质量完全取决于 guide 是否写全、脚本是否可用、CHECKLIST 是否真的能承载跨步骤状态。

## Solution

### 绑定权威的选择（C-parse 的第一个岔路）

输入有两份，规模差异极大：

| 来源 | 规模 | 角色 |
|------|------|------|
| `_def.cpp`（ge-op 定义） | 62 输入 / 16 属性 | **超集**，含 ge 图层参数 |
| V3 aclnn `GetWorkspaceSize` 签名（`.md` + CANN 头） | 27 tensor/array + 14 attr | **绑定权威** |

最终 torch 侧 **40 参数**取自 aclnn V3 签名，不是 `_def.cpp`。判据：op-plugin 绑定的是 aclnn C 接口，`_def.cpp` 的超集参数在 aclnn 层不可达。

**V5-only 参数明确排除**（`dequant_scale_query` / `learnable_sink` / `q_start_idx` / `kv_start_idx` / `query_quant_mode` / `pse_type` / `out_dtype`）——它们出现在 `.md` 的行文里，但不在 V3 签名内，照抄会产生调不通的死参数。

### C-confirm 的两个正交问题真正分叉了后续流程

skill 强调 Q1（torch.compile 入图）与 Q2（ACLGraph 手工 update）是**两个独立问题**，本次两者都为 YES，实测确认二者互不蕴含、各自驱动不同产物：

| | Q1 = YES 驱动 | Q2 = YES 驱动 |
|---|---|---|
| YAML | — | +2 条目（`.out` / `_get_max_workspace`）×2 分组 |
| C++ | — | +2 函数 + 1 共享 helper |
| Meta | **+1 `@impl .default`** | **不新增 meta**（handler 直调，不经 Dynamo） |
| build | 去掉 `--disable_torchair` | — |
| 测试/文档 | 图模式用例 + 示例 | eager 三路径对齐用例 |

若把两者混为一谈，最典型的错法是"因为做了 ACLGraph 就给 `.out` 也注册 meta"——skill 明确约定不注册，C-verify 也把辅助接口 meta 缺失列为**预期**、不 BLOCK。本次严格遵守，构建与注册校验均通过。

### 假冲突的识别（C-confirm 必须先清 ⚠️ 项）

C-parse 报了 4 条冲突，逐条查证后**全部是命名风格造成的假冲突**：

| parse 报的冲突 | 实情 |
|---|---|
| `dequant_scale1/2` | aclnn 侧名为 `deqScale1/2`，同参数不同命名风格 |
| `scale` | aclnn 侧名为 `scaleValue`，同参数 |

教训：跨层比对参数时，**命名风格差异会被机械匹配器报成冲突**，必须人工按语义核对再决定纳入/排除，不能见冲突就删参数。

## File Changes

| File | Action | Details |
|------|--------|---------|
| `op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp` | add | 1 共享 helper + 3 对外函数 |
| `op_plugin/config/op_plugin_functions.yaml` | modify | +6 条目（`custom:` / `symint:` 各 3） |
| `op_plugin/python/meta/_meta_registrations.py` | modify | +1 `@impl` + 6 helper 镜像 |
| `ci/build.sh` | modify | 去掉 `--disable_torchair`（Q1=YES 所需） |
| `codegen/templates/_op_plugin_docs.py` | modify | +1 `_add_torch_npu_docstr` |
| `docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md` | add | 新文档页 |
| `docs/zh/custom_APIs/{menu_Pytorch_API.md, torch_npu/torch_npu_list.md}` | modify | 导航 2 处 |
| `test/test_custom_ops/test_npu_fused_infer_attention_score.py` | add | 13 正向 + 4 负向 |
| `test/core_tests/test_fake_tensor.py` | modify | +1 测试类（3 meta 用例） |
| `test/core_tests/torch_npu_OpApi_schema_all.json`、`test/allowlist_for_publicAPI.json` | modify | 兼容性基线 |

### SymInt 三条目同步（本次最易漏的结构点）

`actual_seq_lengths` / `actual_seq_lengths_kv` / `actual_shared_prefix_len` 为 `SymInt[]?`，functional 进 `symint:` 分组后，**`.out` 与 `_get_max_workspace` 必须同步登记**，否则 codegen 生成的实现名与 C++ 定义名对不上：

```yaml
# custom: 与 symint: 两组各 3 条，共 6 条
- func: npu_fused_infer_attention_score(Tensor query, Tensor[] key, Tensor[] value, *, ..., SymInt[]? actual_seq_lengths=None, ...) -> (Tensor, Tensor)
- func: npu_fused_infer_attention_score.out(... , Tensor? workspace=None, Tensor(a!) attention_out, Tensor(b!) softmax_lse) -> (Tensor(a!), Tensor(b!))
- func: _npu_fused_infer_attention_score_get_max_workspace(...) -> Tensor
```

C++ 侧三个函数名统一带 `_symint` 后缀：

```cpp
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_symint(...);
std::tuple<at::Tensor&, at::Tensor&> npu_fused_infer_attention_score_out_symint(...);
at::Tensor _npu_fused_infer_attention_score_get_max_workspace_symint(...);
```

### 共享 helper 是 ACLGraph 正确性的关键

```cpp
// op_api:: 命名空间普通函数，functional 与 _get_max_workspace 共用
std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor(...);
```

必须共用而非各写一份：`_get_max_workspace` 探测 workspace 时构造的输出描述，如果与实跑路径的输出描述不一致，探到的 workspace 尺寸就是错的。

### workspace 字节数

`.out` 的 workspace 路径按 `numel() * element_size()` 算字节数。本次 `_get_max_workspace` 返回 1D `at::kByte` Tensor，使 numel 恰等于字节数——这是 cpp-guide 的新契约，比仓内历史实现（返回 query dtype 的 Tensor）更不易错。

## Pitfalls

1. **ACLGraph EXEC 宏的头文件依赖（本次唯一一次构建失败的根因）**
   第 1 次构建报 4 条错误，表象是 **`aclnnFusedInferAttentionScoreV3` 未声明**——极易误判为"CANN 头文件缺失/版本不对"，从而去翻 CANN 安装目录。实际是**级联报错**：该 aclnn 名是被未声明宏消费的裸标识符，真正未声明的是 `EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD` / `EXEC_GET_MAX_WORKSPACE_CMD`。

   根因：这两个宏定义在 `op_plugin/utils/custom_functions/opapi/update_op_api_common.h`，**不在** `op_api_common.h`。

   定位方法（可复用）：
   ```bash
   grep -rl "define EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD" op_plugin/
   # → 定位头文件，再对照同类 FlashAttentionKernelNpuOpApi.cpp 的 include 列表
   ```
   补 include 后第 2 次构建 0 error。**遇到"aclnn 名未声明"先查是不是宏没声明，再怀疑 CANN。**

2. **共享主机上用 `pgrep -f ci/build.sh` 判断自己的构建是否结束**
   会匹配到其他用户的构建进程。skill 已明确要求：启动时捕获 PID（`nohup bash ci/build.sh ... & echo $!`），用 `kill -0 <PID>` 轮询该 PID。本次遵守，构建状态判断无误。

3. **CHECKLIST 首行 opName 写成草稿名**
   曾写为 `npu_fused_infer_attention_score_v3`（`.md` 文件名带 V3，但那只是 aclnn 版本号，torch API 名不带）。后果不是"看着别扭"——`auto_adapt.py` 等自动化**按名取值**，会建出错名的 pta 目录并误报 "not found in `_op_plugin_docs.py`"。由 op-delimitation 子代理发现并修正。
   **CHECKLIST 的字段是被脚本消费的数据，不是给人看的备注。**

4. **ATK 基线对齐：`actual_shared_prefix_len` 必须置 `None`**
   ATK 生成器给出了 `[32, 4096]`，但 aclnn 标杆侧 `input_args[23]`（= `actualSharedPrefixLenOptional`）显式置空。若照抄生成器的值，两侧在解**不同的题**，会伪造出一个不存在的精度失败。
   **精度比对前必须逐参数核对两侧是否同题。**

5. **C-constraints 的 uncovered 不等于漏实现**
   checker 报 162 条中 71 条 uncovered。逐条判读后确认**全部是 device-side / kernel-tiling 约束**（SoC 相关的 N/D 维界、sparse/prefix/global 模式限制、rope d=64），由 aclnn V3 内核校验并返错码，不是适配层契约。适配层可实现的结构子集已全部 `TORCH_CHECK` 覆盖（layout 合法性、rank-per-layout、TensorList 非空、valueD 维界、antiquant_mode 一致性）。
   **不要为了把覆盖率刷到 100% 而在适配层复刻内核的 tiling 约束**——既写不准，又会拦掉合法输入。

6. **C-verify 脚本对 `_symint` 后缀与 `@impl` 装饰有 false-negative**
   脚本报 cpp/meta "not found"，实为解析器识别不了 `_symint` 后缀函数名和 `@impl` 装饰器。人工交叉核对结论：YAML ↔ C++ 40 参数完全一致（diff SAME），YAML ↔ Meta 40 参数一致，shape/dtype 6 项逻辑逐项对齐。
   **脚本给草稿，LLM 做判读——这正是 Scripts-First Protocol 的设计意图，不要见 not found 就改代码。**

7. **设备不可用时不要伪造执行结果**
   本次 NPU 8 卡全被其他用户占满（`aclInit` error 507899 / `Resource_Busy(EL0005)` / `npu-smi` ret -8020），后台重试 40/40 次全 BUSY。失败点在 `aclInit`（驱动层，**选卡之前**），`Ran 0 tests`——尚未进入任何测试方法。
   按 op-test 三分支归因法判为**第 3 类基础设施问题**，非用例缺陷、非适配源码缺陷。UT 17 条与 ATK `atk task` 均如实标记为**未执行**，summary 中以 ⚠️ 记录而非计入通过。
   **静态核对（子代理确认 TND shape 与 40 入参顺序均正确）不能替代执行证据**，不能因为"看起来没问题"就把未执行写成通过。

## 本次可复用的检查清单

从零重建时按序确认，每项都在本次踩到或用到：

- [ ] 绑定权威取 aclnn `GetWorkspaceSize` 签名，不取 `_def.cpp` 超集
- [ ] 高版本 only 参数明确排除，不照抄 `.md` 行文
- [ ] parse 报的参数冲突逐条按语义核对（命名风格差异会产生假冲突）
- [ ] Q1 / Q2 分别回答，不互相蕴含；Q2 的辅助接口**不注册 meta**
- [ ] SymInt 参数在 `.out` / `_get_max_workspace` 同步登记，实现名带 `_symint`
- [ ] 输出构造 helper 由 functional 与 `_get_max_workspace` 共用
- [ ] ACLGraph EXEC 宏 include `update_op_api_common.h`
- [ ] Q1=YES → `ci/build.sh` 去掉 `--disable_torchair`，构建日志确认 `Built target _torchair`
- [ ] 构建轮询用捕获的 PID + `kill -0`，不用 `pgrep -f`
- [ ] CHECKLIST opName 为最终 torch API 名（被脚本消费）
- [ ] 未执行的验证项以 ⚠️ 如实记录，不计入通过
