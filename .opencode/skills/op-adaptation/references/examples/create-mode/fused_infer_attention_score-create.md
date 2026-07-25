# fused_infer_attention_score 领域事实参考

**type**: `domain-reference`

> 该算子已在 op-plugin 完成对接并持续演进，本文件不复述"如何新建"，而是沉淀 attention 类算子对接中反复用到的领域事实。
>
> **用途**：attention 类多参数融合算子的 shape 推导 / dtype 推导 / aclnn 版本分派决策参考。
>
> **触发**：当接入或修改的算子属于 attention 家族（含 layout 维度、PSE/Mask/Quant 分组、aclnn 多版本分派、PageAttention）时，在 C-confirm 后读取本案例。
>
> **事实基准**：所有 shape 公式、参数名、layout 取值、dtype 规则均已对两侧源码核对到行号。核心源码：
> - Ascend C 算子：`ops-transformer/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp`
> - aclnn API：`ops-transformer/attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score_v{1..5}.{h,cpp}`
> - op-plugin 对接：`op-plugin/op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp`（v1 入口）、`FusedInferAttentionScoreV2KernelNpuOpApi.cpp`（v2 入口）
>
> **last verified**：op-plugin @ `8fa71cf8` (2026-04-20)，ops-transformer @ `e32fbe542` (2026-06-08)。下文行号以函数名为锚、行号仅供参考；源码演进后需按函数名重新核对。

---

## Background

`aclnnFusedInferAttentionScore` 是适配增量（ IncreFlashAttention）与全量（PromptFlashAttention）推理场景的 FlashAttention 融合算子。它是 CANN 体系中参数最多、分支最复杂的算子之一，融合了 attention 计算、PSE 偏移、attention mask、稀疏模式、量化/反量化、KV-Cache（PageAttention）、shared prefix、RoPE 等多组正交特性。

**对接现状（截至本案例撰写时）**：

| 层级 | 现状 |
|------|------|
| op-plugin 入口算子 | **2 个**：`npu_fused_infer_attention_score`（v1 入口）、`npu_fused_infer_attention_score_v2`（v2 入口，参数命名体系不同） |
| aclnn 版本 | **5 个**：V1 / V2 / V3 / V4 / V5；其中 V1、V3 已标 `deprecated → 由 V5 替代（2026.12）` |
| op-plugin 当前分派 | v1 入口按 `is_gte_cann_version_810rc1()` 分派 V2 / V3；v2 入口调 V4；**均尚未接 V5** |

> 因此本案例记录的是"已沉淀的领域事实"，不是"如何新建"——它对后续接入 V5、或接入同类 attention 算子有直接参考价值。

---

## 算子结构与三层映射（Solution）

接入 attention 类算子时，最容易混淆的是"哪份源码对应哪个层级"。三层映射如下：

### Layer 1 — Ascend C 算子定义（算子本体）

源码：`ops-transformer/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp:26-560`

| 项 | 数量 | 说明 |
|----|------|------|
| Input | **31 个** | `query` ParamType=REQUIRED；`key`/`value`=DYNAMIC；其余 28 个均为 OPTIONAL（pse_shift、atten_mask、各类 quant/antiquant scale、block_table、*_rope、*_shared_prefix 等） |
| Output | **2 个** | `attention_out`(REQUIRED)、`softmax_lse`(REQUIRED) |
| Attr | **16 个** | `num_heads`=REQUIRED；其余 OPTIONAL，含 `input_layout` 默认 `"BSH"`、`num_key_value_heads` 默认 0、`softmax_lse_flag` 默认 false、`sparse_mode`/`antiquant_mode`/`inner_precise` 等 |
| 芯片配置 | 多套 | `ascend910b` / `ascend910_93` / `mc62` 共用一套 `aicore_config`；`95`（950PR/950DT）有**独立的 `aicore_config_95`**（dtype 组合不同） |

> ⚠️ 注意：本算子 `_def.cpp` 里**没有 `.Version()` 标记**——它的多版本是体现在 aclnn API 层（V1~V5），而不是 `_def.cpp` 的 `.Version()` 机制里。此场景由 `parse-guide.md` § Feature-Group & Dimension Analysis → "aclnn-layer version variants" 专门覆盖：无 `.Version()` 时应枚举 aclnn 多版本头文件并在 C-confirm 确认。

### Layer 2 — aclnn API（对外稳定接口）

源码：`ops-transformer/attention/fused_infer_attention_score/op_api/`

5 个版本，每个版本一对 `.h/.cpp`，核心是 `aclnnFusedInferAttentionScoreV{n}GetWorkspaceSize` 签名：

| 版本 | 状态 | 相对前一版的主要变化 |
|------|------|----------------------|
| V1 | deprecated → V5 (2026.12) | 基线版本 |
| V2 | 在用 | 新增 quant/antiquant 完整体系 |
| V3 | deprecated → V5 (2026.12) | 新增 `queryPaddingSize` / `kvPaddingSize` / shared prefix / RoPE（`queryRope`/`keyRope`/`keyRopeAntiquantScale`） |
| V4 | 在用 | v2 入口专用变体（参数命名体系不同，见下） |
| V5 | 在用（最新） | 新增 `dequantScaleQuery` / `learnableSink` / `qStartIdx` / `kvStartIdx` / `queryQuantMode` / `pseType` |

> **V5 相对 V3 的参数差异**（来自 `aclnn_fused_infer_attention_score_v5.h:23-41`）：新增 6 个参数。若后续要接 V5，需对照 V5 头文件逐项补齐。

> **版本间的两类命名体系**：V1/V2/V3/V5 用 `dequant_scale1`/`quant_scale2`/`actual_seq_lengths` 体系；V4（v2 入口）用 `dequant_scale_query`/`quant_scale_out`/`actual_seq_qlen` 体系——两套参数语义相近但命名不同，不能混用。

### Layer 3 — op-plugin 对接（torch_npu 绑定）

源码：`op-plugin/op_plugin/ops/opapi/FusedInferAttentionScore{,V2}KernelNpuOpApi.cpp`

| 入口算子 | 文件 | 分派的 aclnn | 分派依据 |
|----------|------|-------------|----------|
| `npu_fused_infer_attention_score` | `FusedInferAttentionScoreKernelNpuOpApi.cpp` | V3 / V2 | `is_gte_cann_version_810rc1()` ? V3 : V2（`:414-424`） |
| `npu_fused_infer_attention_score_v2` | `FusedInferAttentionScoreV2KernelNpuOpApi.cpp` | V4 | 直接调（v2 参数体系与 V4 对齐） |

关键设计：**shape 推导与 aclnn 调用解耦**——两个入口都把输出张量构造抽到独立函数 `construct_fia_output_tensor`（`:294-357`），由它先算出 `attention_out` 与 `softmax_lse` 的 shape/dtype，再 `EXEC_NPU_*_CMD` 调 aclnn。这是应对 attention 多 layout/多 dtype 的关键架构。

---

## 领域事实（File Changes → Domain Facts）

以下 7 节是 attention 类算子对接中最易出错、最 attention 专属的部分。每一节给"是什么 + 源码位置 + 对 op-plugin 实现的指导"。

### 1. Layout 双变量体系（最核心的复杂度来源）

`input_layout` 不是单值概念，而是**两个独立变量的组合**：`query_layout`（query 张量的排列）与 `attention_out_layout`（输出张量的排列）。源码 `get_query_and_attention_out_layout`（`FusedInferAttentionScoreKernelNpuOpApi.cpp:42-90`）的 `LAYOUT_MAP` 共 **14 个取值**：

| 类别 | 取值 | query_layout | attention_out_layout | query 期望维数 |
|------|------|--------------|----------------------|----------------|
| query↔out 相同（6 个） | `BSH` | BSH | BSH | 3 |
| | `BSND` | BSND | BSND | 4 |
| | `BNSD` | BNSD | BNSD | 4 |
| | `TND` | TND | TND | 3 |
| | `NTD` | NTD | NTD | 3 |
| | `NSD` | NSD | NSD | 3 |
| query↔out **不同**（8 个） | `BNSD_BSND` | BNSD | BSND | 4 |
| | `BSH_BNSD` | BSH | BNSD | 3 |
| | `BSND_BNSD` | BSND | BNSD | 4 |
| | `NTD_TND` | NTD | TND | 3 |
| | `BSH_NBSD` | BSH | NBSD | 3 |
| | `BSND_NBSD` | BSND | NBSD | 4 |
| | `BNSD_NBSD` | BNSD | NBSD | 4 |
| | `TND_NTD` | TND | NTD | 3 |

**对 op-plugin 实现的指导**：shape 推导必须**先解析出 query_layout 与 attention_out_layout 两个独立值**，再用 `attention_out_layout` 决定输出形状（见下表）。绝不能假设"输入 layout = 输出 layout"。`infer_attention_out_shape`（`:198-269`）按 `attention_out_layout` 分支，各分支公式：

| attention_out_layout | attention_out shape |
|----------------------|---------------------|
| `BSH` | `{b, s1, num_heads * valueD}`（其中 `outH = num_heads*valueD`，0 时回退到 `query.size(2)`） |
| `BSND` | `{b, s1, n1, valueD}` |
| `BNSD` | `{b, n1, s1, valueD}` |
| `NBSD` | `{n1, b, s1, valueD}` |
| `TND` | `{t, n1, valueD}` |
| `NTD` | `{n1, t, valueD}` |
| `NSD` | `{n1, s1, valueD}` |

（`b/n1/s1/d1` 由 `get_query_b_n_s_d:92-130` 按 query_layout 解析；`t/n1/d1` 由 `get_query_t_n_d:132-156` 解析。）

`softmax_lse` 的 shape 由 `infer_lse_out_shape:271-292` 决定：TND/NTD 系（含 `TND_NTD`/`NTD_TND`）→ `{t, n1, 1}`；其余 → `{b, n1, s1, 1}`，dtype 恒为 Float。

### 2. valueD 多路径推导（PageAttention / GQA 维度）

`valueD`（value 的 head_dim）不能简单取 `value.size(-1)`。`get_value_d:158-196` 分两大场景：

**PageAttention 场景（`block_table` 有值）**——value 维度可变：

| value 维数 | valueD 公式 |
|-----------|-------------|
| 3D | `value.size(2) / kv_num_heads` |
| 4D | `value.size(3)` |
| **5D** | **`value.size(2) * value.size(4)`** |

> ⚠️ **5D 公式必须是 `size(2) * size(4)`，不是 `size(-1)`（= size(4)）**。5D value 形如 `[batch, blocks, heads, block_size, dim]`，valueD = blocks 数 × dim，漏掉 `size(2)` 会导致 PA 5D 场景 shape 严重错误。这是 attention 算子最常被记错的公式。

**非 PageAttention 场景**——按 query_layout 分支（且要求 `value.dim() == query.dim()`）：

| query_layout | valueD 公式 |
|--------------|-------------|
| `BSH` | `value.size(2) / kv_num_heads` |
| `BSND` / `BNSD` | `value.size(3)` |
| `TND` / `NTD` / `NSD` | `value.size(2)` |

**GQA（Grouped Query Attention）解耦**：当 `num_key_value_heads != num_heads` 时，valueD 的计算使用 `kv_num_heads`（= `num_key_value_heads`，0 时回退到 `num_heads`），与 query 的 head 数解耦。因此 BSH 的 `attention_out` shape 是 `{b, s1, num_heads * valueD}`，其中 `valueD` 来自 value 侧，`num_heads` 来自 query 侧——两者独立。

### 3. Output dtype 4 路推导（非"quant→INT8"单路）

`construct_fia_output_tensor:335-346` 是完整的 dtype 决策树，共 **4 条路径**，按优先级：

```
if (quant_scale2.has_value()):
    attention_out dtype = Char (INT8)        # 量化输出路径
elif (query.dtype() == at::kChar (int8)):    # query INT8 但无 quant_scale2
    if (query_rope.has_value()):
        dtype = query_rope.dtype()           # 走 RoPE 的 dtype
    else:
        dtype = Half                          # 回退 FP16
else:
    dtype = query.dtype()                     # 默认同 query
```

> ⚠️ 不要简化成"quant_scale2 存在 → 输出 INT8"单一路径。`query INT8 + 无 quant_scale2` 还会进一步看 `query_rope` 决定是 RoPE dtype 还是 Half。少判一条路径会导致 INT8 query 场景 dtype 错误。

### 4. softmax_lse 是条件输出但始终返回 2-tuple

`softmax_lse` 的存在性由 `softmax_lse_flag`（YAML 参数名）控制，但函数签名**始终返回 `(attention_out, softmax_lse)` 二元组**。

- `softmax_lse_flag = true` → 按布局算 `{b,n1,s1,1}` 或 `{t,n1,1}`，dtype Float（`infer_lse_out_shape`）
- `softmax_lse_flag = false` → 返回 **`{0}` 占位张量**（dtype Float），**不是 None、不是不返回**（`construct_fia_output_tensor:349-354`）

> 对接时 YAML 返回类型必须是 `(Tensor, Tensor)`，meta 实现也必须始终返回两个张量。把它当成"可选输出"会导致返回值数量不匹配。

### 5. aclnn 版本分派是合理模式（不是"改调用方绕过编译"）

两个入口都按 CANN 版本分派不同 aclnn 符号：

```cpp
// FusedInferAttentionScoreKernelNpuOpApi.cpp:414-424
if (op_plugin::utils::is_gte_cann_version_810rc1()) {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV3, ...);
} else {
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV2, ...);
}
```

> 这是**符合规范的版本分派**，不是"为了编译通过而改调用方"。区分原则：**根据环境（CANN/芯片版本）选择不同但都正确实现的目标** = 合理分派；**删除/跳过功能或引用不存在的符号以绕过缺失实现** = 应禁止的 workaround。skill guide 里"function not found → 实现它，绝不要改调用方"的绝对化表述，对这类多版本算子是过度约束。

### 6. INT4 packed 与 legacy 路径

两处特殊处理（`construct_fia_output_tensor:323-332`）：

- **INT4 packed**：value 为 `DT_INT32` 实际是 INT4 按 INT32 打包，`changeDScale = INT4_IN_INT32 = 8`。
- **BNSD 非 PA legacy 路径**：当 `input_layout == "BNSD"` 且 `!block_table.has_value()` 时，走一条 legacy 特例：
  ```cpp
  tmp_output = apply_tensor_without_format(
      {query.size(0), query.size(1), query.size(2), value.size(3) * changeDScale}, ...)
  ```
  即 outD = `value.size(3) * 8`（INT4 时）。这是历史兼容代码，新 layout 不要套用此公式。

### 7. op-plugin 命名规范（全程 snake_case，无缩写）

对照 YAML `func:` 行（`op_plugin/config/op_plugin_functions.yaml`）与 cpp 签名，参数命名**全程 snake_case**：

| 正确（snake_case） | ❌ 常见错误（camelCase / 缩写） |
|--------------------|-------------------------------|
| `pse_shift` | `pseShift` |
| `atten_mask` | `attenMask` |
| `pre_tokens` / `next_tokens` | `preTokens` / `nextTokens` |
| `num_key_value_heads` | `numKeyValueHeads` |
| `softmax_lse_flag` | `softmaxLseFlag` |
| `dequant_scale1` | **`deqScale1`（本算子用 `dequant_scale` 全称风格，不缩写为 `deq`）** |
| `quant_scale2` | `quantScale2` |
| `quant_offset2` | `quantOffset2` |

> aclnn C 接口侧（V5 头文件）用 camelCase（`quantScale2Optional`），但**一旦进入 op-plugin 的 YAML / cpp / meta，必须转 snake_case**。这是 PyTorch 生态约定，不要从 aclnn 头直接抄命名。

### 8. Meta 层镜像（C++ ↔ Python 双写，最大易漏点）

§1–§3 的 shape/dtype 推导不只写在 C++ 侧——**`_meta_registrations.py` 里有一份 Python 镜像，必须与 C++ 同步**，不是一方推导另一方。FakeTensor 路径走 meta 实现，meta 与 C++ 不一致会导致 trace/compile 期 shape/dtype 与实际运行不符。

op-plugin 侧的 meta 镜像（`op_plugin/python/meta/_meta_registrations.py`）：

| 函数 | 行号 | 对应的 C++ 侧 |
|------|------|--------------|
| `get_value_d` | `:905` | C++ `get_value_d:158-196`（含 PA 5D 的 `size(2)*size(4)`） |
| `infer_attention_out_shape` | `:936` | C++ `infer_attention_out_shape:198-269`（14 layout 分支） |
| `infer_lse_out_shape` | `:962` | C++ `infer_lse_out_shape:271-292` |
| `@impl npu_fused_infer_attention_score` | `:975` | v1 入口 forward，调用上述 helper |
| `@impl npu_fused_infer_attention_score_v2` | `:1030` | v2 入口 forward，**共用** `get_value_d` |

**双写纪律**：

- §1 的 14 layout、§2 的 valueD 多路径（含 5D 公式）、§4 的 softmax_lse 占位——这些 shape 逻辑在 C++ 和 meta 两侧**各写一份**，改一处必须同步另一处。
- **dtype 4 路决策（§3）也要在 meta 侧体现**：meta 构造输出张量时的 dtype 不能只取 `query.dtype()`，需对齐 `construct_fia_output_tensor` 的 4 路决策（quant_scale2→INT8、query INT8→看 query_rope/Half 等）。若 meta 仅按 `query.dtype()` 简单推导，FakeTensor 在 quant/INT8 分支下会得到与实际运行不符的 dtype。
- 落地 CHECKLIST 时，Dimension Branch Table 的 `Meta` 列（见 `checklist-template.md` 的 Dimension Branch Table）必须逐行填——每个 shape/dtype 分支既要有 C++ 实现，也要有 meta 镜像，两列都不能 MISSING。

### 9. CHECKLIST Dimension Branch Table 样例（节选）

把散在 §1/§2 的 shape 公式填入 CHECKLIST 的 Dimension Branch Table（见 `checklist-template.md`），可直接 seed。下表节选 3 行示范 `C++ | Meta` 双列填法（非穷举，按本算子实际分支补齐）：

| # | Dimension | Value | Output（shape / dtype） | C++ | Meta |
|---|-----------|-------|-------------------------|-----|------|
| 1 | attention_out_layout | `BSH` | `{b, s1, num_heads*valueD}` / §3 4 路 dtype | `infer_attention_out_shape:198` | `infer_attention_out_shape:936` |
| 2 | attention_out_layout | `TND` | `{t, n1, valueD}` / §3 4 路 dtype | `infer_attention_out_shape:198` | `infer_attention_out_shape:936` |
| 3 | value 维数（PA 场景） | 5D | valueD = `size(2)*size(4)` | `get_value_d:172` | `get_value_d:912` |

> 杠杆在于"每个分支 C++ 与 Meta 双填、不留 MISSING"，不在于穷举行数。其余 layout 取值、非 PA 的 valueD 分支、dtype 4 路逐行展开，按本算子 `.md`/aclnn 实际分支补齐。

### 10. 测试矩阵关键覆盖维度

14 layout × valueD 多路径 × dtype 4 路 × PA/非PA 的笛卡尔积不可能穷举，op-test 生成 UT 时优先覆盖以下正交维度的代表组合：

| 维度 | 必覆盖 | 可省/合并 |
|------|--------|-----------|
| input_layout | query↔out 相同取 1（如 `BSH`）+ query≠out 取 1（如 `BNSD_BSND`） | 14 值不必全测，按"相同/不同"两类各取代表 |
| valueD 路径 | 非 PA 的 `BSH`/`BSND` 各 1 + PA 的 5D（`size(2)*size(4)`）1 | 3D/4D value 可合并到基础用例 |
| output dtype | quant_scale2 路径（INT8）1 + query INT8 无 quant_scale2（看 query_rope/Half）1 + 默认同 query 1 | — |
| softmax_lse | flag=true 1 + flag=false（验证返回 `{0}` 占位非 None）1 | — |
| PA（block_table） | 有/无各 1 | — |

> ⚠️ PA 5D valueD（`size(2)*size(4)`）和 query INT8 dtype 分支是 attention 算子测试最常漏的用例，必须独立覆盖。

---

## Pitfalls

以下是把本算子领域知识**写错**的几种典型方式（部分曾在 skill guide 中出现，已在此案例纠正）：

1. **5D valueD 公式错记为 `size(-1)`** → 实为 `value.size(2) * value.size(4)`（`get_value_d:171-172`）。漏 `size(2)` 会让 PageAttention 5D 场景 shape 错。
2. **layout 被当作 BSH/BNSD/TND 三值** → 实为 14 值，且 8 个组合里 query_layout ≠ attention_out_layout。shape 推导必须双变量解析。
3. **参数名错记 camelCase + 臆造 `deqScale1`** → 本算子（fused_infer_attention_score）用 `dequant_scale1` 全称风格。注：库内 `npu_ffn`/`npu_prompt_flash_attention` 确有 `deq_scale` 缩写，但本算子属 `dequant_scale` 风格，勿混淆、勿臆造 `deqScale1`。
4. **output dtype 简化为"quant_scale2 → INT8"单路** → 实为 4 路决策树，`query INT8 无 quant_scale2` 时还要看 `query_rope`。
5. **把 layout × quant 当作正交维度** → 不成立：`quant_scale2` 存在会**覆盖** layout 维度决定的 dtype，二者在 dtype 轴上非独立。建模分支表时不要给"BSH + quant"这种组合留空白，要给出合成规则。
6. **把 softmax_lse 当可选输出** → 函数始终返回 2-tuple；`flag=false` 时返回 `{0}` 占位张量，不是 None。
7. **用 `.Version()` 检测本算子的多版本** → 本算子 `_def.cpp` 无 `.Version()` 标记，多版本体现在 aclnn API 层（V1~V5），需查 aclnn 头文件。
8. **把版本分派当 workaround 禁止** → 按 CANN 版本选 V2/V3/V4 是合理分派，不是"改调用方绕过编译"。区分见 §5。

---

## ACLGraph 手工 Update 接口（2 接口范式仓内实例）

本算子是 ACLGraph 手工 update 的**仓内 conformance 参考实现**（IFA v1；v2 同构，见 `FusedInferAttentionScoreV2KernelNpuOpApi.cpp`）。规范见 `references/guides/aclgraph-attention-guide.md`。

**交付物 = 2 个新增接口 + 1 个共享 helper**（`_infer_output` 为历史存在的 auto-dispatch 接口，不在手工 update 范围）：

| 接口 | 要点（以本算子为例） |
|---|---|
| `npu_fused_infer_attention_score.out` | `.default` 参数段（含 `*,`）逐字照抄 + 尾部 `Tensor? workspace=None, Tensor(a!) attention_out, Tensor(b!) softmax_lse`；workspace 双路径（`EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD` vs `EXEC_NPU_NO_FORMAT_CHECK_CMD`），两路径同 aclnn 版本分派（`is_gte_cann_version_810rc1` → V3/V2）；返回 `std::tuple<at::Tensor&, at::Tensor&>` |
| `_npu_fused_infer_attention_score_get_max_workspace` | `.default` 全参数 → 内部经 helper 构造临时输出 → `EXEC_GET_MAX_WORKSPACE_CMD` → 返回 1D workspace Tensor |
| `construct_fia_output_tensor`（helper） | `op_api::` 命名空间普通函数，functional 与 get-max 复用；shape/dtype 推导见本案例 §1-§4 |

**SymInt 形态（codegen-type 默认规则实例）**：`.default` 的 `SymInt[]? actual_seq_lengths` 在两个辅助接口中**照抄保留**，且 `.out` / `_get_max_workspace` 均同步登记 `symint:` 分组，C++ 函数名带 `_symint` 后缀；`_infer_output` 签名无 SymInt，仅入 custom 区。

**Meta**：`.out` / `_get_max_workspace` / `_infer_output` 均无 meta 注册（handler 直调，不经 Dynamo），C-verify 视为预期。

**历史包袱（新适配勿沿用）**：`_get_max_workspace` 返回 Tensor 用 query dtype 而非 `at::kByte`（`.out` 按 `numel*element_size` 回传字节数，自洽但多分配）。新接口按 cpp-guide §Attention ACLGraph Variants 统一 `at::kByte`。

**测试**：`test/test_custom_ops/test_npu_fused_infer_attention_score_with_workspace.py`（v2 同名 `_v2_`）——eager 三路径（functional / `.out(workspace=None)` / `.out(workspace=max_ws)`）全量输出对齐的样板。

> ⚠️ FA v3（`npu_fusion_attention_v3`）虽在 yaml 中也有同名接口条目，但**未完成真实接入验证**——其 `prefix` SymInt→`int[]` 降级、全位置参数 `*,` 形式、6 输出取 3 的 `_infer_output` 子集均不作依据。
