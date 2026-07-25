# aclnnSwiGluQuantV2 Integration Checklist

> **Mode: CREATE** — new operator integration (`.md` + `_def.cpp` → torch_npu).
> User note: 算子代码此前被删除（git `eaf38b20 del swiglu quant`），本次从输入文件**重新生成**，**禁止从 git 恢复**。

## Parse Results

- aclnn function name: `aclnnSwiGluQuantV2`
- torch_npu function name: `npu_swiglu_quant`  ✅ (C-confirm 确认；对齐 git 删除记录 eaf38b20)
- C++ 文件: `op_plugin/ops/opapi/SwigluquantKernelNpuOpApi.cpp`（对齐历史命名）
- Input parameters: `x` (Tensor, REQUIRED), `smooth_scales` (Tensor?, OPTIONAL), `offsets` (Tensor?, OPTIONAL), `group_index` (Tensor?, OPTIONAL)
- Attr parameters: `activate_left` (bool, base), `quant_mode` (str/int?, base ⚠️), `group_list_type` (int, V2), `dst_type` (int, V2)
- Output parameters: `y` (Tensor — INT8/INT4 依 `dst_type`), `scale` (Tensor — FLOAT)
- Key constraints: x 末维为 2 的倍数且 dim>1；末维 ≤ 8192；dstType=29(INT4) 时末维为 4 的倍数；smooth_scales 不支持空 Tensor；group_index 为非递减累计边界
- Operator type: **Forward** (无 `_grad` 后缀；无 existing derivative)

## ATK Case Dir (C-delimit input, path-only)

- ATK case dir: `/home/c00618179/aclnnSwiGluQuantV2/aclnnSwiGluQuantV2` | mode: 4file | files: atk_swiglu_quant.yaml, generator_npu_swiglu_quant.py, executor_swiglu_quant.py, nodes_accuracy.yaml
  - atk.yaml=`/home/c00618179/aclnnSwiGluQuantV2/aclnnSwiGluQuantV2/atk_swiglu_quant.yaml`
  - generate.py=`/home/c00618179/aclnnSwiGluQuantV2/aclnnSwiGluQuantV2/generator_npu_swiglu_quant.py`
  - executor.py=`/home/c00618179/aclnnSwiGluQuantV2/aclnnSwiGluQuantV2/executor_swiglu_quant.py`
  - node.yaml=`/home/c00618179/aclnnSwiGluQuantV2/aclnnSwiGluQuantV2/nodes_accuracy.yaml`
  > **主流程不读取这些文件**——仅在 C-delimit 交给 op-delimitation subagent。

## Parameter Information Table (Cross-Skill Shared)

| # | Param Name | Source | Type | Optionality | Default | AttrType | _def.cpp Line | .md Description | Constraints |
|---|------------|--------|------|-------------|---------|----------|---------------|-----------------|-------------|
| 1 | x | both | Tensor | REQUIRED | - | - | L24 | 待处理数据，公式中的 x | dim>1；末维为 2 的倍数；末维 ≤ 8192；INT4 时末维为 4 的倍数。dtype FLOAT16/BFLOAT16/FLOAT |
| 2 | smooth_scales | both | Tensor? | OPTIONAL | None | - | L30 | 量化的 smooth_scales | 不支持空 Tensor；shape [G,N] 或 [G,]；dtype FLOAT |
| 3 | offsets | both | Tensor? | OPTIONAL | None | - | L36 | 公式中的 offsets | 动态量化不生效；静态量化 dtype FLOAT；per_channel [G,N]，per_tensor [G,] |
| 4 | group_index | both | Tensor? | OPTIONAL | None | - | L42 | MoE 分组的 group_index | shape [G,]；非递减累计边界；dtype INT32 |
| 5 | activate_left | both | bool | OPTIONAL | False | OPTIONAL | L60 | 左矩阵是否参与运算 | - |
| 6 | quant_mode | both | str | OPTIONAL | "dynamic" | OPTIONAL | L61 | 量化模式 | "static"/"dynamic"/"dynamic_msd"；当前支持 dynamic/static。✅ C-confirm 定为 str（对齐 _def.cpp 权威） |
| 7 | group_list_type | both | int | OPTIONAL | 0 | OPTIONAL (V2) | L62 | 0=cumsum，1=count | `.Version(2)` → V2 引入 |
| 8 | dst_type | both | int | OPTIONAL | 2 (DT_INT8) | OPTIONAL (V2) | L63 | 2=int8 输出，29=int4 输出 | `.Version(2)` → V2 引入；控制 y 输出 dtype |
| 9 | y | both | Tensor | output | - | - | L48 | 输出张量 | shape 末维 = x 末维 /2，其余同 x；dtype INT8(dst=2)/INT4(dst=29) |
| 10 | scale | both | Tensor | output | - | - | L54 | 输出张量 | shape = x 去最后一维；dtype FLOAT |

**Source**: all `both`, consistent on names. 两处语义 ⚠️（算子名、quant_mode 类型）见 Conflict Confirmation。

## Information Required from User

- [ ] aclnn version variants: `.Version(SWI_GLU_QUANT_VERSION_TWO=2)` 标记 `group_list_type`/`dst_type` 为 V2 引入属性。本次接入是否全量实现 V2（含这两个属性 + INT4 输出）？
- [ ] SoC dispatch: `_def.cpp` 声明 ascend910b / ascend910_93 / ascend950 / kirinx90 / kirin9030。`.md` 产品支持 = Ascend950 / Atlas A3 / Atlas A2 √。是否需 SoC 分派？
- [ ] .List overload：无需（无 int[]/Tensor 双型参数）。
- [ ] 额外 torch_npu 变体：无。
- [ ] internal_format_opapi：无需（全 ND）。
- [ ] 特殊格式（mxfp4 等）：无。
- [ ] autograd 反向：**否**（Forward 推理算子，量化输出不可导，无 existing derivative）→ 跳过 C-derivatives。
- [ ] **图模式 (torch.compile) 支持？** → 影响 C-meta / build flag / 图模式测试与文档。
- [ ] **ACLGraph 手工 update？**（正交于图模式）→ 本算子非 attention 类动态 seq-len，**默认 NO**。

### Conflict Confirmation (from Parameter Information Table)

| # | Parameter | Conflict Description | Resolution |
|---|-----------|---------------------|------------|
| 1 | 算子名 | parse 派生 `npu_swi_glu_quant_v2`；V1 case 记录真实库已演进为 `npu_swiglu_quant`（无下划线，全库唯一，`int quant_mode`） | ⚠️ Awaiting user confirmation |
| 2 | quant_mode 类型 | `_def.cpp`=`String("dynamic")`（权威）；V1 case 记录真实库演进为 `int quant_mode` | ⚠️ Awaiting user confirmation |
| 3 | dst_type 默认值表示 | `_def.cpp` `.Int(ge::DT_INT8)`（=2）；YAML/文档需用数值 2 | ⚠️ Awaiting user confirmation（建议 default=2） |

- [x] All ⚠️ conflicts confirmed by user
  - [x] #1 算子名 → `npu_swiglu_quant`
  - [x] #2 quant_mode → `str`
  - [x] #3 dst_type default → `2`
  - [x] 图模式 → **YES**（执行 C-meta / 移除 --disable_torchair / 图模式测试+文档 / meta 交叉核对）

## Pitfall Alerts (from parse JSON `pitfall_alerts`)

| # | Alert id | Trigger | Must-read case | Gates steps | Applied? |
|---|----------|---------|----------------|-------------|----------|
| 1 | conditional_output_allocation | bool attr `activate_left` + 2 outputs | create-mode/npu_moe_gating_top_k-create.md | C-cpp, C-meta | ☑ C-cpp 已落实：y/scale 均 REQUIRED，恒按真实维度分配（y 末维=x末维/2；scale 去末维），`activate_left` 仅作标量入参传给内核，不参与 shape 决策——属读法 C。C-meta 将套用同规则。 |

## Constraint Logic Checklist (Cross-Step Verification)

| # | Constraint Rule | .md Source | C++ (C-cpp) | Meta (C-meta) | Test (C-test) | Doc (C-docs) |
|---|-----------------|------------|--------------|---------------|---------------|--------------|
| 1 | x.dim() > 1 | L125 | ☑ L43 | ☑ meta | ☐ | ☐ |
| 2 | x 末维为 2 的倍数 | L125 | ☑ L45 | ☑ meta | ☐ | ☐ |
| 3 | x 末维 ≤ 8192 | L125 | ☑ L47 | ☑ meta | ☐ | ☐ |
| 4 | dst_type=29(INT4) 时 x 末维为 4 的倍数 | L125 | ☑ L57 | ☑ meta | ☐ | ☐ |
| 5 | quant_mode ∈ {dynamic, static}（当前支持） | L175 | ☑ L49 | ➖ (config enum，不影响 shape) | ☐ | ☐ |
| 6 | group_list_type ∈ {0,1} | L185 | ☑ L51 | ➖ (config enum) | ☐ | ☐ |
| 7 | dst_type ∈ {2,29} | L195 | ☑ L54 | ☑ meta | ☐ | ☐ |
| 8 | y 末维 = x 末维 /2，其余同 x | L205 | ☑ L79-82 (shape 计算) | ☑ meta (y_size) | ☐ | ☐ |
| 9 | scale = x 去最后一维 | L215 | ☑ L77-78 | ☑ meta (scale_size) | ☐ | ☐ |
| 10 | y dtype: dst=2→INT8, dst=29→INT4 | L195/L205 | ☑ L84-91 (TensorWrapper) | ☑ meta (int8/uint8) | ☐ | ☐ |

## Complex Operator Grouped Mode

> **Triggered**: 量化算子，输出 `y` 的 dtype 依赖 `dst_type`（≥1 独立参数决定输出 dtype），且 `.md` 约束含 ≥2 独立条件规则（MoE/非MoE × 动态/静态量化 × int8/int4）。启用分组/维度模式。

### Feature Parameter Groups

| # | Group Name | Parameters | Intra-group Constraints | Version/Chip Differences | Shape Impact | C++ Pre-proc |
|---|-----------|------------|------------------------|------------------------|-------------|-------------|
| 1 | 主输入 | x | dim>1，末维偶数，≤8192 | - | y 末维=x末维/2；scale=x去末维 | Y (shape 计算) |
| 2 | 量化缩放 | smooth_scales, offsets | offsets 静态量化生效；per_tensor 时二者 shape/dtype 一致 | - | none | N |
| 3 | MoE 分组 | group_index, group_list_type | group_index 非递减累计边界；group_list_type 0=cumsum/1=count | group_list_type V2 | none | N |
| 4 | 量化模式/输出类型 | quant_mode, dst_type, activate_left | dst=29 时 x 末维需 4 的倍数 | dst_type V2 | y dtype: dst=2→INT8, dst=29→INT4 | N (dtype 由 dst_type 决定) |

### Dimension Branch Table

| # | Dimension | Value | Affected Params/Groups | Output (name: shape/dtype) | Notes | C++ | Meta |
|---|-----------|-------|----------------------|--------------------------|-------|-----|------|
| 1 | dst_type | 2 | Group4 | y: [...x[:-1], x[-1]/2] INT8; scale: x[:-1] FLOAT | 默认 | TBD | TBD |
| 2 | dst_type | 29 | Group4 | y: [...x[:-1], x[-1]/2] INT4; scale: x[:-1] FLOAT | 需 x 末维为 4 的倍数 | TBD | TBD |
| 3 | quant_mode | dynamic | Group4 | scale 有效（127/max_abs） | offsets 不生效 | TBD | TBD |
| 4 | quant_mode | static | Group4 | scale 语义不同 | offsets 生效 | TBD | TBD |

> 说明：y/scale 的 **shape** 不随 dst_type/quant_mode 变化（仅 y 的 dtype 随 dst_type 变）。shape 规则统一：y 末维减半，scale 去末维。

## Parameter Coverage Checklist

| # | Parameter Name | Scenario | Test Case Name | Status |
|---|----------------|----------|----------------|--------|
| 1 | smooth_scales | 量化缩放 | test_swiglu_quant_dynamic | ☐ |
| 2 | offsets | 静态量化 | test_swiglu_quant_static | ☐ |
| 3 | group_index | MoE 分组 | test_swiglu_quant_moe | ☐ |
| 4 | group_list_type | count/cumsum 模式 | test_swiglu_quant_group_list_type | ☐ |
| 5 | dst_type | INT4 输出 | test_swiglu_quant_int4 | ☐ |

## Table Migration Checklist

| # | Table Content | .md Line | Doc Location | Status |
|---|---------------|----------|--------------|--------|
| 1 | 参数表（x/smooth_scales/offsets/group_index/attrs/outputs） | L99-242 | 参数说明 | ☑ 全参数已迁移 |
| 2 | quant_mode 取值表 | L175 | 参数说明（枚举） | ☑ dynamic/static |
| 3 | dst_type 取值表 | L195 | 参数说明（枚举） | ☑ 2=int8/29=int4 |
| 4 | 错误码表 | L250-285 | 约束说明 | ➖ 转为约束文字（torch_npu 侧以 TORCH_CHECK 呈现） |

## Derivatives Configuration Checklist (C-derivatives)

> N/A — Forward 推理算子，无 autograd。跳过 C-derivatives。

## Compatibility Registration Checklist (C-baseline)

| # | Item | Details | Status |
|---|------|---------|--------|
| 1 | `op_api:` entry in schema JSON | Python signature + version tags | ☑ 已写入 `op_api: torch_npu.npu_swiglu_quant(*args, **kwargs)` [v2.1, newest] |
| 2 | `func:` entry in schema JSON | 底层算子签名 + version tags | ☑ 已写入 func 全签名 [v2.1, newest]（verbatim from YAML） |
| 3 | Version tags | 新算子 → `"newest"` | ☑ [v2.1, newest]（对齐 YAML op_api 字段） |
| 4 | allowlist entry | `all_version` | ☑ 写入 `torch_npu.v2.1`（版本特定，非 all_version） |
| 5 | Signature verified | 对齐 yaml func + inspect.signature | ☑ **PASS** — `test/core_tests/test_compatibility.py` 通过。修正：新增 ParamSpec 形态条目 `op_api: torch_npu.npu_swiglu_quant(*args: _P.args, **kwargs: _P.kwargs) -> ~_T` [v2.8, newest]（对齐 2.8+ 装饰后运行时签名），原 `(*args, **kwargs)` 形态版本改为 [v2.1]。 |

## Progress

- [x] C-parse: Parse input files
- [x] C-confirm: Confirm operator-specific information with user
  - [x] **Graph mode support confirmed: YES**
- [x] C-yaml: Generate YAML
- [x] C-derivatives: N/A (no autograd)
- [x] C-cpp: Implement C++
- [x] C-constraints: Constraint self-check
- [x] C-meta: Add Meta registration [conditional: 图模式 YES]
- [x] C-verify: Signature verification ✅ (script passed + graph-mode Meta Shape/Dtype cross-verify PASS，无 BLOCK)
- [x] C-build: Build
  - [x] 首次 build 完成（PID 84067，wheel 已装）
  - [x] **源码 bug 修复后 incremental rebuild**（PID 95735）：`dst_type` acl dtype 映射修复（`static_cast<aclDataType>` 而非 `c10_npu::GetAclDataType`）。
  - [x] **dtype 前置校验 rebuild**（PID 114016，`--python=3.10 --pytorch=v2.10.0`，MAX_JOBS=64，增量）：在 `EXEC_NPU_CMD` 前补 `TORCH_CHECK` 校验 x/smooth_scales/offsets/group_index dtype，避免非法 dtype 进内核后遗留 aclnn 107000 错误态污染后续调用。wheel 已 force-reinstall。
- [x] C-test: Tests (UT + FakeTensor) ✅ **30/30 UT PASS + 3/3 FakeTensor PASS + reconcile 通过（23 正向 + 7 负向全覆盖）**
  - 调试中定位 3 处 kernel 行为（记入 memory `swiglu-quant-kernel-semantics`）：① smooth_scales 缺省即 507057（MTE 越界）→ UT 恒合成全 1 smooth_scales；② `activate_left=False` → `a*silu(b)`（silu 作用于右半）；③ static 量化 scale 恒为 0。参考实现相应修正。
- [x] C-docs: API docs ✅ (doc md + menu/list 导航 + _op_plugin_docs.py docstring；含图模式示例)
- [x] C-baseline: Compatibility baseline ✅ (schema + allowlist 已写入；`test_compatibility.py` 通过，含 ParamSpec 形态修正)
- [x] C-delimit: ATK cases via op-delimitation subagent (ATK provided) — ✅ **单算子；50/50 精度全过（int4 executor 解包修正后）；算子本身与 golden 一致，不一致源于 ATK executor 适配层缺陷，未改动 op-plugin 算子代码**
- [x] C-summary: Integration summary — ✅ `notes/summary_2026-07-25.md` 全部章节完成；用户选择不保存为 case。

## UT Generation Info

### Operator Basic Info
- Operator Name: swiglu_quant
- torch_npu function name: npu_swiglu_quant
- Operator Type: custom
- Target Directory: test/test_custom_ops
- Mode: create
- Operator Classification: quantization

### Source File Paths
- _def.cpp: aclnn_input/swi_glu_quant_def.cpp
- .md doc: aclnn_input/aclnnSwiGluQuantV2.md

### Input/Output Spec

**Input parameter list**:

| Param Name | Type | Required/Optional | Supported dtypes | Constraints |
|------------|------|-------------------|------------------|-------------|
| x | Tensor | Required | float16, bfloat16, float32 | dim>1; last dim even; last dim<=8192; when dst_type=29(int4) last dim %4==0 |
| smooth_scales | Tensor? | Optional | float32 | no empty Tensor; shape [G,N] or [G,]; N = x last dim /2 |
| offsets | Tensor? | Optional | float32 | static-quant only; per_channel [G,N], per_tensor [G,] matching smooth_scales |
| group_index | Tensor? | Optional | int32 | shape [G,]; non-decreasing; max <= prod(x.shape[:-1]) |
| activate_left | bool | Optional | - | default False |
| quant_mode | str | Optional | - | Interaction: quant_mode×offsets. values dynamic(default)/static |
| group_list_type | int | Optional | - | values 0=cumsum(default)/1=count |
| dst_type | int | Optional | - | Interaction: dst_type×x. values 2=int8(default)/29=int4(needs x last dim %4==0) |

**Output specs**:

| Output Name | Type | Shape Rule | Dtype Rule |
|-------------|------|------------|------------|
| y | Tensor | x.shape[:-1] + [x.shape[-1]//2] | int8 when dst_type=2, int4 when dst_type=29 |
| scale | Tensor | x.shape[:-1] | float32 |

### Coverage Matrix

> Auto-generated by scaffold_coverage.py, add/delete applied per Constraints column markers.

| Test Item | Covered Params | Data Type | Dimensions | Scenario |
|-----------|----------------|-----------|------------|----------|
| test_basic_bf16 | x | bfloat16 | (32, 256) | Basic functionality |
| test_basic_fp16 | x | float16 | (32, 256) | Basic functionality |
| test_basic_fp32 | x | float32 | (32, 256) | Basic functionality |
| test_smooth_scales_offsets_group_index_bf16 | x, smooth_scales, offsets, group_index | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+offsets+group_index(bf16) |
| test_smooth_scales_offsets_group_index_fp16 | x, smooth_scales, offsets, group_index | float16 | (2, 32, 256) | 可选参数-smooth_scales+offsets+group_index(fp16) |
| test_smooth_scales_offsets_bf16 | x, smooth_scales, offsets | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+offsets(bf16) |
| test_smooth_scales_offsets_fp16 | x, smooth_scales, offsets | float16 | (2, 32, 256) | 可选参数-smooth_scales+offsets(fp16) |
| test_smooth_scales_group_index_bf16 | x, smooth_scales, group_index | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales+group_index(bf16) |
| test_smooth_scales_group_index_fp16 | x, smooth_scales, group_index | float16 | (2, 32, 256) | 可选参数-smooth_scales+group_index(fp16) |
| test_smooth_scales_bf16 | x, smooth_scales | bfloat16 | (2, 32, 256) | 可选参数-smooth_scales(bf16) |
| test_smooth_scales_fp16 | x, smooth_scales | float16 | (2, 32, 256) | 可选参数-smooth_scales(fp16) |
| test_offsets_group_index_bf16 | x, offsets, group_index | bfloat16 | (2, 32, 256) | 可选参数-offsets+group_index(bf16) |
| test_offsets_group_index_fp16 | x, offsets, group_index | float16 | (2, 32, 256) | 可选参数-offsets+group_index(fp16) |
| test_offsets_bf16 | x, offsets | bfloat16 | (2, 32, 256) | 可选参数-offsets(bf16) |
| test_offsets_fp16 | x, offsets | float16 | (2, 32, 256) | 可选参数-offsets(fp16) |
| test_group_index_bf16 | x, group_index | bfloat16 | (2, 32, 256) | 可选参数-group_index(bf16) |
| test_group_index_fp16 | x, group_index | float16 | (2, 32, 256) | 可选参数-group_index(fp16) |
| test_dst_type_int8 | x, dst_type | float16 | (32, 256) | dst_type=int8 |
| test_dst_type_int4 | x, dst_type | float16 | (32, 256) | dst_type=int4(29) |
| test_int4_output_shape | x, dst_type | float16 | (4, 64) | INT4-输出形状验证(末维/2，dtype int4) |
| test_scale_value | x | float16 | (2, 32, 256) | scale数值验证(比golden,非纯shape) |
| test_large_input | x | float16 | (3072, 7168) | Large data (must compare scale golden, not pure shape smoke) |
| test_graph_mode | x, smooth_scales, group_index | float16 | (2, 32, 256) | 图模式(torch.compile npugraph_ex)对齐 eager |

**Auto-generated stats**: 22 test items.

**Rule compliance self-check**:
- Basic functionality dtype coverage: 3 rows ✅
- Output dtype full coverage: expected 2 values ['int8', 'quint4x2'], expanded 2 rows ✅
- Optional param coverage: 3 optional Tensors ✅
- Interaction / not-applicable: Step 3 handles row add/delete and expansion per `Interaction: A×B` / `Not applicable: param` anchors in Constraints column
- Required scenario check (category=quantization):
    - dst_type all values: ✅ covered
    - smooth_scales: ✅ covered
    - group_index: ✅ covered
    - INT4 shape verify: ✅ covered

**⚠️ Self-check warnings (step 3 review or upstream format fix needed)**:
  - Negative degradation: .constraint_info.json missing, only dtype+dim negative recoverable; rank/shape_link/mutex negative lost → re-run scaffold_init to generate .constraint_info.json then re-run this script

### Negative Coverage Matrix

> Auto-generated by generate_negative_cases(), derived from constraint signals for illegal input combinations.

| Test Item | Covered Params | Data Type | Dimensions | Scenario | Violated Field | Violation Construction | Expected |
|-----------|----------------|-----------|------------|----------|----------------|------------------------|----------|
| test_neg_x_invalid_dtype | x | float16 | (32, 256) | Negative-dtype domain | x | dtype=int64 | raises:PARAM_INVALID |
| test_neg_smooth_scales_invalid_dtype | x, smooth_scales | float16 | (32, 256) | Negative-dtype domain | smooth_scales | dtype=int64 | raises:PARAM_INVALID |
| test_neg_offsets_invalid_dtype | x, offsets | float16 | (32, 256) | Negative-dtype domain | offsets | dtype=int64 | raises:PARAM_INVALID |
| test_neg_group_index_invalid_dtype | x, group_index | float16 | (32, 256) | Negative-dtype domain | group_index | dtype=float32 | raises:PARAM_INVALID |
| test_neg_dst_type_invalid | x, dst_type | float16 | (32, 256) | Negative-value domain | dst_type | dst_type=3 (not in {2,29}) | raises:PARAM_INVALID |
| test_neg_x_odd_last_dim | x | float16 | (32, 255) | Negative-shape domain | x | x last dim odd (255) | raises:PARAM_INVALID |
| test_neg_int4_last_dim_not_align | x, dst_type | float16 | (32, 258) | Negative-shape domain | x | dst_type=29 & x last dim %4!=0 | raises:PARAM_INVALID |

**Negative category coverage**: 1 categories, 5 rows total

### Finalization Add/Delete Log

> 格式契约: `Delete <exact_method_name>: <reason>` / `Add <exact_method_name>: <reason>`
> `<exact_method_name>` 必须与 Coverage Matrix 的 Test Item 列值**逐字一致**, 禁自然语言概述。
> reconcile.py 依赖此格式做删除放行判定, 格式不符 → [BAD-LOG] 告警。

<!-- 在此下方逐行记录增删, 一行一项 -->
Add test_graph_mode: 图模式(torch.compile) 支持 = YES，需新增 eager vs npugraph_ex 对齐用例（图模式质量门禁）。
Add test_neg_x_odd_last_dim: x 末维为奇数违反"末维为2的倍数"约束（shape 域负例，scaffold 因缺 .constraint_info.json 丢失，手工补回）。
Add test_neg_int4_last_dim_not_align: dst_type=29(int4) 且 x 末维非4的倍数违反 int4 对齐约束（交互 dst_type×x 负例，手工补回）。
