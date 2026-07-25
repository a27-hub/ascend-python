# op-adaptation

将 CANN (Ascend) 算子集成到 torch_npu 后端。支持两种模式：**(1) Create** — 从零接入新算子；**(2) Modify** — 修改已有算子（新增变体、接口变更、反向补充、Bug 修复）。其中 Modify 含子模式 **(2.1) M-adapt** — 算子新增 Atlas A5（Ascend950）平台 ACLNN 路径支持，接口签名不变，运行时按 SoC 路由。

作者：Core&API组

---

**目录**

- [适用范围与约束](#适用范围与约束)
- [准备](#准备)
- [添加技能](#添加技能)
- [Create 模式](#create-模式)
- [Modify 模式](#modify-模式)
- [案例](#案例)
- [扩展指南](#扩展指南)
- [相关文档](#相关文档)

---

## 适用范围与约束

- **算子类型**：仅支持自定义算子（custom ops），且 torch_npu 接口与 CANN
  算子为**一对一**映射的场景。**不支持**一个 torch_npu 接口组合调用多个
  CANN 算子实现的场景（组合算子请走 op-delimitation 的组合类分析路径）。

---

## 准备

### 环境

> - op-adaptation 默认运行环境和编译环境为同一个环境。如需在不同环境中编译，编译步骤执行前会主动询问。
> - Python、torch、torch_npu 和 CANN 有版本对应关系，参考[版本配套说明](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E7%9B%B8%E5%85%B3%E4%BA%A7%E5%93%81%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)。

#### [op-plugin](https://gitcode.com/Ascend/op-plugin)代码拉取

```bash
git clone https://gitcode.com/<username>/op-plugin.git -b master
cd op-plugin
```
- 先 fork op-plugin 仓库，再拉取代码
- `<username>` 替换为 GitCode 用户名，`master` 按实际替换分支
- 分支依赖见[版本说明](https://gitcode.com/Ascend/op-plugin#%E7%89%88%E6%9C%AC%E8%AF%B4%E6%98%8E)

#### Python 虚拟环境（conda）

> [miniconda3]安装指引(https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

```bash
# 创建虚拟环境
conda create -n adapt_op python=3.12
# 激活虚拟环境
conda activate adapt_op
# 安装依赖
pip install torch==2.10.0
pip install -r path/to/requirements.txt
```

默认配置：`python=3.10`，`torch_npu branch=v2.10.0`。若有变动需同步修改编译指令（如 `--python=3.11 --pytorch=v2.9.0`）。

#### CANN 安装（conda）

```bash
# 默认安装路径为所在 Conda 虚拟环境的 Ascend 目录下
conda install ascend::cann-toolkit==9.0.0
# 安装ops算子包
conda install ascend::cann-910b-ops==9.0.0
# 配置环境变量，$CONDA_PREFIX 为 Conda 虚拟环境的根目录
source $CONDA_PREFIX/Ascend/cann/set_env.sh
# 若返回芯片型号，则验证 CANN 安装成功
python3 -c "import acl;print(acl.get_soc_name())"
```

参考：[CANN 官方安装教程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=openEuler&InstallType=netconda)。

### 工具

本技能采用通用 Skills 格式（`SKILL.md` + `references/`），主流 AI 编程智能体可直接使用。

### 输入文件

两个输入文件均来自 CANN 算子仓库。CANN 有四个算子库：

| 算子库 | 地址 | 类别 |
|--------|------|------|
| ops-nn | [gitcode.com/cann/ops-nn](https://gitcode.com/cann/ops-nn) | 神经网络 |
| ops-math | [gitcode.com/cann/ops-math](https://gitcode.com/cann/ops-math) | 数学运算 |
| ops-transformer | [gitcode.com/cann/ops-transformer](https://gitcode.com/cann/ops-transformer) | Transformer |
| ops-cv | [gitcode.com/cann/ops-cv](https://gitcode.com/cann/ops-cv) | 计算机视觉 |

#### 算子文档（.md）

描述算子函数签名、参数说明和数据类型约束。获取路径：`${op_class}/${op_name}/docs/aclnn${OpName}.md`

#### 算子原型文件（_def.cpp）

定义算子的输入输出参数和属性。获取路径：`${op_class}/${op_name}/op_host/${op_name}_def.cpp`

#### 放置建议

```
aclnn_input/
└── DynamicQuant/
    ├── aclnnDynamicQuant.md
    └── dynamic_quant_def.cpp
```

#### ATK 用例目录（Create 模式 · C-delimit 阶段，可选）

流程后段（测试完成后）的 ATK 定界步骤可使用 CANN ATK 用例目录，两种形态均可：

**4 件套**（需流程生成用例）：

| 文件 | 作用 |
|------|------|
| `atk.yaml` | 用例定义 |
| `generate.py` | 数据生成 |
| `executor.py` | 执行器 |
| `node.yaml` | 节点配置 |

**3 件套**（用户已生成用例）：`case.json`（用例文件）+ `executor.py` + `node.yaml`，
提供后跳过用例生成（`atk case`）步骤，`atk task -c` 直接用传入的 case.json。

- 获取途径：来自 CANN 算子对应的 ATK 测试工程；仓内示例可参考 `moeTopK/`。
- **可选输入**：提供则执行 ATK 用例验证与定界；未提供则自动跳过该步骤，
  并在集成摘要中注明"无 ATK 输入，未执行 ATK 用例验证"。
- 一开始就提供路径也没问题——主流程只登记路径、不读取内容，C-delimit 时
  原样转交 op-delimitation 子代理。
- **新手建议**：首次使用本技能，推荐**先不带 ATK 跑通主流程**（编译→测试→
  文档→基线→总结），验收 13 个产物文件后再补 ATK 定界。ATK 为可选增强，
  缺它不影响算子接入本身的正确性。

---

## 添加技能

以 **Trae CLI** 添加技能为例：

> Trae CLI 安装见[此](https://docs.trae.cn/cli/get-started-with-trae-cli)  
> Trae CLI 技能介绍见[此](https://docs.trae.cn/cli/skills)

将 op-adaptation 技能及其子技能放到项目根目录的 `.trae/skills/` 下：

```
.trae/
└── skills/
    ├── op-adaptation/
    │   ├── SKILL.md
    │   └── references/
    ├── op-test/
    │   ├── SKILL.md
    │   └── references/
    ├── op-apidoc/
    │   ├── SKILL.md
    │   └── references/
    └── op-delimitation/
        └── skills/ tools/ knowledge/ 等子目录（技能套件，整体拷贝）
```

> Claude Code 环境部署时对应 `.claude/skills/`，且 C-delimit 经子代理入口 `.claude/agents/op-delimitation.md` 调用，该文件需一并就位。

## Create 模式

从零接入新算子。提供 `.md` + `_def.cpp` 两个文件，自动完成全流程。

### 快速开始

```bash
conda activate adapt_op
source $CONDA_PREFIX/Ascend/cann/set_env.sh

# 启动 Trae CLI
traecli

# 调用技能
/op-adaptation 将 aclnnDynamicQuant 算子接入 torch_npu 后端，输入文件在 ./aclnn_input/DynamicQuant/ 下

# （可选）如有 ATK 用例目录，可一并提供路径：
/op-adaptation 将 aclnnDynamicQuant 算子接入 torch_npu 后端，输入文件在 ./aclnn_input/DynamicQuant/ 下，ATK 用例目录在 ./atk_cases/DynamicQuant/ 下
```

#### 一次完整对话回放（新手参考）

以 DynamicQuant 为例，打完调用命令后的典型交互时序：

1. **你**：`/op-adaptation 将 aclnnDynamicQuant 接入 torch_npu，输入在 ./aclnn_input/DynamicQuant/`
2. **Claude**：解析 .md + _def.cpp，生成参数信息表，回报 ⚠️ 冲突项（如 `dst_type` 仅 _def.cpp 有）→ 问你"保留还是排除？"
3. **你**：逐条回答冲突 + 标准提问（版本分发 / SoC / .List / 额外变体 / autograd / **图模式**，见下方"交互环节"表）。**答不上就填"不需要 / 否"**——保守选项不会破坏流程。
4. **Claude**：无需再交互，自动跑完 YAML → C++ → Meta → 校验 → 编译 → 测试 → 文档 → 基线 → 总结。
5. **产出**：13 个文件（见"最终产物"表）+ `CHECKLIST.md` + `notes/summary_*.md`。

> 交互只集中在第 2-3 步（"确认信息"步骤）；之后全自动。若中途某步失败，见"案例"章节后的"卡住了怎么办"。

### 工作流程

```
配置阶段: 解析输入 → 确认信息 → 生成YAML → [Derivatives]
编码阶段: 生成C++ → 约束检查 → [Meta注册] → 签名校验
验证阶段: 编译 → 测试 → 文档 → 基线 → [ATK 定界] → 总结
```

`[方括号]` 为条件步骤。

| 步骤 | 做什么 | 用户做什么 |
|:----:|--------|-----------|
| 解析输入 | 解析 .md + _def.cpp，生成 CHECKLIST 参数信息表 | 无 |
| 确认信息 | 先确认参数冲突，再回答版本/SoC/autograd 等 | 回答冲突和提问 |
| 生成 YAML | 在 `op_plugin_functions.yaml` 注册算子签名 | 确认签名 |
| Derivatives | [条件] 在 `derivatives.yaml` 配置前反向绑定 | 确认绑定关系 |
| 生成 C++ | 创建 `*KernelNpuOpApi.cpp` | 无 |
| 约束检查 | 验证 CHECKLIST 约束均已实现为 TORCH_CHECK | 无 |
| Meta 注册 | [条件: 图模式] 在 `_meta_registrations.py` 注册 FakeTensor 实现 | 无 |
| 签名校验 | 交叉校验 YAML↔C++↔Meta↔Derivatives | 无 |
| 编译 | `bash ci/build.sh` 构建并安装 whl | 确保构建环境就绪 |
| 测试 | 调用 `op-test` 生成并执行 UT + FakeTensor | 无 |
| 文档 | 调用 `op-apidoc` 生成 API 文档 | 无 |
| 基线 | 注册兼容性基线（schema JSON + allowlist） | 无 |
| ATK 定界 | [条件] 调用 `op-delimitation` 子技能执行 ATK 用例并定界；无 ATK 输入则跳过并记入摘要 | 提供 ATK 用例目录路径（可选） |
| 总结 | 生成集成摘要 | 无 |

### 交互环节

"确认信息"步骤中会询问以下内容，根据实际情况回答：

| 问题 | 示例回答 |
|------|---------|
| 参数冲突（_def.cpp 有但 .md 没有） | "保留，纳入实现" / "排除" |
| CANN 版本分发（V2/V3/V5）？ | "不需要" |
| SoC 特定分发？ | "不需要" |
| .List 重载？ | "不需要" |
| 额外函数变体（_asymmetric, _v0 等）？ | "需要 npu_xxx_asymmetric" |
| autograd 反向传播？ | "是" / "不需要" |
| 支持图模式（torch.compile / npugraph_ex）？ | "是" / "否" |

> 选"是"将联动 Meta 注册、编译去掉 `--disable_torchair`、图模式测试用例与文档示例、Meta Shape/Dtype 交叉验证；选"否"以上全部跳过（依据 `confirm-guide.md`）。

---

## Modify 模式

修改已有算子。提供算子名 + 修改描述，自动发现影响范围、仅修改受影响的文件。

### 场景

| 场景 | 触发条件 | 涉及文件（典型） |
|------|----------|-----------------|
| 新增 variant | 同一算子新增 `_v2`/`_v3` 变体，旧版保留 | yaml + meta + output_size + test + json + derivatives(可能) |
| 接口变更 | 已有算子参数增删/类型变化 | yaml + meta + output_size + test + json + derivatives(可能) |
| 反向补充 | 为已有前向算子新增/修改 backward | derivatives + yaml(如反向未注册) + test + json |
| Bug 修复 | 实现逻辑修正，接口不变 | cpp + test |
| A5 平台适配（M-adapt 子模式） | 已有算子（已 A2/A3 适配）新增 Ascend950 平台 ACLNN 路径，接口不变 | yaml + cpp + meta + test + doc |

### 工作流程

```
M-discover → M-scope → M-gate → M-execute
(grep+状态)  (推导范围+确认) (git检查)  (映射→执行Create子集)
```

**M-adapt 子模式**：当 M-discover 自动检测到算子命中以下 5 状态之一时,路由到对应处置:

| Step 0 状态 | 信号 | 处置 |
|------------|------|------|
| **A 已手工适配** | YAML 有 `op_api:` + `opapi/<Op>.cpp` 存在 | 退出(勿动) |
| **B 已结构化适配** | YAML 含 `gen_opapi:` 子块 | 退出(已在 M-adapt 范围外) |
| **C 真 M-adapt 候选** | YAML 仅 `acl_op:` + 4 条件 AND 门通过 | 进入 **Scenario 1/2/3**(910+A5 双平台) |
| **D 无原实现** | `aclops/<Op>.cpp` + `op_api:` 都缺失 | 路由到 Create 模式 |
| **E A5-only 候选** | YAML 仅 `op_api:` + `aclops/<Op>.cpp` 缺失 | 进入 **Scenario 4**(A5 独有,op-plugin 中 242 个候选) |

详见 `references/guides/m-adapt-guide.md` §2(状态分类)+ §6(场景路由表)。

### 快速开始

```bash
conda activate adapt_op
source $CONDA_PREFIX/Ascend/cann/set_env.sh

# 启动 Trae CLI
traecli

# 调用技能
/op-adaptation 修改 npu_moe_token_permute_grad，参数 tokens Tensor 换为 tokens_size_0 int + tokens_dtype ScalarType
```

---

## 案例

### Create: DynamicQuant

**输入文件**：`aclnnDynamicQuant.md` + `dynamic_quant_def.cpp`

**解析输入**，提取参数信息表：

| 参数 | 类型 | 可选性 | 默认值 | 来源 |
|------|------|--------|--------|------|
| x | Tensor | REQUIRED | — | _def.cpp |
| smooth_scales | Tensor | OPTIONAL | None | _def.cpp |
| group_index | Tensor | OPTIONAL | None | _def.cpp |
| dst_type | int | OPTIONAL | DT_INT8 | _def.cpp Attr |

用户确认需要额外的 `_asymmetric` 变体，其他项不需要。

**生成 YAML**（`op_plugin_functions.yaml`）：

```yaml
- func: npu_dynamic_quant(Tensor input, *, Tensor? smooth_scales=None,
      Tensor? group_index=None, int? dst_type=None, str quant_mode="pertoken")
      -> (Tensor, Tensor)
  acl_op: all_version
  op_api: all_version
```

**生成 C++**（`DynamicQuantKernelNpuOpApi.cpp`）：

```cpp
std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type,
    c10::string_view quant_mode)
{
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, ...);
    TensorWrapper y_wrapper = {output, y_acltype};
    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, y_wrapper, scale);
    return std::make_tuple(output, scale);
}
```

**注册 Meta**（`_meta_registrations.py`）：

```python
@torch.ops.op_api.npu_dynamic_quant.default
def meta_npu_dynamic_quant(input, *, smooth_scales=None, group_index=None,
                            dst_type=None, quant_mode="pertoken"):
    if dst_type == torch.quint4x2:
        y_shape = list(input.shape)
        y_shape[-1] = y_shape[-1] // 8
        return (input.new_empty(y_shape, dtype=torch.int32),
                input.new_empty(input.shape[:-1], dtype=torch.float32))
    return (input.new_empty(input.shape, dtype=torch.int8),
            input.new_empty(input.shape[:-1], dtype=torch.float32))
```

**编译**：

```bash
rm -rf build
bash ci/build.sh --python=3.10 --pytorch=v2.10.0-26.0.0 2>&1 | tee build.log
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

后续步骤（测试/文档/基线/总结）自动执行。

#### 最终产物

| 文件 | 操作 | 说明 |
|------|------|------|
| `CHECKLIST.md` | 新增 | 集成检查清单 |
| `op_plugin/config/op_plugin_functions.yaml` | 修改 | 新增算子函数签名 |
| `op_plugin/ops/opapi/DynamicQuantKernelNpuOpApi.cpp` | 新增 | C++ 实现 |
| `op_plugin/python/meta/_meta_registrations.py` | 修改 | Meta 注册 |
| `test/test_custom_ops/test_npu_dynamic_quant.py` | 新增 | UT 测试 |
| `test/core_tests/test_fake_tensor.py` | 修改 | FakeTensor 测试 |
| `docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md` | 新增 | API 文档 |
| `docs/zh/custom_APIs/menu_Pytorch_API.md` | 修改 | 导航菜单 |
| `docs/zh/custom_APIs/torch_npu/torch_npu_list.md` | 修改 | 算子列表 |
| `codegen/templates/_op_plugin_docs.py` | 修改 | Python docstring |
| `test/core_tests/torch_npu_OpApi_schema_all.json` | 修改 | 兼容性基线 |
| `test/allowlist_for_publicAPI.json` | 修改 | 公开 API 白名单 |
| `notes/summary_{YYYY-MM-DD}.md` | 新增 | 集成摘要 |

#### 验收重点

产物共 13 个文件，验收时重点看 4 处：

- `CHECKLIST.md`（项目根）—— 全流程单一事实源：参数信息表、约束覆盖、
  测试结果的汇聚点。
- C-verify 签名校验报告 —— 出现 BLOCK 必须先修复才能编译；图模式算子另有
  Meta Shape/Dtype 交叉验证（防图模式静默错误）。
- op-test 结构化测试报告 —— Passed/Failed/Skipped 计数；失败按三分支定性
  （UT 自身 bug / 适配层 bug / 环境问题）。
- `notes/summary_{YYYY-MM-DD}.md` —— 集成摘要终稿；注意 C-delimit 小节，
  ⚠️ 表示无 ATK 输入跳过了用例验证，不代表验证通过。

#### 卡住了怎么办

流程非线性，常见三类卡点与处置：

| 卡点 | 现象 | 怎么办 |
|------|------|--------|
| C-verify BLOCK | 签名校验报 BLOCK（YAML↔C++↔Meta 不一致） | **必须先修**再编译。把 BLOCK 报告贴回对话，Claude 会定位是哪份文件签名漂了并改。不要跳过。 |
| 编译失败 | `ci/build.sh` 报错 | 保留 `build.log`，把报错段贴回对话。多为签名 / 头文件 / 类型不匹配，Claude 依日志修；环境类错误（CANN 未 source）按"准备"章节自查。 |
| 测试 Failed | op-test 报告有 Failed | 看 op-test 的三分支定性：**UT 自身 bug**（改用例）/ **适配层 bug**（改 C++/Meta）/ **环境问题**（查驱动 / CANN）。定性由 op-test 子技能给出，照分支处置。 |

> 原则：**失败不静默跳过**。贴日志 > 自己猜。C-verify BLOCK 与编译失败是硬门，必须清零才能继续。

### Modify: npu_moe_token_permute_grad_v2

**需求**：`npu_moe_token_permute_grad` 的 `tokens` 和 `indices` 输入仅用于提取标量属性（shape[0]、dtype、topK），导致反向传播时持有不必要的大 Tensor 引用，浪费显存。精简接口为直接传入标量值。

**调用**：

```bash
/op-adaptation 修改 npu_moe_token_permute_grad，
参数 tokens Tensor → tokens_size_0 int + tokens_dtype ScalarType，
参数 indices Tensor → num_topK int
```

**grep 全量搜索** — 发现：

```
  ✅ op_plugin_functions.yaml            — 已有 func 条目
  ✅ derivatives.yaml                    — 已绑定到 forward npu_moe_token_permute
  ✅ KernelNpuOutputSize.cpp/h           — 已有 _out_size 函数
  ✅ _meta_registrations.py              — 已有 meta 注册
  ✅ test_fake_tensor.py                 — 已有测试类
  ✅ torch_npu_OpApi_schema_all.json     — 已有 schema 条目
```

**推导影响范围** — 接口变更 → yaml + meta + output_size + test + json + derivatives(rebind)，共 6 类文件。用户确认后执行。

**执行（子集模式）**：只修改以上文件 + 编译 + 测试。跳过解析/确认/约束检查。Derivatives rebind 将旧公式替换为新 variant：

```yaml
# 旧
tokens: npu_moe_token_permute_grad(tokens, grad, indices, result1, padded_mode)
# 新
tokens: "npu_moe_token_permute_grad_v2(grad, result1, tokens.sym_size(0).expect_int(), tokens.scalar_type(), (indices.dim() == 1) ? 1 : indices.sym_size(1).expect_int(), padded_mode)"
```

### M-adapt: npu_ciou A5 平台适配

**需求**：`npu_ciou` 已有 A2/A3 的 ACL 实现，需新增 Atlas A5（Ascend950）平台 ACLNN 路径（`aclnnCIoU`），接口签名不变。

**调用**：

```bash
/op-adaptation 给 npu_ciou 适配 A5 平台
```

**M-discover 自动检测** — 4 条件全部满足，触发 M-adapt 建议：

```
✓ YAML 中 npu_ciou 仅有 acl_op: all_version（无 op_api:）
✓ op_plugin/ops/opapi/CiouKernelNpuOpApi.cpp 不存在
✓ op_plugin/ops/acl/ 下存在 CiouKernelNpu.cpp（ACL 老实现）
✓ 接口签名未变更
→ 建议执行 M-adapt 模式
```

**执行（M-adapt 子集）**：跳过 C-parse/C-confirm/C-derivatives/C-constraints，执行：

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `op_plugin_functions.yaml` | 修改 | `_npu_ciou`/`npu_ciou`/`npu_ciou_backward` 加 `op_api: all_version` |
| `CiouKernelNpuOpApi.cpp` | 新增 | `GetSocVersion()` 路由 + fp16→fp32 转换 + `EXEC_NPU_CMD(aclnnCIoU)`；反向走硬报错 |
| `_meta_registrations.py` | 新增 | `_npu_ciou`/`npu_ciou` 的 Meta 注册 |
| `test_ciou.py` | 修改 | 旧用例加 `@SupportedDevices(['Ascend910A','Ascend910B'])`，新增 `['Ascend950']` 用例 |

**核心路由代码**：

```cpp
if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
    return acl_op::npu_ciou(...);   // 老平台走 ACL
}
// A5 走 ACLNN：fp16→fp32 → aclnnCIoU → fp32→fp16
```

**关键差异点**：dtype 转换（aclnnCIoU 不支持 fp16）+ 反向硬报错（A5 不支持 backward，`TORCH_CHECK(false, ...NOT_SUPPORT)`）。详见 `references/examples/m-adapt/npu_ciou-a5.md`。

---

## 扩展指南

技能已解耦为骨架（SKILL.md）+ 领域知识（references）。扩展时影响范围极小。

### 新增步骤

在 C++ 和约束检查之间插入 "Affinity 优化"：

| 操作 | 影响 |
|------|------|
| 在 `references/guides/` 新增 `affinity-guide.md` | 1 个新文件 |
| 在 SKILL.md 流程列表中插入新步骤行 | SKILL.md 1 行 |
| 后续步骤顺序号顺延 | SKILL.md 仅顺序号 |
| references / templates / examples | **零改动**（语义 ID 不变） |

### 新增模式

如 "Auto-Fix 模式"：

| 操作 | 影响 |
|------|------|
| 在 `references/guides/` 新增 `autofix-guide.md` | 1 个新文件 |
| 在 SKILL.md Mode Selection 表加一行 | SKILL.md 1 行 |
| 在 SKILL.md 新增 Auto-Fix 步骤列表 | SKILL.md ~10 行 |
| 在 modify-guide.md Mode-Specific Differences 表追加列 | 1 列 |

### 新增 guide / template / case

| 类型 | 位置 | 命名 | 示例 |
|------|------|------|------|
| 操作指导 | `references/guides/` | `{步骤}-guide.md` | `affinity-guide.md` |
| 模板 | `references/templates/` | `{名称}-template.md` | `checklist-template.md` |
| 案例 | `references/examples/{create-mode|modify-mode|m-adapt}/` | 场景名 | `interface-refactor.md` |

案例使用标准格式：**场景 → 变更类型 → 关键决策 → 文件变更 → 踩坑**。参考 `templates/case-template.md`。新增案例只需在 `examples/README.md` 索引表加一行。

### 命名规范

```
SKILL.md                   — 纯流程骨架
guides/{step}-guide.md     — 操作方法指导
templates/{name}-template.md — 模板文件
examples/{create-mode|modify-mode|m-adapt}/            — 案例库
```

SKILL.md 使用语义 ID（`C-parse`、`C-cpp`、`M-discover` 等），references 内一律引用语义 ID。插入新步骤时语义 ID 不变，仅调整流程列表的顺序号。

---

## 相关文档

### 操作指导（references/guides/）

| 文档 | 对应步骤 |
|------|----------|
| `parse-guide.md` | 解析输入 |
| `confirm-guide.md` | 确认信息 |
| `yaml-guide.md` | 生成 YAML |
| `derivatives-guide.md` | 配置 Derivatives（新建 + 换绑） |
| `cpp-guide.md` | 生成 C++ + 约束覆盖检查 |
| `meta-guide.md` | 注册 Meta |
| `verify-guide.md` | 签名校验 |
| `modify-guide.md` | Modify 模式入口 |
| `compatibility-guide.md` | 注册兼容性基线 |

### 模板（references/templates/）

| 文档 | 用途 |
|------|------|
| `checklist-template.md` | 集成检查清单 |
| `summary-template.md` | 集成摘要模板 |
| `case-template.md` | 案例格式模板 |

### 案例（references/examples/）

| 文档 | 用途 |
|------|------|
| `README.md` | 案例索引表 |
| `create-mode/` | Create 模式案例 |
| `modify-mode/` | Modify 模式案例 |
| `m-adapt/` | M-adapt（A5 平台适配）案例 |

### 子技能

**op-test**（测试生成与执行）：

| 文档 | 用途 |
|------|------|
| `SKILL.md` | 测试工作流（UT + FakeTensor） |
| `references/test-template.md` | UT 测试文件模板 |
| `references/test-coverage-guide.md` | 测试覆盖指南 |
| `references/fake-tensor-guide.md` | FakeTensor 测试模板（Pattern A-E） |
| `references/patterns/custom-ops.md` | 自定义算子测试模式 |
| `references/patterns/native-ops.md` | 原生算子测试模式 |

**op-apidoc**（API 文档生成）：

| 文档 | 用途 |
|------|------|
| `SKILL.md` | 文档生成工作流 |
| `references/writing-standards.md` | 参数/约束/示例写作规范 |
| `references/template.md` | API 文档模板 |
| `references/docstring-template.md` | Python docstring 模板 |

**op-delimitation**（API 一致性定界）：

| 文档 | 用途 |
|------|------|
| `SKILL.md` | 套件总入口（Gate 0 分流 → ATK 执行 → 归因定界 → 结单报告） |

由 Create 模式 C-delimit 步骤经子代理（`.claude/agents/op-delimitation.md`）调用；ATK 用例目录为可选输入，由用户提供，主流程不读取。
