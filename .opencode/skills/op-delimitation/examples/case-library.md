# 历史 DTS 问题单案例库（详情）

> **说明**：本文件基于历史问题单分类，根因模式对 PTA ATK 流程仍有参考价值。内部交叉引用指向已删除的文件（dts-reproduce-cases.md / reproduce 流程 / DTS 单号），链接已失效，仅参考根因模式本身。

> **来源**：原 `op-delimitation-a2a3/knowledge/history.md`（712 行）整编而来，作为详情库归档于 `examples/`。
> **与本套件索引的关系**：[`knowledge/history.md`](../knowledge/history.md) 是 **精简索引 + 高频根因速查**；本文件是 **逐 API 的根因/结论详情**，供 Step 3 定位时按 API 名 / 症状 **由用户对照检索**。
> **纪律**：助手 **提示用户** 用关键词检索，**不**自动抓取内网表、**不**编造命中行（与 [A2/A3/A5 纪律](../knowledge/a2-a3-a5-discipline.md) 一致）。

---

## 一、结论分类总览

| 结论类型 | 数量 | 说明 |
|---------|------|------|
| 非问题 | ~25 | 输入不一致、标准变更、环境差异等 |
| 用例/测试问题 | ~30 | 测试脚本 bug、用例设计缺陷、range 不合理 |
| 算子问题（转第三方） | ~18 | 单算子精度/功能问题，PTA 侧无需修改 |
| PTA 代码修复 | ~25 | torch_npu 适配代码 Bug，已修复或需修复 |
| 精度标准/配置问题 | ~12 | 标杆配置错误、标准不匹配 |
| CANN 修复 | ~8 | CANN 版本升级后解决 |
| ATK 工具问题 | ~6 | ATK 框架 Bug |
| 环境/版本问题 | ~5 | CANN 版本不匹配、API 版本适配 |

---

## 二、API 问题详情

### 2.1 功能报错类

#### torch_npu.npu_grouped_matmul_swiglu_quant_v2
- **类型**：功能
- **根因**：A2 torch 资料和 A5 torch 资料未保持一致，导致 dequant_dtype 可选参数校验出错。dequant_dtype 校验拦截了 A3 通路，修改为仅在 A5 下校验该参数。
- **结论**：PTA 代码修复（`GroupedMatmulSwigluQuantV2NpuOpapi.cpp`）

#### torch_npu.npu_fusion_attention（PT_UT_Dtensor）
- **类型**：功能
- **根因**：`npu_fusion_attention_grad` 的 meta 注册中将 pse 的梯度值返回为 None，与分片策略中固定为 `Replicate()` 冲突。
- **结论**：PTA 代码修复

#### torch.nn.LSTM
- **类型**：功能（bf16 不支持 / 混精 / 拦截报错）
- **根因**：接入 aclnn 后：①不支持 bf16（aclop 支持，因底层 ge 插件做了 cast）；②混精计算不支持（GPU 本身不支持）；③hx 列表 tensor 数量拦截。PTA 通过判断输入数据类型回退 aclop 实现。
- **结论**：PTA 适配修复（FrameworkPTAdapter 26.0.0.B101）

#### torch.nn.LSTM（反向）
- **类型**：功能
- **根因**：lstm 反向算子报错，需算子进一步定位。
- **结论**：转算子

#### torch.nn.TripletMarginLoss
- **类型**：功能
- **根因**：迭代新需求 norm 算子 aclop 切换 aclnn，aclnn 只支持 p=0,1,2,3,inf,-inf 场景。错误用例 p=10，原本应走 aclop 逻辑，PTA 版本更新后走 aclnn，但使用的 CANN 包不包含新需求代码，报错找不到二进制文件。
- **结论**：CANN 版本不匹配（CANN B080 已解决）

#### Tensor.histc
- **类型**：功能拦截
- **根因**：原代码为对齐 CUDA 校验逻辑，对 dtype 为 float16 的输入进行了拦截。
- **结论**：PTA 代码修复（FrameworkPTAdapter 26.0.0 B091）

#### torch.nn.functional.linear
- **类型**：功能拦截
- **根因**：算子侧新增了拦截逻辑，但 CPU/GPU 侧支持该场景输入。最小复现：`torch.nn.functional.linear(torch.randn(16,1).npu(), torch.randn(8,1).npu(), torch.tensor(3.14).npu())`
- **结论**：CANN 修复（CANN 9.0.0 B073）

#### torch_npu.npu_quant_matmul
- **类型**：功能
- **根因**：x2_offset 传参有误。
- **结论**：PTA 代码修复（B091 已归档）

#### torch_npu.npu_nsa_compress_infer
- **类型**：功能
- **根因**：`aclnnNsaCompressWithCache` 算子侧校验不合理，ND 格式强校验拦截不合理（ND 应泛指非私有格式）。
- **结论**：CANN 修复（CANN B095）

#### torch.cdist
- **类型**：功能 / 精度
- **根因**：①compute_mode 只支持 2，传入默认值报错拦截，PTA 透传参数引出报错；②aclnn bf16 不支持（原 aclop 支持）；③反向 DSL 算子无法根据输入属性 p 走入特殊分支，导致精度不通过单标杆。
- **结论**：PTA 适配 + CANN 修复（compute_mode 参数映射已合入 op-plugin PR #4381）

#### torch.matmul（Dtensor 支持）
- **类型**：功能
- **根因**：meta 函数实现漏掉了一种情况，已补充。
- **结论**：PTA 代码修复

#### torch.linalg.norm（ord=nuc）
- **类型**：功能拦截
- **根因**：`aclnnLinalgVectorNorm`，ord=nuc 模式下单算子调用 NPU 报错拦截。
- **结论**：算子问题（新版本 CANN 已修复）

#### Tensor.renorm / Tensor.dist
- **类型**：功能拦截
- **根因**：底层调用 `aclnnLinalgVectorNorm`，被拦截报错 "Tensor out not implemented for DT_DOUBLE"。
- **结论**：算子问题

#### torch.angle / Tensor.angle
- **类型**：功能
- **根因**：报错 `aclnnAngleV2 or aclnnAngleV2GetWorkspaceSize not in libopapi.so, or libopapi.so not found`。
- **结论**：环境/版本问题

#### torch.logspace
- **类型**：功能
- **根因**：报错（无详细分析）
- **结论**：待确认

#### torch.sparse_coo_tensor
- **类型**：功能
- **根因**：用例执行报错（无详细分析）
- **结论**：待确认

#### torch_npu.npu_fusion_attention_v3（入图）
- **类型**：功能
- **根因**：高版本 torch（2.10/master）的 `build_device_mesh` 接口发生变化，测试用例使用了该接口导致 device mesh 被设置到 CPU 导致报错。
- **结论**：环境/版本适配

#### torch.cdist（temu 反向模型）
- **类型**：精度（反向）
- **根因**：基于 CANN 9.0.0.B020 回归。
- **结论**：CANN 修复

#### B060 fx 入图报错
- **类型**：功能
- **根因**：`Failed to get function: "profRegCtrlCallback" from "libprofapi.so"`
- **结论**：环境问题

#### torch_npu.npu_add_rms_norm_quant
- **类型**：功能 / 精度 / 资料
- **根因**：①PTA 适配缺少一个输出（暂不修改）；②资料描述多处存在错误（PR #4478 已修复）；③精度不达标场景未约束（算子侧已修改）；④dtype 校验针对 x1 不准确（算子侧已修复）。
- **结论**：PTA 资料修复 + 算子修复

#### torch_npu.npu_dense_lightning_indexer_softmax_lse
- **类型**：功能（资料）
- **根因**：示例代码中 layout 参数传递错误（传了未定义的变量 input_layout）。
- **结论**：PTA 资料修复

---

### 2.2 精度问题类

#### torch.nn.TransformerEncoder.forward
- **类型**：精度（全量 + 小值域）
- **根因**：①正反向全量精度：ThresholdBackward 算子输入存在极小误差（0 和 10e-9），导致走入不同分支引起误差跳变，单算子满足精度标准，组合场景无法避免；②小值域：API 由多个小算子拼接，fp32 下放宽单标杆最大绝对误差标准由 2^-14 改为 2^-10。
- **结论**：已有结论（修改用例，切换为单标杆）

#### torch.nn.TransformerEncoder.forward（边界值）
- **类型**：精度（边界值）
- **根因**：num_layers 太大导致误差累计。mask 为 float 类型时 CPU 与 GPU/NPU 实现路径不一致。改为 num_layers=1 + 切换 GPU 单标杆可通过。
- **结论**：已有结论

#### Tensor.nanmean / torch.nanmean
- **类型**：精度
- **根因**：源码实现为先 nansum 再 div，nansum 本身无精度问题，div 操作引入精度误差。双标杆场景下个别用例精度不达标，重复压测后通过统计验证。
- **结论**：非问题（统计验证通过）

#### Tensor.subtract_
- **类型**：精度 / 确定性
- **根因**：subtract_ 没有单独实现，直接转给 sub_，sub_ 直接调用 `aclnnInplaceSubs`。老算子需切换双千双万标准。反向涉及多个算子（aclnnInplaceFillScalar、aclnnDiv、aclnnMul 等）。
- **结论**：算子问题（继承已有结论 [DTS 单号]）

#### torch.expm1
- **类型**：精度
- **根因**：底层算子存在精度问题，新版本已修复。
- **结论**：CANN 修复（同步问题单 [DTS 单号]）

#### torch._fused_adamw_
- **类型**：精度 / 确定性
- **根因**：①确定性精度问题：修改用例 range 后通过；②全量精度不达标：框架侧只做校验和传递，调用 `aclnnApplyAdamWV2`，输入 range 为 (-inf, inf) 会出现精度问题，改为 (-100, 100) 通过。
- **结论**：用例 range 问题

#### torch._addmm_activation
- **类型**：精度 / 确定性
- **根因**：内部用 addmm_out，底层用 `aclnnAddmm` 算子有精度问题。需配置 CubeMathType + 修改精度标准为 high_performance 单标杆。
- **结论**：算子精度问题 + 配置调整

#### torch.nn.Embedding
- **类型**：精度
- **根因**：norm_type 为 0 时，`aten::embedding_renorm_` 修改了 Embedding 里面的 weight，且与 CPU 和 CUDA 修改不一致。底层调用 `aclnnEmbeddingRenorm` 算子，该算子在 norm_type=0 时实现与 CPU/CUDA 都不一致。
- **结论**：算子问题

#### torch.nn.functional.kl_div
- **类型**：精度（反向 target）
- **根因**：kl_div 原本没有 target 反向算子，导致与标杆输出不一致。现已新增 `aclnnkldivtargetbackward` 算子，框架侧也已接入。涉及 vexp 和 vlog 计算操作，需改用"单标杆正确性"标准。
- **结论**：PTA 修复（PTA B091, CANN B091）

#### torch.nn.KLDivLoss
- **类型**：精度 / 内存 / 确定性
- **根因**：正向精度：`aclnnKIDiv` 算子涉及 vcadd 计算操作，需改用单标杆正确性标准。正向内存：不要使用 ATK 批跑易误报。正向确定性：改为和 CPU 单标杆比。反向精度：已支持 target 反向 `aclnnkldivtargetbackward`。
- **结论**：标准问题 + 算子更新（同 [DTS 单号]）

#### torch.linalg.svd
- **类型**：精度 / 确定性
- **根因**：`aclnnSvd` 底层采用 TensorFlow 源码实现，与 PyTorch 实现采用的数学库不一致。反向 RuntimeError 相关文档已有结论。
- **结论**：已有结论（非问题）

#### torch.topk
- **类型**：精度 / 确定性（反向）
- **根因**：输入存在重复数值时，topk 正向排序具有不稳定性（海思 2023 年已有结论），导致 NPU 和 CPU 正向输出的 index 不同，引起反向梯度分配的错位。梯度值一致但位置对调。
- **结论**：已有结论（非问题，正常现象）

#### torch.where
- **类型**：精度 / 确定性（反向）
- **根因**：ATK 用 torch.equal 作为精度标准（要求二进制一致）。实际误差很小(1e-6~1e-8)，误差总出现在 'other' 的反向梯度上，由 reduce_sum 归约操作引入。CPU 和 NPU 对浮点数 reduce_sum 实现不完全一致导致二进制差异。
- **结论**：已有结论（非问题，reduce_sum 引入的正常误差）

#### torch.sin / Tensor.sin（反向）
- **类型**：精度 / 确定性（反向）
- **根因**：sin 反向为 cos、mul 等小算子拼接实现：`grad * self.cos().conj()`，底层调用 `aclnnmul`、`aclnncos`。fp16/bf16 场景下小算子拼接累计误差符合 aclnn 历史算子 Float16/Bfloat 类型精度泛化问题场景。
- **结论**：已有结论（算子 fp16/bf16 精度泛化问题，不解决）

#### torch.addmm / Tensor.addmm（正向）
- **类型**：精度 / 小值域 / 边界值 / 确定性
- **根因**：`aclnnAddmm` 单算子精度用例失败。fp32 场景下 GPU 切 K 但 NPU 未切 K，已有结论。需调整标杆为双标杆 L0、range 改为 [-5,5]、cubeMathType 配置为 5（USE_FP32_ADDMM）。
- **结论**：算子问题（已有结论）

#### torch.addmm / Tensor.addmm（反向）
- **类型**：精度 / 小值域（反向）
- **根因**：CANN 已知问题，对齐算子一致性标准改范围。
- **结论**：CANN 已知问题

#### torch.baddbmm
- **类型**：精度 / 确定性（正向 + 反向）
- **根因**：cubeMathType 配置为 4/5 用上融合算子升精度特性即可解决。需 PTA 适配 + 修正用例文件 + cube_math_type 环境配置。
- **结论**：PTA 适配修复（PTA 开发者实现，FrameworkPTAdapter 26.0.0.B107）

#### torch.baddbmm（另一单）
- **类型**：精度 / 确定性
- **根因**：正向底层是 `aclnnBaddbmm` 算子；反向涉及多个算子，dispatch 定位于 muls 算子，复检后可通过对精度标准。
- **结论**：算子问题 + 用例复检通过

#### torch.bmm / Tensor.bmm（反向）
- **类型**：精度（反向）
- **根因**：有问题的 case 输入 range 是 [-inf, inf]，超出精度标准的数据值域分布范围。改为 [-5,5] 后全部 pass。
- **结论**：用例 range 问题

#### torch.mm / Tensor.mm（正向 + 反向）
- **类型**：精度
- **根因**：matmul 算子精度问题（fp16 类型），已有结论。
- **结论**：算子问题（已有结论 [DTS 单号]）

#### Tensor.matmul（反向）
- **类型**：精度 / 内存
- **根因**：精度问题：fp32 场景涉及 3 次 `aclnnMatMul` 运算，fp32 下有已知精度问题，已有结论。内存问题：新版本下执行通过。
- **结论**：算子问题（已有结论）+ 内存非问题

#### torch.nn.functional.binary_cross_entropy
- **类型**：精度
- **根因**：单算子双标杆检测 fail。修改 input range 后部分用例 pass，部分 fail。双标杆复检也 fail，最大相对误差比例 29.7 超阈值 10.0，小值域错误差比例 6.0 超阈值 2.0。
- **结论**：算子问题

#### torch.nn.functional.adaptive_max_pool3d
- **类型**：精度
- **根因**：已和 CANN 对齐，同步转单给算子。
- **结论**：转算子

#### torch.nn.functional.interpolate
- **类型**：精度 / 确定性
- **根因**：部分场景下不支持传入 scale_factor 导致精度问题，算子侧已修复。需要使用 GPU 双标杆精度。
- **结论**：CANN 修复

#### torch.optim.AdamW
- **类型**：精度 / 确定性
- **根因**：AdamW 算子存在已知精度问题（已有结论 [DTS 单号]）。由小算子组成，调用 addcdiv_ 算子（已有已知精度结论）。div、mul、add 计算过程中因数据存储位数限制有尾数截断误差。
- **结论**：已有结论

#### torch._foreach_lerp
- **类型**：精度 / 确定性
- **根因**：调用 `aclnnForeachLerpList`，与算子确认精度标准为单标杆。
- **结论**：已有结论

#### torch._foreach_tanh_
- **类型**：精度 / 确定性
- **根因**：与 CANN 确认，更新阈值标准后可 pass。
- **结论**：精度标准问题

#### torch._foreach_add / torch._foreach_add_
- **类型**：精度 / 确定性
- **根因**：底层在不同 case 时调用不同算子（`aclnnadds`/`aclnnInplaceadds` 或 `aclnnForeachAdd` 系列）。`aclnnadds`/`aclnnInplaceadds` 测试标杆为双千双万，`aclnnForeachAdd` 在部分 bfloat16 场景使用双标杆。
- **结论**：精度标准/用例配置问题

#### torch._foreach_addcdiv
- **类型**：精度 / 确定性
- **根因**：测试用例随机生成 choice 走不同分支。正向：choice=0→`aclnnForeachAddcdivScalarList`，choice=1→`aclnnAddcdiv`，choice=2/3→`aclnnForeachAddcdivScalarV2`。反向：`aclnnInplaceFillScalar`、`aclnnDiv`、`aclnnMul`、`aclnnNeg`、`aclnnMuls` 等。
- **结论**：已有结论

#### torch._foreach_norm
- **类型**：精度 / 确定性
- **根因**：确定性复测为非问题。精度：foreach_norm 输出只有一个值，不适用均方根误差来衡量，建议改较大阈值过滤误判（如 10000）。
- **结论**：精度标准问题

#### torch.pow（反向）
- **类型**：精度 / 确定性 / 小值域（反向）
- **根因**：`aclnnPowTensorScalar`，根据 CANN 结论，该算子无改动，继承 CANN 已有看护作为对外口径。
- **结论**：算子问题（继承 CANN 结论）

#### Tensor.pow（反向）
- **类型**：精度（反向）
- **根因**：全量 3 个 case fail。其中 2 个 fp32 底层调用 `aclnnReduceSum` 算子（已有结论），1 个 complex64 底层调用 `aclnnMul` 算子（已有结论）。
- **结论**：算子问题（已有结论）

#### Tensor.add（反向）
- **类型**：精度 / 小值域 / 确定性（反向）
- **根因**：反向调用 `aclnnReduceSum` 算子，该算子在单标杆高精度标准下精度不通过（已有结论）。改成 cv_fused_double_benchmark 精度标准后全部 pass。
- **结论**：精度标准问题（ReduceSum 需用双标杆）

#### torch.div / Tensor.div（反向）
- **类型**：精度（反向）
- **根因**：精度失败定位为 ReduceSum 算子，在双千双万精度标准下无法满足要求。ATK 失败用例（float64 → A2/A3 降级为 float32 触发 Autograd dtype mismatch）已有 SEG 结论。
- **结论**：精度标准问题 + 已有 SEG 结论

#### torch.remainder / Tensor.remainder（反向）
- **类型**：精度 / 小值域（反向）
- **根因**：底层调用 DivMod + Neg + Mul + ReduceSum（if broadcast），误差主要由 DivMod 算子引入，已有结论（[DTS 单号]）。
- **结论**：算子问题（已有结论）

#### torch.repeat_interleave（反向）
- **类型**：精度 / 小值域 / 确定性（反向）
- **根因**：反向调用 `aclnnReduceSum` 算子，确认使用双标杆。双标杆测试结果：小值域、确定性通过，反向精度有 2 个不达标用例，复检通过。
- **结论**：精度标准问题（reduce_sum 需双标杆）

#### torch.fill_ / Tensor.fill_（反向）
- **类型**：精度 / 小值域（反向）
- **根因**：fill_ 反向用了 sum 和 reduce_sum，不能用同正向的 md5 标准，应同 reduce_sum 使用双千双万双标杆。
- **结论**：精度标准问题（已有结论 [DTS 单号]）

#### Tensor.cumsum / torch.cumsum（反向）
- **类型**：精度 / 小值域（反向）
- **根因**：精度标杆应为双标杆。
- **结论**：精度标准问题

#### torch.clamp_min
- **类型**：精度 / 内存 / 小值域 / 边界值
- **根因**：①正向/反向内存：ATK 工具引入的测试问题，API 本身内存正常；②正向精度：md5 标准问题（+0 和 -0 二进制不一致，但数学一致），应换成 equal 精度标准；③反向精度：min 维度 shape 与 input 不一致导致广播。
- **结论**：非问题（ATK 工具 + md5 标准 + 用例问题）

#### torch.clamp（反向）
- **类型**：精度 / 小值域（反向）
- **根因**：反向调用 `>=` 算子，NPU 升精度处理，CPU 没有升精度，已有结论。
- **结论**：已有结论（非问题，跳过用例）

#### Tensor.clamp（反向）
- **类型**：精度（反向）
- **根因**：NPU 会调用 broadcast 操作，ATK 构造的数据 shape 不满足 CANN 中的 broadcast 限制导致失败。24 个失败用例中有 23 个单独执行可通过，1 个（id=372）不满足限制。
- **结论**：算子问题（broadcast 限制）

#### torch.scatter_add / Tensor.scatter_add / Tensor.scatter_add_
- **类型**：精度（确定性）
- **根因**：部分算子已切换双标杆看护。CANN 采用双标杆覆盖全量用例。
- **结论**：精度标准问题

#### Tensor.__imul__
- **类型**：精度（正向，complex128 和 fp16）
- **根因**：`aclnnInplaceMuls`，框架侧精度标准与 CANN 精度标准不一致。根据 CANN 结论该算子无改动，继承 CANN 已有看护。
- **结论**：精度标准问题

#### torch.index（反向）
- **类型**：精度 / 小值域 / 功能（反向）
- **根因**：`aclnnIndexPutImpl`，框架侧精度标准与 CANN 精度标准不一致。根据 CANN 结论该算子无改动。
- **结论**：精度标准问题

#### torch.maximum / Tensor.maximum / torch.minimum / Tensor.minimum
- **类型**：精度（确定性）
- **根因**：输入 uint8 tensor([12]) 和 int8 tensor(-128) 时，minimum NPU 得 128，CPU/GPU 得 12。NPU 内部先升精度比较（得 -128），再 cast 到 uint8（得 128），而 CPU 先 cast 再比较（得 12）。NPU 结果错误。
- **结论**：算子问题（转算子）

#### torch.rsqrt / Tensor.rsqrt（反向）
- **类型**：精度 / 小值域 / 确定性（反向）
- **根因**：反向走 PyTorch 原生 autograd 流程，为 rsqrt、pow、mul 小算子拼接。小算子拼接误差叠加放大，算子侧已有结论。失败用例均为 fp16 类型。
- **结论**：算子问题（已有结论）


#### torch.as_tensor
- **类型**：精度
- **根因**：CPU 标准变更：CPU + PyTorch 2.1.0 与 CPU + PyTorch 2.7.1 结果不一致，NPU 未变，且 NPU 与 CPU+2.1.0 及 GPU 结果一致。
- **结论**：非问题（CPU 标准变更）

#### torch.nn.functional.ctc_loss（精度）
- **类型**：功能（精度不通过）
- **根因**：无详细分析。
- **结论**：待确认

#### torch.histc
- **类型**：精度
- **根因**：无详细分析。
- **结论**：待确认

#### torch.distributions.categorical.Categorical.log_prob
- **类型**：精度（边界值）
- **根因**：调用 `aclnnGather` 算子。CPU gather 越界报错，NPU gather 越界返回脏数据，ATK 比对脏数据导致精度不通过。`aclnnGather` 算子缺少越界边界检查，由于硬件限制无法在 kernel 做报错。
- **结论**：非问题关闭（CANN 确认硬件限制）

#### Tensor.bernoulli_ / torch.bernoulli
- **类型**：精度（边界值）
- **根因**：底层算子边界值精度问题，与 PTA 侧无关。
- **结论**：算子问题（CANN 9.0.0.B086 修复）

#### Tensor.resize_
- **类型**：精度
- **根因**：已提 JDC 需求，RAT 不接纳需求，需求不做，问题单关单。
- **结论**：需求不解决

#### Torchvision.transforms.Resize
- **类型**：精度（反向）
- **根因**：`aclnnUpsampleBicubic2dAAGrad` 有差异。
- **结论**：算子问题（转算子）

#### torch_npu.npu_dense_lightning_indexer_softmax_lse
- **类型**：精度
- **根因**：ATK 工具需刷新 compute_times，不刷新则使用默认值导致精度标准阈值不对。
- **结论**：用例/配置问题

#### torch_npu.npu_dense_lightning_indexer_grad_kl_loss
- **类型**：精度（fp32）
- **根因**：算子已更新 fp32 场景修改，PTA 基于最新 CANN 包自验通过。
- **结论**：CANN 更新

#### torch_npu.npu_swiglu
- **类型**：精度（反向）
- **根因**：问题来自 `aclnnSwiGluGrad`，算子给出分析和用例修改建议，修改用例后精度通过。
- **结论**：非问题（用例修改后通过）

#### torch.all / Tensor.all
- **类型**：精度
- **根因**：底层调用 `aclnnAll` 算子精度不达标。（无详细分析）
- **结论**：待确认

#### Tensor.equal
- **类型**：精度
- **根因**：底层调用 `aclnnEqual`，带 inf 的 tensor 精度不达标。（无详细分析）
- **结论**：待确认

#### torch.threshold
- **类型**：精度 / 确定性
- **根因**：无详细分析。
- **结论**：待确认

#### torch.optim.AdamW（fp16 精度）
- **类型**：精度（fp16）
- **根因**：无详细分析。
- **结论**：待确认

---

### 2.3 确定性问题类

#### torch.nn.functional.max_pool2d
- **类型**：确定性
- **根因**：精度不过的用例 output 均为全 nan tensor。
- **结论**：非问题（删除用例）

#### torch.nn.functional.adaptive_avg_pool2d
- **类型**：确定性（反向）
- **根因**：`aten._adaptive_avg_pool2d_backward` 直调 aclnn 算子，确定性不固定（连续两次 load 对比 fail，约 e-7 左右）。使用环境变量后确定性固定。
- **结论**：转算子

#### torch.distributions.uniform.Uniform.rsample
- **类型**：确定性
- **根因**：ATK 测试脚本问题，直接 return 输入就会出现确定性问题。修改测试脚本不对 input_data 进行赋值操作（赋值会修改输入）。
- **结论**：非问题（ATK 不推荐在测确定性时进行输入赋值操作）

#### torch.distributions.multivariate_normal.MultivariateNormal.rsample
- **类型**：确定性
- **根因**：组合 API 包含 `Tensor._normal()` 创建正态分布，是随机过程。用例中随机种子未固定住，导致执行结果不一致。在 call 函数中加入 seed 设定后全部通过。
- **结论**：用例问题（非问题）

#### torch.distributions.multivariate_normal.MultivariateNormal.log_prob
- **类型**：确定性
- **根因**：确定性精度问题在自定义测试脚本的入参就存在，导致被测试 API 确定性也出现问题。修改自定义脚本确保入参确定性一致后通过。
- **结论**：用例问题

#### torch.nn.functional.leaky_relu
- **类型**：确定性
- **根因**：测试确定性时使用 inplace=True，输入会被改变，导致后续测试结果与之前对不上。
- **结论**：用例问题（非问题）

#### torch.nn.functional.embedding
- **类型**：确定性
- **根因**：max_norm 不为 None 时，会原地修改 weight，导致多次执行前后结果不一致。建议 ATK 中不单独设置 max_norm 参数。
- **结论**：用例问题

#### torch.nn.functional.ctc_loss
- **类型**：确定性
- **根因**：测试脚本 log_probs 被原地修改，导致循环调用时实际输入不完全一致。修改测试脚本后通过。
- **结论**：用例问题（非问题）

#### torch.nn.Bilinear
- **类型**：确定性
- **根因**：每轮 func 的 weight 和 bias 会更新，属于测试用例 bug。确定性的要求是 weight 和 bias 始终不变。将 weight 和 bias 写死后 100% pass。
- **结论**：用例问题

#### torch.stack（反向）
- **类型**：确定性（反向）
- **根因**：2026-01-22 前发布的 ATK 测试工具存在已知 bug。确定性计算中梯度累计：输入导数和输出导数是同一份，梯度累计导致输入导数变化，结果判断为 false。新 ATK 工具已解决。
- **结论**：ATK 工具问题

#### torch._amp_update_scale_
- **类型**：精度 / 确定性
- **根因**：精度问题：算子底层实现没有和 CPU 对齐（算子侧已修复）。确定性问题：ATK 工具 bug（执行 In-place 算子确定性问题时会修改输入，尝试 clone 自定义标杆输入避免原地写入）。
- **结论**：CANN 修复 + ATK 工具问题

#### torch._amp_foreach_non_finite_check_and_unscale_
- **类型**：精度 / 确定性
- **根因**：精度问题：GPU 和 NPU 独立产生的 input.bin 在某些位置有数据误差，加 -sd 参数保证复用同一套输入后修正。确定性问题：ATK 框架缺陷（已提 issue）。
- **结论**：用例问题（加 -sd）+ ATK 工具问题

#### Tensor.logit_
- **类型**：精度 / 确定性
- **根因**：ATK 工具问题（确定性任务直接从内存读取输入，logit_ 是 inplace 操作会修改输入）。
- **结论**：非问题（ATK 工具问题）

#### 33 个确定性 UT 用例 bf16
- **类型**：确定性
- **根因**：`aclnnequal` 算子的问题。
- **结论**：转算子


---

### 2.4 内存一致性问题

#### torch._chunk_cat
- **类型**：内存一致性（>5%）
- **根因**：NPU 走了 native 小算子拼接实现（当前 CANN 包无对应 aclnn 算子 + torch_npu 无 aclnn 路径适配）。更换 CANN + torch_npu 最新版本后通过（aclnn 适配代码 3.25 合入：`ChunkCatKernelOpApi.cpp`）。
- **结论**：环境/版本问题（aclnn 路径适配后解决）

#### torch.split_with_sizes_copy
- **类型**：内存一致性（>5%）
- **根因**：内存占用峰值来自前置语句 `torch.randn(1024, 256, 1024, dtype=float16, device=device)`。将前置语句改用 numpy 后 NPU 与 GPU 内存占用一致，目标接口非问题。
- **结论**：测试脚本问题（非问题）

#### 371 个 API 内存一致性
- **类型**：内存一致性
- **根因**：371 个 API 中仅 1 个 L1 API（torch.ormqr）已满足 L1 标准（<50%），其余为 L2。按策略：单子关闭，下个版本按单独 API 提单。
- **结论**：按策略关闭

---

### 2.5 分布式 / FSDP 相关

#### torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce
- **类型**：精度 / 确定性
- **根因**：返回值包含 `torch.Event` 实例（用于异步执行流同步的底层标记，与 runtime 有关，天然不确定）。对比时将 Event 也进行了对比导致不通过。
- **结论**：用例问题（需排除 Event 对比）

#### torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocAllGather
- **类型**：精度
- **根因**：离线测试 GPU 和 NPU 分别保存输入，md5 码不相同。加 -sd 参数在线验证精度通过。
- **结论**：非问题（输入不同步导致，加 -sd 解决）

#### torch.distributed.fsdp._fully_shard._fsdp_collectives.ProcessGroupAllocReduceScatter
- **类型**：精度 / 确定性
- **根因**：失败用例的 GPU 标杆有问题，NPU 和 CPU 的值一致。失败用例全为 int64 输入，GPU 与 NPU/CPU 输出不同，问题在 GPU 侧。
- **结论**：非问题（GPU 标杆有问题）

#### torch.distributed._functional_collectives.reduce_scatter_tensor
- **类型**：精度 / 确定性
- **根因**：当前版本 prod 操作不支持 int16、bf16 数据类型。已在多版本资料补充说明 + 增加 use_compatible_impl 接口文档说明。
- **结论**：PTA 资料修复

#### torch.distributed._functional_collectives.AsyncCollectiveTensor
- **类型**：精度
- **根因**：离线测试 GPU 与 NPU 输入 md5 不一致，导致输出仅能达 1e-6 精度，二进制比较时精度不通过。
- **结论**：非问题（输入不同步）

#### torch.distributed._reduce_scatter_base
- **类型**：精度 / 确定性
- **根因**：①用例需去掉 dtypes:bf16（reduce-scatter 不支持 prod 操作）；②yaml 需适配异步执行；③部分 int64 用例输入不一致。迁移 GPU 输入到 NPU 全部通过。
- **结论**：用例问题

#### torch.distributed.tensor.placement_types.Shard
- **类型**：精度 / 确定性
- **根因**：浮点数场景都成功，失败仅限整数类型。GPU 和 NPU 输入的二进制不一致，不是接口功能问题。
- **结论**：用例问题

#### torch.distributed.tensor.placement_types._StridedShard
- **类型**：精度 / 确定性
- **根因**：与 Shard 同类型，浮点数成功，整数失败。GPU 和 NPU 输入不一致。
- **结论**：用例问题

#### torch.distributed.tensor._redistribute.redistribute_local_tensor
- **类型**：精度 / 确定性
- **根因**：NPU 和 GPU 在 input 上有区别。该接口在测试场景下实际不进行数值计算，NPU 输入输出结果一致。加 -sd 后通过。
- **结论**：用例问题

#### torch.distributed.tensor.distribute_tensor
- **类型**：精度 / 确定性
- **根因**：ATK 工具在 GPU 和 NPU 上同一种子随机得到的输入不一致（int 类型预期输出一致但因输入差异导致不相等）。
- **结论**：非问题（ATK 工具 + 输入差异）

#### torch.distributed.nn.all_reduce
- **类型**：精度 / 确定性
- **根因**：①OOM：创建通信组后不释放；②确定性：node.yaml 和测试用例配置的卡数不一致。修改：新建通信组后退出前销毁 + 保证卡数一致。
- **结论**：用例/配置问题

#### torch.distributed.all_reduce_coalesced
- **类型**：精度 / 确定性
- **根因**：单算子 HcclAllreduce 场景。
- **结论**：转算子

#### torch.distributed.pipelining.stage.*
- **类型**：功能
- **根因**：test_stage.py 和 test_schedule_multiproc.py 未和开源 PyTorch 测试用例同步。
- **结论**：环境/版本适配（PTA 修复 FrameworkPTAdapter 26.0.0.B082）

#### 9 个通信 + 5 个框架用例未适配 PT2.9
- **类型**：功能
- **根因**：test_checkpoint.py 等文件 2.9.0 后未和开源版本同步。
- **结论**：PTA 版本适配修复

---

### 2.6 边界值 / 其他

#### Tensor.unsqueeze（精度）
- **类型**：精度（边界值/intercept）
- **根因**：8 维输入指定 dim=6 后输出升为 9 维。unsqueeze 是 view 类操作不触发数据搬运，单独执行不报错。但 9 维输出参与后续算子计算时触发维度不超过 8 维拦截校验（ATK 工具中调用 clone() 触发 aclnnInplaceCopy 算子计算前 shape 校验）。
- **结论**：非问题（预期 shape 拦截，用例和工具需适配）

#### Tensor.chunk（精度）
- **类型**：精度
- **根因**：测试 GPU 与 NPU 对比时未加 -sd 命令导致精度异常。添加 -sd 后重新测试 pass。
- **结论**：非问题（测试命令问题）

#### Tensor.reshape（精度）
- **类型**：精度
- **根因**：GPU 与 NPU 输入不一致导致 reshape 精度对比不过。用例增加 -sd 参数同步输入后精度通过。
- **结论**：非问题（输入不同步）

#### Tensor.trace / torch.trace
- **类型**：功能
- **根因**：CANN 9.0.0 B106 已修复，全量通过。
- **结论**：CANN 修复

#### Tensor.scatter_
- **类型**：精度 / 确定性 / 边界值
- **根因**：CANN B060 引入精度问题，B102 版本修复。
- **结论**：CANN 修复（非 PTA 问题）

#### torch.diff
- **类型**：边界值
- **根因**：本地验证直接通过，测试提供的 4 个 failed 用例都能 pass。
- **结论**：非问题（无法复现）

#### quantizeperchannel / quantizepertensor
- **类型**：功能
- **根因**：测试用例版本校验用 `2.1 in torch.__version__` 判断，导致 2.10 版本走错分支。
- **结论**：用例问题

#### scaled_dot_product_attention（入图）
- **类型**：功能（入图报错）
- **根因**：图模式下 seed/offset 在 CPU 初始化导致 2.9 以上版本出现回退无法入图。
- **结论**：环境/版本问题

#### aclnnBinaryCrossEntropyBackward
- **类型**：功能
- **根因**：算子问题。
- **结论**：转算子

#### aclnnBatchMatMul（weightnz 场景）
- **类型**：精度
- **根因**：无详细分析。
- **结论**：待确认

#### torch_npu.npu_fusion_attention_v3（PTA 侧修复）
- **类型**：功能（入图报错）
- **根因**：高版本 torch `build_device_mesh` 接口变化导致 device mesh 被设置到 CPU。
- **结论**：环境/版本问题

#### torch.linalg.matrix_norm / torch.linalg.svdvals
- **类型**：功能/确定性
- **根因**：`_linalg_svd` 适配中涉及 transpose，需要判断 dim 后再 transpose。同结论 [DTS 单号]。
- **结论**：PTA 代码修复

#### 部分 API 入图失败
- **类型**：功能（入图）
- **根因**：部分缺失 trace_rule 已补充。torch.cuda.memory.reset_accumulated_host_memory_stats 等 API 在 2.6.0 不支持入图。
- **结论**：PTA 代码修复
