# Create 模式 Case 索引

> 全新算子接入（`.md` + `_def.cpp` → torch_npu 后端）的案例库。C-confirm 后按算子类型/场景匹配，命中则读全文作决策与避坑参考。

| # | Case | 类型 | 场景 | 影响文件 | 关键词 |
|---|------|------|------|----------|--------|
| 1 | [npu_swi_glu_quant-create](npu_swi_glu_quant-create.md) | workflow-review | SwiGlu+量化融合算子接入 | yaml, cpp, meta, test, doc, json, allowlist | V1 only, 动态/静态量化, MoE, group_index, smooth_scales 不可空 |
| 2 | [fused_infer_attention_score-create](fused_infer_attention_score-create.md) | domain-reference | attention 类多参数融合算子（已对接，领域参考）；**ACLGraph 手工 update 2 接口范式仓内 conformance 实例** | yaml, cpp, meta | layout双变量体系(14值), valueD多路径推导, output dtype 4路, softmax_lse条件输出, aclnn多版本分派, PageAttention, INT4 packed, ACLGraph `.out`/`_get_max_workspace`/SymInt保留/kByte契约 |
| 3 | [npu_moe_gating_top_k-create](npu_moe_gating_top_k-create.md) | domain-reference | **通用陷阱·必读**：任意带「可选/条件输出」(out_flag / output_softmax / 条件 out) 的算子 | cpp, test, (meta) | 可选输出分配 shape 不能跨算子假设, "为空"三义, flag只控写入不控分配, aclnn头required+参数表优先, tiling CheckOutShape, 兄弟算子{0}照搬陷阱 |
| 4 | [fused_infer_attention_score-rebuild](fused_infer_attention_score-rebuild.md) | workflow-review | **全流程实测**：代码删空后仅凭 skill + 输入文件从零重建（V3 单版本 + ACLGraph 2 接口）；流程经验，领域事实见 #2 | yaml, cpp, meta, test, doc, json, build | 绑定权威取 aclnn 签名非 _def.cpp, 高版本 only 参数排除, 假冲突识别, Q1/Q2 正交且辅助接口不注册 meta, SymInt 三条目同步, EXEC 宏在 update_op_api_common.h（"aclnn 名未声明"是级联报错）, 共享 helper, 构建按 PID 轮询, CHECKLIST opName 被脚本消费, 设备不可用不伪造结果 |

> 格式参考：`../../templates/case-template.md`。
