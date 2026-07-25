# Create 模式 Case 索引

> 全新算子接入（`.md` + `_def.cpp` → torch_npu 后端）的案例库。C-confirm 后按算子类型/场景匹配，命中则读全文作决策与避坑参考。

| # | Case | 类型 | 场景 | 影响文件 | 关键词 |
|---|------|------|------|----------|--------|
| 1 | [npu_swi_glu_quant-create](npu_swi_glu_quant-create.md) | workflow-review | SwiGlu+量化融合算子接入 | yaml, cpp, meta, test, doc, json, allowlist | V1 only, 动态/静态量化, MoE, group_index, smooth_scales 不可空 |
| 2 | [fused_infer_attention_score-create](fused_infer_attention_score-create.md) | domain-reference | attention 类多参数融合算子（已对接，领域参考）；**ACLGraph 手工 update 2 接口范式仓内 conformance 实例** | yaml, cpp, meta | layout双变量体系(14值), valueD多路径推导, output dtype 4路, softmax_lse条件输出, aclnn多版本分派, PageAttention, INT4 packed, ACLGraph `.out`/`_get_max_workspace`/SymInt保留/kByte契约 |
| 3 | [npu_moe_gating_top_k-create](npu_moe_gating_top_k-create.md) | domain-reference | **通用陷阱·必读**：任意带「可选/条件输出」(out_flag / output_softmax / 条件 out) 的算子 | cpp, test, (meta) | 可选输出分配 shape 不能跨算子假设, "为空"三义, flag只控写入不控分配, aclnn头required+参数表优先, tiling CheckOutShape, 兄弟算子{0}照搬陷阱 |

> 格式参考：`../../templates/case-template.md`。
