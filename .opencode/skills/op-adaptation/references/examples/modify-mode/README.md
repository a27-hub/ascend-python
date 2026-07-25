# Modify 模式 Case 索引

> 已有算子的修改/重构（算子名 + 修改描述）的案例库。M-discover 后按同类改动类型或受影响文件匹配，命中则向用户呈现该 case 的关键决策与陷阱作参考。

> **注**：A5 平台适配（M-adapt 子模式）的 case 单独归档在 [`../m-adapt/`](../m-adapt/README.md)，含专用选型指南。

| # | Case | 类型 | 场景 | 影响文件 | 关键词 |
|---|------|------|------|----------|--------|
| 1 | [npu_moe_token_permute_grad_v2-refactor](npu_moe_token_permute_grad_v2-refactor.md) | workflow-review | 接口重构 + derivatives 换绑 | yaml, derivatives, output_size, meta, test, json | backward, scalar params, sym_size, rebind, coexistence |

> 格式参考：`../../templates/case-template.md`。
