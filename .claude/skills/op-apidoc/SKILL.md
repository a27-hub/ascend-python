---
name: op-apidoc
description: Generate Chinese API reference documentation for torch_npu custom operators, following the standard structure used in docs/zh/custom_APIs/torch_npu/. Use when the user asks to write or generate API docs for a torch_npu operator. Covers product support tables, function signatures, parameter descriptions, return values, constraints, and call examples (single-op and graph mode).
---

# torch_npu API 文档生成

为 torch_npu 自定义算子生成中文 API 参考文档。

## 工作流

1. 读取 `references/template.md` 获取完整文档结构模板
2. 根据算子信息填充模板各章节
3. 输出到 `docs/zh/custom_APIs/torch_npu/torch_npu-{operator_name}.md`

## 写作规范

- 所有正文内容使用中文；产品名称用 `<term>` 标签；数学公式用 LaTeX `$$...$$`
- 无计算公式的算子省略"计算公式"项；无注意事项省略 `> [!NOTE]` 块
- 参数顺序与函数原型一致；不同产品数据类型不同时按产品分条列出
- 单算子示例用 `>>>` 交互式风格并展示真实输出值；图模式用 `torch.compile` + `npu_backend`
- 不支持图模式的算子省略图模式示例
