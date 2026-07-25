# 可选/条件输出（out_flag 类）分配契约 —— 通用陷阱参考

**type**: `domain-reference`

> **适用范围（不限某一算子）**：任何算子只要有「可选输出 / 条件输出 / 由 bool flag（`out_flag`、`output_softmax`、`return_xxx` 等）控制的 out 参数」，接入时都应先读本 case。以 `npu_moe_gating_top_k` 触发并记录，但**结论对所有此类算子通用**——具体算子只在末尾"实例"小节出现，正文一律用占位签名。
>
> **一句话核心**：可选输出的**分配 shape 不能跨算子假设**；bool flag 只控制"内核是否写入有效值"，**不控制"是否分配、分配成什么 shape"**。分配契约以**该算子自己的 aclnn 头 required/optional 标记 + 参数表维度**为准，绝不照搬兄弟算子模板、绝不望文生义。

---

## 1. "为空"是歧义词 —— 三种语义，别望文生义

功能说明里常见「若 flag 为 True 则输出 X，**否则为空**」这类措辞。"为空"至少有三种互不相同的实现语义，**望文生义会选错**：

| 读法 | 含义 | 实现 | 何时成立 |
|------|------|------|----------|
| ❌ A. 参数不存在 | 该输出根本不返回 / 传 `None` / undefined tensor | 不 `apply_tensor`，返回 `at::Tensor()` 或 `c10::nullopt` | **几乎不适用于"位置固定的第 N 个返回值"**——返回元组元数固定，不能少一个 |
| ⚠️ B. 空 tensor（zero-sized） | tensor 存在但 shape=`{0}`，`numel()==0` | `apply_tensor_without_format({0}, ...)` | 仅当该算子 aclnn 内核**容忍**秩不匹配的空占位时 |
| ✅ C. 真实 shape、内容无意义 | tensor 按参数表维度**照常分配**，flag=False 时内核不写有效值 | `apply_tensor_without_format({真实维度}, ...)` | 当 aclnn 头把该输出标 **required**、或参数表给了确定维度时 |

**关键澄清（本 case 的核心）**：
- "为空" **首先排除读法 A**——固定位置的输出 tensor **必须分配**、必须存在，不能因 flag=False 就不传/传 None。这是最容易踩的第一层误读。
- "为空"到底是 **B（zero-sized）还是 C（真实 shape 内容无意义）**，**取决于该算子内核契约，不能靠文档措辞猜**。若 aclnn 头标 required 或参数表给了确定维度 → 是 C；只有内核明确容忍空占位 → 才是 B。
- **默认按 C 处理更安全**：内核 tiling 普遍有 `CheckOutShape` 类硬校验，要求 out 与输入同秩；按 B 传 `{0}`（秩变小）会被直接拒。

> **判定规则（可迁移）**：文档功能说明里的「为空 / 不输出 / 可选」是**行为描述**（内核是否写有效值），**不是 shape 声明**。张量的分配 shape/dtype 一律以**参数表维度 + aclnn 头 required/optional 标记**为准。两处冲突时——参数表 / 头文件优先，功能说明的"为空"只理解为"内容无意义"，tensor 照常按真实维度分配。

---

## 2. 判定流程（接入任一 out_flag 类算子时执行）

```
遇到 "若 flag 则输出 X，否则为空" 的输出：
  1. 打开该算子 aclnn 头 (aclnn_xxx.h)，看 X 的 required/optional 标记
     ├─ required        → 读法 C：无条件按真实维度分配，flag 只传给内核控写入
     └─ optional        → 再看参数表
  2. 看 .md 参数表该输出行的「维度(shape)」列
     ├─ 给了确定维度(如"2"、"与x一致")  → 读法 C：按该维度无条件分配
     └─ 明确写"可空/零元素"              → 可能读法 B，需内核确认
  3. 冲突时：aclnn头 required > 参数表维度 > 功能说明"为空"措辞
  4. 拿不准 → 默认读法 C（真实 shape），交给真机 UT 兜底验证
```

**反模式（务必避免）**：参考兄弟/同系列算子，看它对可选输出用 `{0}` 空张量占位就照抄。**同系列、相似签名 ≠ 相同内核约定**——某些内核容忍空占位（读法 B），另一些 tiling 硬校验拒绝（读法 C）。复用模板时，可选输出分配必须回到**当前算子**自己的头 + 参数表重新判定。

---

## 3. 通用代码骨架

占位签名：`op(Tensor x, ..., bool out_flag=False, ...) -> (Tensor main, ..., Tensor out)`，其中 `out` 是受 `out_flag` 控制的条件输出，参数表给出维度 `[D0, D1]`。

**✅ 正确（读法 C，无条件按真实维度分配）**：

```cpp
// out 恒按参数表维度分配；out_flag 只作为标量入参传给内核，控制是否写入有效值。
// NOTE: sync with meta (if graph mode supported). Kernel tiling requires out to keep its
// real rank/shape regardless of out_flag; the flag only gates whether valid data is written.
at::Tensor out = npu_preparation::apply_tensor_without_format(
    {D0, D1}, x.options().dtype(at::kFloat));

EXEC_NPU_CMD(aclnnXxx, x, ..., out_flag, ..., main, out);
```

**❌ 错误（把 flag 语义从"是否写入"外推成"是否分配"）**：

```cpp
at::Tensor out;
if (out_flag) {
    out = npu_preparation::apply_tensor_without_format({D0, D1}, x.options().dtype(at::kFloat));
} else {
    out = npu_preparation::apply_tensor_without_format({0}, x.options().dtype(at::kFloat));  // 秩变小 → 被 tiling 拒
}
```

**图模式 meta 同规则（若算子支持图模式）**：meta 侧也必须无条件按真实维度返回，否则 meta↔C++ 一致写错、图模式 silent 产错误 shape。

```python
# NOTE: sync with C++. out keeps real 2D shape regardless of out_flag.
return (x.new_empty((D0, k), dtype=x.dtype),
        ...,
        x.new_empty((D0, D1), dtype=torch.float32))
```

---

## 4. 快速自检清单

- [ ] 每个受 flag 控制的输出，**都已 `apply_tensor` 分配**（没有传 None / 不返回）——排除读法 A。
- [ ] 分配维度来自**该算子 aclnn 头 required 标记 / 参数表维度**，不是抄兄弟算子。
- [ ] flag=False 分支**没有**把 out 改成 `{0}` 或降秩（除非该内核明确容忍空占位）。
- [ ] flag 只作为标量入参进 `EXEC_NPU_CMD`，不参与 out 的 shape 决策。
- [ ] 若支持图模式：meta 与 C++ 用**同一维度规则**，均无条件按真实 shape。
- [ ] UT 含 flag=True **和** flag=False 两条真机用例。

---

## 5. 实例：npu_moe_gating_top_k（触发本 case 的算子）

> 仅作实例，规则见上文正文。

- 签名：`npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, ..., bool out_flag=False, ...) -> (Tensor y, Tensor expert_idx, Tensor out)`。第三输出 `out`(normOut) 受 `out_flag` 控制。
- .md 两处冲突：
  - **参数表**（权威）：`outOut | 输出 | ... | shape要求与x保持一致 | FLOAT32 | ND | 2` → 无条件 2D `[batch, expert]`。
  - **功能说明 Step 6**：「若 outFlag 为 True，第三个输出为 normOut；**否则为空**」。
- aclnn 头 `aclnn_moe_gating_top_k.h`：`yOut / expertIdxOut / outOut` 三输出**全标 required**。
- 初版误读"为空"为 zero-sized，flag=False 时分配 `{0}`（秩=1）→ 真机报 `norm out shape num 1 and x shape num 2 not equal [FUNC:CheckOutShape][FILE:moe_gating_top_k_tiling.cpp]`、err 561002，多个 UT fail。
- 修复：`out` 恒 `[x_size[0], expert_num]` 分配，`out_flag` 只传给内核控写入。属**读法 C**。
- 图模式：**本算子支持图模式**，但本次 C-confirm 误选"不支持"、跳过 C-meta，故本次错误仅在 C++ 一处；**图模式全链（meta 注册 + 移除 `--disable_torchair` + 图模式测试/文档）待补**，补 meta 时须套用 §3 同规则。
