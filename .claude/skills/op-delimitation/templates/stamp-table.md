# 开场契约 · 状态 stamp 模板

> 编排器首轮回复必带。字段含义见 [knowledge/intake-contract.md](../knowledge/intake-contract.md#契约状态-stamp--首轮回复必带)。
> 字段清单与跨字段校验对照见 [intake-contract.md · 字段清单](../knowledge/intake-contract.md#字段清单用户可一次粘齐或多轮散落给)。

```markdown
## 开场契约 · 状态 stamp

**Gate 0 判定**：A · ATK / 极小出口（二选一）
**本pta 目录（{opName}_pta_atk/）**：{opName}_pta_atk/

**已收到 ✓**：CANN ATK 用例目录 / op-plugin 仓根 / pta 目录（{opName}_pta_atk/）

| 待补字段 | 状态 | 备注 |
|---------|------|------|
| 目标 opName | ✗ 阻塞 | CHECKLIST.md 缺 `torch_npu function name:` 行；待用户补 opName（或 auto_adapt.py 提取失败）|
| NPU 设备 id | ◯ 可选 | 默认 0；远端时为远端 NPU id |
| 远端 SSH 字段 | ◯ 窄条件必填 | 仅当 op-plugin 仓 + NPU 同在远端时阻塞；默认本地跳过 |
| xlsx 路径 | ◯ 待 op-atk-adapt 跑完回填 | 按 `atk_output/all_<api_type>_<ts>/` 拼（CaseSet = api_type，不是 opName）|
| 跨字段一致性 | ⚠ opName vs atk.yaml name | 两者不同是正常的（opName 来自 CHECKLIST，api_type 来自 atk.yaml；3 件套模式无 atk.yaml，api_type 取自 executor `@register` / case.json）；结单引用 xlsx 按 api_type 拼 |
```

> **4 标记**：✓ 已收到 ｜ ✗ 缺失·阻塞（不补不进下一阶段） ｜ ◯ 缺失·可选（按下游降级规则继续） ｜ ⚠ 收到但与其它字段冲突（须用户复核）。
