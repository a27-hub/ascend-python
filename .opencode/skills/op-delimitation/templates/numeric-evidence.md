# UT / 确定性 / Dump 数值记录 · 结单必填块

> 走过 **UT / determinism_check / Dump** 任一路径时，`分析结论.md`「定位过程」**必须**含具体数值，禁止「差异大/小」等定性句 alone。

```markdown
## 数值对比记录

| 项目 | 结果 | 备注 |
|------|------|------|
| UT 正向 max diff | | `ut_forward.py` 输出 |
| UT 反向 max diff | | `ut_backward.py` 输出 |
| allclose | PASS / FAIL | rtol/atol 若自定义须写明 |
| 确定性 50 次 | PASS / FAIL | 不一致次数 / 首次分叉轮次 / 首次 Max diff |
| Dump 首个分叉算子 | | op_name |
| Dump max diff | | `.pt` 全量对比数值 |
| 日志路径 | | `运行日志/...` 或 `analyze/...` |
```

**脚本路径**：

- UT：`tools/op-delimitation-analyze-precision/scripts/ut_forward.py` / `ut_backward.py`  
- 确定性：`tools/op-delimitation-analyze-precision/scripts/determinism_check.py`  
- Dump：见 [`references/dump-analysis.md`](../references/dump-analysis.md)
