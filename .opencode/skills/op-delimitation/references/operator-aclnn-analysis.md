# ACLNN 算子动态分析（正反向分别 · 单算子 vs 组合）

> **何时 Read**：`analyze-single-op` 两遍规则触发后；或 flow-atk 一屏表「aclnn 计数 / 正反向」卡壳。  
> **与 single-op SKILL 关系**：本节是 **操作规范与输出格式**；计数规则仍以 [analyze-single-op §3.1.1](../tools/op-delimitation-analyze-single-op/SKILL.md) 为执行权威。

---

## 核心原则

1. **以运行时日志为准**：`export TORCH_NPU_LOGS=op_plugin` 后跑 **`atk task -wl '[case_id]'`**，从 tee 提取 aclnn 名。  
2. **禁止**在无日志时写「未找到实现」或凭源码猜 aclnn 列表。  
3. **正反向必须分别分析**——同一 API 正向可能 1 个 aclnn、反向多个（例：`Tensor.div_` 正向 `aclnnInplaceDiv`、反向组合链）。

---

## 单算子 vs 组合类（去重规则）

| 去重后 aclnn **种类数** | 判定 | 后续 |
|------------------------|------|------|
| **= 1** | 单算子（该方向） | 可 early-stop → report「转算子/第三方」 |
| **> 1** | 组合类（该方向） | 精度 → [dump-analysis.md](dump-analysis.md)；功能 → functional |
| **= 0** | 无 aclnn | 纯 Python/CPU 路径或日志未开 op_plugin → 查 env / 用例 |

**去重按 kind，不按调用次数**：日志 `aclnnMul, aclnnMul, aclnnMul` → **1 种** → 单算子。

**反向用例**：日志含 backward 时，先剥离 **正向算子名**，再对 **剩余反向链** 去重计数（与 single-op §3.1.1 一致）。

---

## 正反向分别分析（必做）

| 方向 | 分析对象 | 表格行 |
|------|---------|--------|
| **forward** | API 本体执行时的 aclnn | 每个 API **至少一行**「正向」 |
| **backward** | autograd 反传时的 aclnn | 每个 API **至少一行**「反向」；优化器 `step/zero_grad` 等无反传标 **N/A** |

**典型分裂**：正向单算子 + 反向组合 → **不能**因正向 =1 就把整次分析标「单算子」；结单写清 **正向/反向各自计数**。

---

## 操作步骤（ATK）

```bash
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2
export TORCH_NPU_LOGS=op_plugin
atk task -c <case_json> -n <yaml> -wl '[<case_id>]' 2>&1 | tee 运行日志/single_op_<case_id>.log
```

从 log grep / 人工摘录 **aclnn 执行行** → 去重 → 填入下表。

可选：`bash tools/op-delimitation-analyze-single-op/scripts/extract_aclnn.sh <log>` 得 token 频次表（**启发式**，定案仍以本节去重规则 + xlsx 为准）。

---

## 分析结果表（建议落盘 `analyze/aclnn_<case_id>.md`）

| API 名 | 方向 | case_id | 去重后 aclnn 列表 | 种类数 | 判定 | 日志路径 |
|--------|------|---------|-------------------|--------|------|---------|
| `torch.xxx` | forward | 12 | aclnnFoo | 1 | 单算子 | `运行日志/...` |
| `torch.xxx` | backward | 12 | aclnnBar, aclnnBaz | 2 | 组合类 | 同上 |

---

## 与 PyTorch 算子 / ACLNN 映射

- 日志为 **aclnn 名**；若需 GitCode / wiki 映射，**URL 仅存档**，映射要点 **用户粘贴**。  
- Dump 定界得到的是 **aten 算子序列**；与 aclnn 对照时以用户粘贴的内网映射为准。

---

## 反模式

- **只跑正向就结单「单算子」**（反向未看）。  
- **把调用次数当种类数**（3 次 Mul → 误判组合类）。  
- **批跑 log 混 case** 未 `-wl` 单 case 就计数。
