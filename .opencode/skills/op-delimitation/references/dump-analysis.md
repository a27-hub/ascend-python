# 算子级 Dump 与精度定界

> **权威脚本**：`tools/op-delimitation-analyze-precision/scripts/dump_npu_cpu.py`  
> **何时 Read**：xlsx 真精度行 + **组合类 aclnn（>1）**；flow-atk / analyze-precision 两遍规则指向 Dump 时。  
> **纪律**：Mean **仅导航**；终裁 **必须** `.pt` 全量对比（见 [gotchas §2](../knowledge/gotchas.md#2-dump--csv-的-mean-不可作最终裁决)）。

---

## 一、工作流概览

1. **数据准备**：ATK `--save_data` / `-sd` 产物（`.bin` / `.pt`）置于pta 目录子目录。  
2. **脚本定制**：改 `dump_npu_cpu.py` 的 `__main__`——加载路径、API 调用、`requires_grad`、backward 与 grad `.pt`。  
3. **双路径执行**：NPU 与 CPU **各跑一遍** → 两份 CSV + 逐算子 `.pt`。  
4. **分析**：对齐 `op_name`，找 **首个分叉**；`.pt` 上 `torch.allclose` / max diff。  
5. **结单 / 交付**：路径写入 `分析结论.md` + `analyze/phase_dump/`；见 [report §Dump](../report/op-delimitation-report/SKILL.md)。

---

## 二、数据准备

| 用途 | 典型文件名 |
|------|-----------|
| 输入 | `input.bin`、`tensor_input.bin` |
| 反传梯度 | `output_grad_0.pt` 或 `cpu_output_grad_0.pt` / `npu_output_grad_0.pt` |

**前缀约定**（便于区分设备来源）：必要时手动重命名或分目录，避免 NPU/CPU dump 互相覆盖。

---

## 三、脚本定制要点

`TorchDispatchMode` 子类在 `__torch_dispatch__` 中：

- 执行原算子 → 保存 input/output `.pt` → 记录 mean/shape/dtype 到 CSV。  
- NPU 段与 CPU 段使用 **不同 CSV 前缀**（如 `*_ops_npu.csv` / `*_ops_cpu.csv`）。

**必改项**（每 case 不同）：

1. `torch.load(...)` 路径  
2. 目标 API 调用表达式  
3. 是否 `requires_grad_(True)`  
4. backward 时 `grad_tensor` 来源  

**一般不改**：`TorchDump` 类本体。

---

## 四、执行

```bash
cd <pta 目录>/<含数据与定制脚本的子目录>
python ../../../tools/op-delimitation-analyze-precision/scripts/dump_npu_cpu.py \
  2>&1 | tee ../运行日志/dump_<case_id>.log
```

（路径按实际仓库布局调整；原则：**log tee 到 `运行日志/`**。）

**ACLNN 辅助**（可选）：

```bash
export TORCH_NPU_LOGS=op_plugin
```

日志 aclnn 序列 **不替代** Dump 定界，仅作映射参考。正反向采集规范见 [operator-aclnn-analysis.md](operator-aclnn-analysis.md)。

---

## 五、结果分析

1. 打开 `*_ops_npu.csv` 与 `*_ops_cpu.csv`，按 **调用顺序** 对齐 `op_name`。  
2. 对比 input/output 记录的 **Mean** → 定位 **首个显著分叉** 行。  
3. 对该行及 **前驱算子** 的 `.pt` 做：

```python
import torch
a = torch.load("..._input_0.pt", weights_only=False)
b = torch.load("..._input_0.pt", weights_only=False)  # 对侧路径
print(torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3))
print((a.float() - b.float()).abs().max().item())  # 结单必填数值
```

4. **结单数值纪律**：`分析结论.md` 写 **max diff 具体数值**，禁止仅写「差异较大」。

---

## 六、产物存档建议

```text
{opName}_{日期}/
├── analyze/
│   └── phase_dump/
│       ├── *_ops_npu.csv
│       ├── *_ops_cpu.csv
│       ├── README.txt          # 首个分叉 op_name + max diff
│       └── <关键>.pt           # 或仅写绝对路径索引
└── 运行日志/
    └── dump_<case_id>.log
```

---

## 七、第三方 / 转算子交付最小集

| 类别 | 内容 |
|------|------|
| 脚本 | 已按 case 改好的 `dump_npu_cpu.py` 副本或路径 + 修改说明 |
| 表 | 两份 CSV，**标出首个分叉行** |
| 数据 | 分叉算子 input/output `.pt`（及必要前驱） |
| 元数据 | case_id / API 名 / aclnn 名（若有 log 摘录） |

---

## 八、与 analyze-precision 的衔接

- **入口条件**：`运行结果=SUCCESS` 且 `*精度通过=false`（见 analyze-precision §入口）。  
- **单算子（=1 aclnn）**：**跳过 Dump**，走 single-op early-stop。  
- **确定性 Fail**：优先 [determinism_check.py](../tools/op-delimitation-analyze-precision/scripts/determinism_check.py)，非 Dump 首选。
