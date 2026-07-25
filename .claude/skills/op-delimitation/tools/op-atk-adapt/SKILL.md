---
name: op-delimitation-op-atk-adapt
description: >-
  CANN ATK → PTA ATK adaptation and execution. Adapts CANN aclnn ATK deliverable
  (4 files: atk.yaml/generate.py/executor.py/node.yaml, or 3 files with
  pre-generated cases: case.json/executor.py/node.yaml — skips `atk case`) into
  PTA ATK case via auto_adapt.py, then runs atk task to produce
  xlsx for downstream analyze-* consumption.
---

# 步骤 2 — CANN ATK → PTA ATK 适配与执行

> **前提**：已通过 [orchestrator](../../skills/op-delimitation-orchestrator/SKILL.md) 的 **Gate 0**（输入 = CANN ATK 用例目录 + op-plugin 仓根）。
>
> 目标：**适配 CANN ATK 用例 → 跑 PTA ATK → 产出 xlsx + tee 日志**，供 3.1/3.2/4 按行判读。
>
> **跑前环境基线**（每次 `atk task` 前必做）：`unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2`。proxy 残留会导致 NPU 直连超时；缺 `ATK_BIND_CPU_TYPE=2` 会 worker 抢核。proxy 仅用于 pip/git clone 等公网下载，不写进 `remote-config.yaml`。
>
> **本地同机约束**：`auto_adapt.py` 要求仓可写 + `atk task` 要 NPU 同机 → 二者必须同机。默认本地；远端仅限「仓+NPU 同在远端同一台机器」。不支持本地建 pta + 远端跑。

## 30 秒速览

| 要做什么 | 要点 |
|---|---|
| 输入 | ① CANN ATK 用例目录（4 件套 `atk.yaml/generate.py/executor.py/node.yaml`，或 3 件套 `case.json/executor.py/node.yaml`——用户已生成用例）；② op-plugin 仓根（含 `CHECKLIST.md` + `codegen/templates/_op_plugin_docs.py`）；③ NPU 设备 id（默认 0）|
| **cwd 钉死** | ③④ 全程 **cwd = `{opName}_pta_atk/`**；否则 `atk_output/` 散到仓根 |
| 第一步 | `python scripts/auto_adapt.py <cann_dir>` — 自动判模式（有 atk.yaml=4file；否则 case json+executor+node=3file）、复制交付件、改 node.yaml |
| 第二步 | `cd {opName}_pta_atk && atk case -f atk.yaml -p generate.py` — 生成 `result/<api_type>.json`。**仅模式 A；模式 B（传入 case.json）跳过本步** |
| 第三步 | 改 executor.py：加 `elif self.device == "npu"` 分支调 `torch_npu.<opName>(...)`；tensor `deepcopy + .to(self.set_device)` |
| 第四步 | `cd {opName}_pta_atk && unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && atk task -c <case json> -p executor.py -n node.yaml [-e 50] 2>&1 \| tee 运行日志/pta_atk_<opName>_<ts>.log`。`-c` 按模式：A = `result/<api_type>.json`；B = auto_adapt.py 输出的 case.json（pta 目录内副本，相对路径） |
| **两套命名** | pta 目录/tee 用 **opName**（`CHECKLIST.md` 的 `torch_npu function name:`）；xlsx 路径 CaseSet 用 **api_type**（模式 A = atk.yaml `name` = `@register` 值；模式 B 无 atk.yaml，取自 executor `@register` / case.json）。结单按 `atk_output/all_<api_type>_<ts>/` 拼 |
| 产物 | `atk_output/.../report/*.xlsx` — 列含 `运行结果`/`失败原因`/`编号`/`*精度通过`/`精度详情`（列结构兼容性见 [接缝论证](reference/seam-argument.md)）|
| -mt/-to | **默认不触发**（本地单设备）；仅 OOM/timeout 证据时调（见 §`-mt`/`-to`）|

## 工作流（4 步）

### 1. Auto-extract + copy + patch node

```bash
python scripts/auto_adapt.py <cann_dir>          # add --dry-run to preview only
```

Automated: read opName from `CHECKLIST.md` → create `{project_root}/{opName}_pta_atk/` and copy the deliverable files → read interface prototype from `codegen/templates/_op_plugin_docs.py` → rewrite `pyaclnn` to `npu` in `node*.yaml`.

脚本自动判输入模式：有 `atk.yaml` → 模式 A（4file，复制 yaml/py）；否则顶层 `*.json` + `executor*.py` + `node*.yaml` → 模式 B（3file，额外复制 case.json 并在 SUMMARY 输出 `case_json` 相对路径）。两者都不满足（如 executor+node 齐但无 json）→ 报错退出；多个顶层 json → 列候选要求指定，不擅自选取。

Emits: PTA dir path, mode, opName, call form, tensor/scalar/optional param classification, executor class（继承 `BaseApi` 的类，改它的 `__call__`）, api_type（`@register` 值）, case_json 路径（模式 B）。

> **降级**：缺 `CHECKLIST.md` → 让用户给 opName；缺 `_op_plugin_docs.py` → 手工从源码抽原型。

### 2. Generate cases

**仅模式 A（4 件套）执行；模式 B（传入 case.json）跳过本步**，case.json 直接用于 §4 的 `-c`。

```bash
cd {opName}_pta_atk
atk case -f atk.yaml -p generate.py
```

### 3. Adapt the executor

两种模式都执行本步（与输入模式无关）。模式 B 下无 atk.yaml，`api_type` 从 executor `@register` / case.json 提取（auto_adapt.py Step 5 已兜底输出）。

**Pre-check:** `if self.device == "npu"` 已存在但只调 `torch.npu.set_compile_mode(jit_compile=False)` → 那是 CANN 编译开关，不是 PTA 路径。需改为调 `torch_npu.<opName>(...)`。

1. 在 `{opName}_pta_atk/executor*.py` 中找**直接继承 `BaseApi` 的类**——这是两种模式统一的定位方式（模式 B 无 atk.yaml 抽取 api_type，不能依赖 `@register` 找类）；auto_adapt.py Step 5 已自动定位并输出该类名。注意：**继承 `AclnnBaseApi` 的类要排除**——那是 CANN aclnn 算子类，不是我们要验证的 PTA 算子。`@register` 值 = `api_type`，仅用于核对与 CaseSet 命名。
2. 在该类的 `__call__` 方法中加 device 分支：

```python
def __init__(self, task_result: TaskResult):
    super().__init__(task_result)
    self.set_device = f"npu:{self.device_id}" if self.device == "npu" else "cpu"

def __call__(self, input_data, with_output=False):
    # ... unpack inputs from input_data.kwargs ...
    if self.device == "cpu":
        # --- original CANN golden logic, unchanged ---
        return golden_func(deepcopy(grad_y).cpu(), ...)

    elif self.device == "npu":
        # --- call torch_npu.<opName>(...) per the extracted prototype ---
        grad_y_npu = deepcopy(grad_y).to(self.set_device)
        weight_npu = deepcopy(weight).to(self.set_device) if weight is not None else None
        return torch_npu.npu_xxx(grad_y_npu, weight=weight_npu, ...)
```

**Rules:** Tensor `deepcopy + .to(self.set_device)`（null-check optional first）；scalar 透传；`api_type`（模式 A 取 yaml；模式 B 取 executor `@register` / case.json）= `@register`（executor）。

### 4. Run

`-c` 路径按模式区分：模式 A = `result/<api_type>.json`（§2 生成）；模式 B = auto_adapt.py SUMMARY 输出的 `case_json`（传入 case.json 在 pta 目录内的副本，相对路径，如 `case.json`）。

**Verify single case:**

```bash
cd {opName}_pta_atk
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && \
atk task -c <case json 按模式> -p <executor>.py -n <node>.yaml -e 1
```

失败时检查 executor 代码（参数名/类型/调用形式 vs 接口原型），修复重试。

**Run all:**

```bash
cd {opName}_pta_atk
unset http_proxy https_proxy && export ATK_BIND_CPU_TYPE=2 && \
OPNAME=$(basename "$PWD" | sed 's/_pta_atk//') && \
atk task -c <case json 按模式> -p <executor>.py -n <node>.yaml \
  2>&1 | tee "运行日志/pta_atk_${OPNAME}_$(date +%Y%m%d_%H%M%S).log"
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| "no execution class" | yaml `api_type` ≠ executor `@register` | Align the two names |
| NPU branch never runs | missing `elif self.device == "npu"` | Add the NPU branch |
| tensor device error | inputs not moved to npu | Add `.to(self.set_device)` to every tensor |
| NPU node still runs aclnn | `node.yaml` `backend` still `pyaclnn` | `auto_adapt.py` patches this — check it wasn't skipped |

## When not to use

Semantic mismatch（aten IR vs aclnn 参数顺序/类型不同）→ 需编辑 yaml `inputs`，超出本简化流程。使用完整的 `cann-atk-to-pta-atk` skill 替代。

---

## 产物定位表

```text
{opName}_pta_atk/                          ← cwd 钉死
├── atk.yaml / generate.py / executor.py / node.yaml
├── result/all_<api_type>_<ts>.json        ← atk case 生成（模式 B 无此目录）
└── atk_output/all_<api_type>_<ts>/        ← CaseSet = api_type（不是 opName）
    ├── log/  mss/
    └── report/all_<api_type>_reports_<ts>.xlsx
```

模式 B（3 件套）目录形态：无 `atk.yaml`/`generate.py`、无 `result/` 子目录；case json 为用户输入的复制件（如 `case.json`，由 auto_adapt.py 复制进 pta 目录）。

| 产物 | 取何列 |
|---|---|
| `report/*.xlsx` | `运行结果` / `失败原因` / `编号` / `*精度通过` / `精度详情` |
| `运行日志/pta_atk_*.log` | `save result excel file:` / `acc_pass_result:` / `atk task success!` |

> `acc_pass_result:Failed` 仅汇总提示，仍需按 xlsx 逐行区分 `FAILED` vs `SUCCESS+*精度通过=false`。`atk task success!` ≠ 业务全过。

---

## 读 xlsx

> **模板**：[`references/atk-xlsx-parsing.md`](../../references/atk-xlsx-parsing.md)（openpyxl/pandas）。`cat`/`grep` 直读 xlsx **物理不可用**。
>
> **⚠ 布尔陷阱**：`openpyxl` 将 `*精度通过` 列读为 Python `bool`。用 `isinstance(val, bool)` 或 `val is False` 判断，**不要** `str(val).lower() == 'false'`。

---

## `--save_data` / `-sd` 产物

```bash
# SKILL_ROOT = op-delimitation skill 根目录绝对路径
export SKILL_ROOT=/path/to/skills/op-delimitation
bash ${SKILL_ROOT}/tools/op-atk-adapt/scripts/save_atk_data.sh \
  <case json 按模式：A=result/<api_type>.json；B=传入的 case.json> node.yaml <case_id> <输出目录>
```

case json 路径本身是脚本入参，天然兼容两种模式。

脚本已内置环境基线。cwd = pta 目录 → `atk_output/` 就地生成，无需 copy。产物：`input/input.bin`（及 `tensor_input.bin`）、`output/output_grad_0.pt`。

---

## ATK 参数纪律

> 命令由 4 步生成（非用户粘贴）。`-c`/`-n`/`-p` 路径保持生成值不动。`-c` 按模式：A = `result/<api_type>.json`；B = 传入的 case.json（auto_adapt.py 输出路径）。

| 可改 | 依据 |
|---|---|
| `-wl '[id,...]'` | case_id 来自 xlsx/tee/用户确认，不得臆造 |
| `-bl '[id,...]'` | 同上 |
| `-mt`/`-to` | 仅 OOM/timeout 证据时（见下） |
| `--save_data`/`-sd`/`-e <N>` | 按需叠 |

**`-bl`/`-wl` 格式**：单引号包裹 JSON 数组 `'[id1,id2]'`；连续区间 `'[[start,end]]'`。DTS 转义残留 `\""[...]"\"` → 改为 `'[...]'`。`TypeError: case_list expected list` → 格式问题。

**执行纪律**：单实例（Flask 8881 端口冲突）；后台监控以 tee 日志为准（`grep "ATK任务进度" <log>`）；确定性测试 5-10 次即可判定。

---

## `-mt`/`-to` 触发证据 + 调优记录

**默认不触发**（本地单设备单 worker）。仅 xlsx 失败原因命中以下时才调：

| 参数 | 触发证据（保留项） | 起跳/阶梯 |
|---|---|---|
| `-mt` | `OOM` / `OOM Killer` / `out of memory` | 从 5 起往下调 |
| `-to` | `timeout` / `Timed out` / `runtime timeout` / `Watchdog`（非 collective）/ 大 tensor 反向耗时 | 120→180→300→600 |

调完写「调优参数记录表」到 `运行记录.md`（追加新行不覆写）：

| 参数 | 默认值 | 调优后 | 调整原因 | 证据来源 |
|---|---|---|---|---|
| `-mt` | （不写） | | | |
| `-to` | （不写） | | | |

---

## 移交清单

- [ ] op-atk 执行记录已落 `运行记录.md`（auto_adapt 输出 / ② atk case 的 result 路径，模式 B 记「跳过（用户提供 case.json）+ case.json 路径」/ executor 改写摘要 / atk task 命令 + tee + xlsx 路径）。
- [ ] `report/*.xlsx` 真实路径已记下。
- [ ] 若叠了 `-mt`/`-to`：证据已截取 + 调优参数记录表已填。
