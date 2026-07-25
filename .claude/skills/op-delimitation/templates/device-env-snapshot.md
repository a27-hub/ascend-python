# 设备环境快照 · 结单必填块

> 复制到 **`分析结论.md`** 或 **`运行记录.md`** 末尾；**5.end 自检**要求字段齐或显式标 `[未测]`。  
> 命令在 **远端或本机** 由用户执行，助手只记录 **用户粘贴的输出**（[hard-constraints · 统一口径](../knowledge/hard-constraints.md)）。
>
> **默认本地**：op-plugin 仓 + NPU 同机时只填「本地环境」段；**仅当仓 + NPU 同在远端**时补填「远端环境」段（见 [hard-constraints · 本地同机约束](../knowledge/hard-constraints.md)）。

## 本地环境（默认必填）

```markdown
## 设备环境快照 · 本地

| 项 | 值 | 采集命令（参考） |
|----|-----|-----------------|
| local_cann_version | | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` 或 `set_env.sh` 路径旁 version |
| torch | | `python -c "import torch; print(torch.__version__)"` |
| local_torch_npu_version | | `python -c "import torch_npu; print(torch_npu.__version__)"` |
| local_torch_npu_dir | | `python -c "import torch_npu; print(torch_npu.__path__)"` |
| local_npu_info | | 用户粘贴 `npu-smi info` 摘要（型号 / 卡数 / 健康）|
| 驱动 / firmware | | 用户粘贴 `npu-smi` 或团队规范命令输出 |
| ATK 版本 | | `atk --version` 或用户粘贴 |
| op_plugin_repo_root | | op-plugin 仓根绝对路径（`auto_adapt.py` 的 `project_root`）|
| op_plugin_git_commit | | `git rev-parse --short HEAD`（在仓根下执行；回归可追溯）|
| 采集时间 | | YYYY-MM-DD |
```

## 远端环境（仅当 op-plugin 仓 + NPU 同在远端时填）

> 触发条件：经 [manual-connect](../tools/manual-connect-npu-server/SKILL.md) 接入远端。默认本地时本段整体标 `[不适用 · 本地同机]`。

```markdown
## 设备环境快照 · 远端（条件性）

| 项 | 值 | 采集命令（参考） |
|----|-----|-----------------|
| remote_torch_npu_version | | 来自 manual-connect 会话 |
| remote_torch_npu_dir | | `remote_torch_npu_dir`（manual-connect 会话）|
| 远端 CANN 版本 | | 用户 `set_env.sh` 路径旁 version |
| 远端驱动 / firmware | | 用户粘贴 `npu-smi` |
| 远端 NPU 型号 / 卡数 | | 用户粘贴 |
| 采集时间 | | YYYY-MM-DD |
```

**纪律**：不得编造版本号；缺项写 `[未测 · 原因]` 并列入「待办」。
