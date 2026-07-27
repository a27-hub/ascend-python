---
name: manual-connect-npu-server
description: >-
  Use when an op-delimitation ticket needs remote NPU server access, SSH
  key-based connection setup, optional existing container/env entry, read-only
  environment inspection, and remote_torch_npu metadata capture.
---

# 远端 SSH 接入（默认密钥；ATK 通用）

> 本 skill 做两件事：
> ① 把用户从 **Windows + OpenSSH** 经**默认 ed25519 密钥认证**送到远端一个可用 shell（或可选进容器 shell）；密钥不通时走两步装机；两步装机也做不了时走一次性 python 脚本兜底。**`sshpass` / `expect` / `pexpect` / `-o PreferredAuthentications=password` 等任何把密码持久化或跨工具暴露的做法被硬禁**（详见「认证策略 → 硬禁清单」）。
> ② 登录后立即核对远端 `torch_npu` 的安装位置与版本，写入会话上下文，供下游分析 skill 使用。
>
> 其它 ATK / tee / 同步 / 编译均不在本 skill 范围：
> - [op-atk-adapt](../op-atk-adapt/SKILL.md)：ATK 命令与日志落盘。

## 认证策略（权威外置 · 三档自上而下）

<a id="认证策略写死三档①-默认密钥--②-装机两步--③-一次性-python-脚本保底"></a>

> **全文**：[`references/ssh-auth-policy.md`](../../references/ssh-auth-policy.md) — ① 直接 `ssh -i <key>` → ② Windows 两步装机 → ③ 一次性 python 脚本；含**硬禁清单**（`sshpass`/`expect`/密码写盘等，详见 [`§硬禁清单`](../../references/ssh-auth-policy.md#硬禁清单任何时候都不可绕过违反即视为操作失败)）与出路对照表。
>
> **本 skill 仍执行的摘要**：永远先做 ①；`Permission denied (publickey)` → ②（用户手输一次密码）；② 做不了且用户同意 → ③；**任何情况禁止 sshpass**。

## 用户必须先告知（4 项输入）

> 这 4 项**全部由用户给出**；助手不从 DTS URL、历史 chat 或截图推断。

| 输入 | 示例 | 何处用到 |
|------|------|---------|
| 服务器 IP | `141.61.49.25` | 替换命令里的 `<SSH_HOST>`，写入 `remote-config.yaml: ssh.host` |
| 远端用户名 | `root` | 替换命令里的 `<SSH_USER>`，写入 `ssh.user`；本 skill 默认 `root` |
| 远端密码 | （用户口头/会话告知） | **只在「首次装机 — 步骤 2」一次性手动键入**；不入仓库、不入聊天持久层 |
| Windows 计算机用户名 | `Adelaide` | 替换命令里 `C:\Users\<WINDOWS_USER>\.ssh\id_ed25519` 的目录段 |

> SSH 端口若用户未声明，默认 `22`；如自建端口，请用户显式给出 `<SSH_PORT>`。

## 何时使用

- 用户需要通过 SSH 登录远端执行 ATK；
- 需要配置 / 核对 `config/remote-config.yaml` 的 `ssh.*` 与 `container.*`；
- **Windows 上首次生成 ed25519 + 写入远端 `authorized_keys` + 填 `ssh.key_path`**。

## 不在本 skill 范围

- `remote.workspace` / `remote.scope` / `environment.setup_cmd` 等业务字段 → 仅在远端接入场景中约定含义；具体 ATK 适配+执行交给 op-atk-adapt/analyze-*。
- tmux / 会话保活 / rsync / 自动编译。
- **非 Windows 客户端**：macOS/Linux 上等价命令为 `ssh-keygen` + `cat ...pub | ssh ...`，路径换 `~/.ssh/id_ed25519`；本 skill 不以 macOS 为主叙述。

## 配置文件

读取（与本 `SKILL.md` 同级）：`config/remote-config.yaml`。

**本 skill 直接使用**：

| 字段 | 用途 |
|------|------|
| `ssh.host` / `ssh.port` / `ssh.user` | 目标机与登录用户 |
| `ssh.key_path` | **必填**：本机 **ed25519 私钥**路径，Windows 推荐 `C:/Users/<WINDOWS_USER>/.ssh/id_ed25519` |
| `container.use_container` / `container.runtime` / `container.name` | 若 `true`，可在 SSH 后一次性进容器 shell |
| `conda.init_script` / `conda.env_name` | 不使用容器时的 conda 激活前缀（仅在接入层一笔带过） |
| `local.torch_npu_dir`（**可选**） | 本机 torch_npu 源码路径；下游分析可做跨版本对照；未填时下游仅依赖远端 site-packages |

**不由本 skill 展开**：`remote.workspace` / `remote.scope` / `environment.setup_cmd` / `sync.*` / `commands.*`。

**`ssh.key_path` 未填或私钥文件不存在** → 必须先完成下文「首次装机 — Windows」两步，再写 `remote-config.yaml`，再继续自检。

## 首次装机 + 日常连接（权威外置）

> **全文**：[`references/ssh-windows-setup.md`](../../references/ssh-windows-setup.md) — `ssh-keygen`、公钥 pipe、remote-config 填写、日常 `ssh -i` 命令。
>
> **`ssh.key_path` 未填或私钥不存在** → 必须先完成装机再写 `remote-config.yaml`。

## 连通性自检（接入后立即执行）

```bat
ssh -i C:\Users\计算机用户名\.ssh\id_ed25519 -p 22 root@141.61.49.25 "whoami && hostname && pwd"
```

期望输出包含：`root` / 远端 hostname / 远端默认 home 目录（如 `/root`）。任何一项缺失 → 视为接入失败，回查密钥与 `authorized_keys` 权限。

## 可选：一次性进入容器 shell

当 `container.use_container: true` 且 `container.runtime` / `container.name` 已填：

```bash
ssh -i <ssh.key_path> -p <ssh.port> <ssh.user>@<ssh.host> \
  "<CONTAINER_RUNTIME> exec -it <CONTAINER_NAME> bash -lc 'whoami && hostname'"
```

> 需要**持续**容器 shell 调试时，让用户在已 SSH 进去的会话里自行 `docker exec` / `podman exec`；本 skill **不规定**保活方案。

## DTS → remote-config 映射（权威外置）



## torch_npu 位置与版本核对（接入后立即执行）

登录成功（需要时先进入容器）后，**立刻**拿到远端真实的 `torch_npu` 安装位置和版本，记入会话上下文 —— 下游 [op-atk-adapt](../op-atk-adapt/SKILL.md) 会复用这两个值做源码对照、报错定位。

主方法：

```bash
python -c "import torch_npu, os; print('version:', torch_npu.__version__); print('file:', torch_npu.__file__); print('dir:', os.path.dirname(torch_npu.__file__))"
```

兜底（python 不便用时）：

```bash
pip show torch-npu | grep -E 'Version|Location'
```

会话上下文字段（**只存会话内，不写入仓库版 yaml / md**）：

| 字段 | 来源 | 下游用途 |
|------|------|---------|
| `remote_torch_npu_version` | `torch_npu.__version__` | 环境对齐、A2/A3/A5 检索词 |
| `remote_torch_npu_dir` | `os.path.dirname(torch_npu.__file__)` | 下游在远端 `grep -R <报错片段>` 定位报错源文件 |
| `local.torch_npu_dir`（可选） | `config/remote-config.yaml` | 本机 grep / 跨版本对照；**未填时下游不得阻塞** |

判定规则：

- 仅远端拿到 → 下游分析只在远端 grep 源码。
- 远端 + 本地都有 → 下游可做跨机对照（版本差异、分支差异）。
- 两者都拿不到（即 `import torch_npu` 失败）→ **不进入下游分析**；先回本 skill 的「DTS → remote-config 映射」补齐 `environment.setup_cmd` 并按 DTS 中的 `set_env` 再试。

## 核心原则

1. **不写死**真实 IP / 用户名 / 密钥**正文**于本 Markdown 或 yaml 仓库版本；**只写占位与用户本机路径形态**。
2. **远端密码**仅出现在「首次装机 — 步骤 2」那一条 `ssh` 的交互提示里、或「认证策略 ③」一次性 python 脚本的本机内存中，由用户键入；**绝不写入** `remote-config.yaml` / `*.md` / `*.env` / 仓库 / shell history / 聊天回显。
3. **日常一律走密钥**（认证策略 ①）：`ssh -i <ed25519 私钥> ...`；torch_npu 位置/版本由 `python -c "import torch_npu, ..."` 主取、`pip show torch-npu` 兜底。
4. **认证三档严格自上而下**：① 直接密钥 ssh → ② 联不通 → 走 Windows 两步装机 → ③ 仅当 ② 当前客户端做不了 → 模型按需生成的一次性 python 脚本（密码仅在用户本机当次 python 进程内中转，**不**回贴聊天、**不**进任何持久层；详见「认证策略 ③」边界）。**任何情况下**都不得改走"硬禁清单"里的 `sshpass` / `expect` / `pexpect` / `-o PreferredAuthentications=password` / 把密码写进 `remote-config.yaml` 等做法。
5. 超出「密钥登录 + 环境映射 + torch_npu 核对」范围 → ATK 分支交给 `op-atk-adapt`。
6. **登录后只读边界（全项目硬约束）**：登录成功后，助手对远端**只**做：① **读**（`whoami` / `hostname` / `pwd` / `ls` / `cat` / `grep` / 读 xlsx + log / `import torch_npu` 取 `__file__` / `pip show torch-npu` 仅查不装）；② **当前 shell 进程内**临时 env 调整（`unset http_proxy https_proxy` / `export ATK_BIND_CPU_TYPE=2` / `source <用户给的 set_env.sh>`，进程退出即消失，**不写**任何 rc 文件 / entrypoint），**含切换到一个已存在的用户给定虚拟环境**（`conda activate <用户给的 env>` / `source <用户给的 venv>/bin/activate`，等同选解释器、不落盘）；③ 在**已与用户约定的pta 目录子目录**下 `mkdir -p` / 写产物（`运行日志/` / `analyze/` 等）。**禁止**：`pip install` / `pip uninstall` / `conda install/update/remove` / `conda env create/remove` / `python -m venv <新>` / 删 venv 目录 / `conda init`（写 rc） / `apt` / `yum` 等**造改删环境**动作；改写 `~/.bashrc` / `~/.zshrc` / `~/.profile` / `set_env.sh` / 容器 entrypoint；`rm` / `mv` / 截断pta 目录 之外的任何文件（包括别人的任务的 `atk_output/` 历史产物）；重启 / 重建 / 删除容器；升降级 torch / torch_npu / cann / driver。**一句话：选**一个已有环境用 = 允许，**造/改/删**环境 = 禁止。**灰区一律停手**（缺包 / 版本不对 / env 错 / 容器没起来 / 磁盘满），把"环境侧待办"写进 `运行记录.md`，由用户来动手。完整白名单 / 黑名单 / 灰区表见 [hard-constraints · 远端只读边界](../../knowledge/hard-constraints.md#远端只读边界)。
