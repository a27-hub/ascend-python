# SSH 认证策略（三档 + 硬禁清单）

> **入口 skill**：[`tools/manual-connect-npu-server/SKILL.md`](../tools/manual-connect-npu-server/SKILL.md)

## 认证策略（写死三档：① 默认密钥 → ② 装机两步 → ③ 一次性 python 脚本保底）

> 助手**必须**自上而下按 ① → ② → ③ 的顺序尝试；**前一档能跑通就停**，不得跳到后一档；**任何情况下不得**改成下面「硬禁清单」里列出的手段。

### ① 默认（永远先做这一步）：直接 SSH 密钥登录

- 第一次见到这台机器、且 `ssh.key_path` 已写：助手**就直接跑**
  ```bat
  ssh -i <ssh.key_path> -p <ssh.port> <ssh.user>@<ssh.host>
  ```
- 通了 → 进入「连通性自检」。失败 → **降到 ②**。

### ② 联不通时：Windows 两步装机（**全流程中唯一**需要远端密码的时刻）

- 现象：① 报 `Permission denied (publickey)` / 远端 `authorized_keys` 没我们的公钥 / 本机还没生成 `id_ed25519` / 首次见到此服务器；
- 在 **cmd.exe 或 PowerShell** 中执行（Windows 10+ 通常已自带 OpenSSH Client；若未安装，先装 OpenSSH Client，再继续）下文「首次装机 — Windows」**步骤 1 + 步骤 2**；
- 步骤 2 那条 `type ... | ssh ...` 会弹出 `password:` 提示 —— **用户亲手键入一次**远端密码（用户自己提供，助手**绝不代输**）；
- 完成后**回到 ①** 直接 `ssh -i <key> ...`，永久免密。

### ③ 保底（仅当 ② 在当前客户端做不了时启用）：模型按需生成的一次性 python 脚本

适用场景（任一命中即可考虑走 ③）：

- 本机没装 OpenSSH Client 且当前无法安装；
- 企业网/代理拦截 `ssh-keygen` 或拦截 outbound SSH 控制端协议；
- 用户机器是无 `ssh` 的极简环境（容器、瘦终端等），但有 python；
- 用户**显式同意**采用一次性脚本绕过 ②。

边界（**全部要满足**才允许给用户运行）：

- 脚本**在用户本机一次性执行**，进程结束即销毁；
- 密码**只**通过该 python 进程内**交互式输入** / 用户本机环境变量获取，**仅存在内存**；
- 助手**不得**把真实密码写进脚本字面量、不得把真实密码贴回聊天 / 仓库 / md / yaml / `.env` / shell history；
- 该脚本与本 `remote-config.yaml` **解耦**：不读它的字段也不写回；
- 跑通后立即提示用户改装 OpenSSH Client + 走 ②，把 ③ 当作**临时纾困**，不作为长期方案；
- 同一台服务器**最多走一次 ③**：装机完成后必须切回 ①。

> 这一档与下面「硬禁清单」不矛盾：硬禁的是**密码持久化或跨工具暴露**（sshpass 把密码当命令行参数 / `*.env` 落盘 / chat 回贴 / shell history 留痕）；③ 允许的是**密码在用户本机当次 python 进程内中转**，离开该进程就消失。

### 硬禁清单（**任何时候**都不可绕过，违反即视为操作失败）

<a id="硬禁清单任何时候都不可绕过违反即视为操作失败"></a>

无论是 ① / ② / ③ 哪一档，助手**不得**提出、生成或运行下列任一手段：

- `sshpass -p ...` / `sshpass -f <password_file>` —— **最强硬禁**；
- `expect` / `pexpect` 脚本（无论 bash / tcl / python，凡是用 `expect "password:"` send 把明文密码注入 ssh 进程的脚本）；
- `ssh -o PreferredAuthentications=password,keyboard-interactive ...`、`ssh -o PubkeyAuthentication=no ...` 这种**主动切回密码认证**的选项；
- 把远端密码写进 `remote-config.yaml` / `*.md` / `.env` / 任何仓库内文件 / shell history / 终端会话持久层；
- 任何**把真实密码贴回聊天 / 在仓库或本机磁盘明文落盘**的辅助脚本（包括但不限于：bash heredoc 注入、powershell `$env:SSHPASS=...`、python 脚本里硬编 `PASSWORD = "..."` 后回贴用户）；
- **助手生成脚本时**不得在 chat 里复述用户的密码原文，也不得让脚本把密码 echo 到 stdout。

### 出路对照

| 现状 | 下一步 |
|------|--------|
| `ssh.key_path` 已配 + 远端已有公钥 | ① 直接登 |
| 第一次见这台机器 / `Permission denied (publickey)` / 本机已有 OpenSSH Client | ② 两步装机 |
| 本机暂无 OpenSSH Client 且能装上 | 先装 OpenSSH Client → 回 ② |
| 本机暂无 OpenSSH Client 且当下装不上 / 代理拦 ssh-keygen | ③ 一次性 python 脚本（按上面边界） |
| 用户拒绝 ① / ② / ③ 任一条件 | 助手**停下**，告知障碍，等待用户决定；**绝不**改走"硬禁清单"里的手段 |
