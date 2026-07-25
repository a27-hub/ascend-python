# Windows 首次装机与日常 SSH 连接

> **入口 skill**：[`tools/manual-connect-npu-server/SKILL.md`](../tools/manual-connect-npu-server/SKILL.md)  
> 认证三档见 [`ssh-auth-policy.md`](ssh-auth-policy.md)。

## 首次装机 — Windows（两步走，仅这一次需要密码）

> **只需替换两处**：`计算机用户名`（Windows 本机用户）与命令里的 `141.61.49.25` / `root` / `22`（即用户给出的服务器 IP / 远端用户 / 端口）。其它字符**逐字保留**。

在 **cmd.exe** 或 **PowerShell** 中执行（Windows 10+ 通常已自带 OpenSSH Client；若未安装，先装 OpenSSH Client，再继续）。

### 步骤 1：生成 ed25519 密钥对

```bat
ssh-keygen -t ed25519 -C "my_key" -f C:\Users\计算机用户名\.ssh\id_ed25519
```

- 提示 passphrase 时**直接回车**（空口令便于脚本化与后续自动化）；如必须设置口令，由用户自定，后续 `ssh` 会多一道本地解锁口令。
- 若 `C:\Users\计算机用户名\.ssh` 不存在，`ssh-keygen` 会自动创建。

### 步骤 2：把公钥追加到远端 `authorized_keys`（**唯一一次需要远端密码**）

```bat
type C:\Users\计算机用户名\.ssh\id_ed25519.pub | ssh -p 22 root@141.61.49.25 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

- 终端会弹出 `root@141.61.49.25's password:` —— **用户在此键入一次远端密码**（用户提供；助手不代输）。
- 命令成功返回 prompt 即装机完成；之后再 `ssh` 不再提示密码。

> **PowerShell 等价写法**（与 `cmd` 二选一即可）：
> ```powershell
> Get-Content C:\Users\计算机用户名\.ssh\id_ed25519.pub | ssh -p 22 root@141.61.49.25 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
> ```

> **若把 IP / 用户 / 端口换成用户提供的真实值**：仅替换 `141.61.49.25` → 用户的服务器 IP，`root` → 用户的远端用户，`22` → 用户的端口；公钥路径里只替换 `计算机用户名`。

### 步骤 3：把私钥路径写入 `remote-config.yaml`

```yaml
ssh:
  host: "141.61.49.25"          # 用户提供的服务器 IP
  port: 22                       # 默认 22；用户另指定时改写
  user: "root"                   # 默认 root；用户另指定时改写
  # Windows 下 OpenSSH 兼容正斜杠，便于跨工具
  key_path: "C:/Users/计算机用户名/.ssh/id_ed25519"
```

> 之后**所有**连接都使用同一密钥。本 skill 不强制写 `~/.ssh/config`，但如用户喜欢，可在该 config 里设 `IdentityFile`，之后命令可省 `-i`。

## 日常连接（永久免密、仅密钥）

```bat
ssh -i C:\Users\计算机用户名\.ssh\id_ed25519 -p 22 root@141.61.49.25
```

或在已写好 `remote-config.yaml` 的项目里以变量形式表达：

```bash
ssh -i <ssh.key_path> -p <ssh.port> <ssh.user>@<ssh.host>
```

若 `Permission denied (publickey)`：回到「首次装机 — 步骤 2」重新把公钥追加到远端 `authorized_keys`，并核对 `ssh.key_path` 与远端文件的权限（`~/.ssh` 700、`authorized_keys` 600）。
