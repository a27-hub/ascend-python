# 全局硬约束（单点权威）

> **适用**：`op-delimitation-orchestrator`、`op-delimitation-flow-atk.md`、`tools/*` 下所有会读证据或起命令的 skill。下游正文**短句引用本节**，不重复展开全文。
>
> **编排器入口**：[`skills/op-delimitation-orchestrator/SKILL.md`](../skills/op-delimitation-orchestrator/SKILL.md)

---

## 统一口径 — 外部资源获取方式

1. **以下仍默认「不抓取、不 OCR、不解析」**：内网 wiki / CodeHub issue。这些一律以「用户粘贴的纯文本」为输入。
2. **xlsx 例外（已升级）**：`atk_output/.../report/*.xlsx` 路径由**用户给出**且用户机器可达时，助手**可以**通过 shell 工具在用户机器上跑 [references/atk-xlsx-parsing.md](../references/atk-xlsx-parsing.md)（或 [op-atk-adapt 入口锚点](../tools/op-atk-adapt/SKILL.md#读-xlsx中文表头--中文单元格的解析模板权威单一展开点)）读取（**xlsx 表头与关键单元格为中文**，用 `cat` / `head` / `grep` / `xlsx2csv` 等纯文本/字节流工具直读会乱码或丢字，**物理上不可用**）；解析输出**只保留判读必需字段**并转写到 `运行记录.md` / 分析结论 md；**不**整表 dump 进聊天 / 仓库。
3. 相关字段 — wiki 要点 / CodeHub issue 要点 / A2/A3 命中行 —— **以「用户粘贴的纯文本」为唯一输入**。
4. **URL 仅作存档**：写进 `运行记录.md` 的「外部参考」；**不作为证据来源**；不得从 URL 推断标题、单号、页面结构、步骤要点。
5. **升级路径**：任一能力打通后，**先**更新本节与对应 skill 的「获取方式」列，**再**启用；不得默默接入。已启用的升级：**xlsx 中文表格 → python 解析模板**（见上条 2）。

**外部链接统一写法**：wiki / CodeHub issue / 内网文档 / DTS / heaven 等外链——正文摘要**由用户粘贴**；URL **完整保留**于 md 作为存档。**严禁助手凭 URL 自造摘要或结论。**

---

## 远端只读边界

> **一句话原则**：助手对远端**只在 pta 目录子目录内写自己产物 + 在当前 shell 进程内做临时 env 调整 + 只读地看现有文件**；**任何**触及"删除原有文件 / 改写他人环境 / 持久化环境变更"的动作都**禁止**，即使理由是"为跑通快速补一个依赖"或"清理上次失败的产物"。

### 禁止（命中即视为操作失败，回到本节重判）

1. **删除 / 移动 / 截断 / 覆盖**服务器上**任何与本次任务产物无关的现有文件**。包括但不限于：`/usr/local/Ascend/*`、`/data/*`、`/root/*`、`~/*`、其它用户/任务的目录、容器内系统目录、与本次任务无关的历史 `atk_output/*` 产物。**`rm` / `rm -rf` / `mv` / `> file` / `truncate` 默认禁用**；少数允许的清理见下文「允许」§4。
2. **修改 / 重建 / 删除环境层对象**：
   - `pip install` / `pip uninstall` / `pip install --upgrade` / `pip download → 装` / `pip cache purge`；
   - `conda install` / `conda update` / `conda remove` / `conda env create` / `conda env remove` / `conda clean`；
   - `python -m venv <newenv>` / 删除已有 venv 目录；
   - `apt` / `apt-get` / `yum` / `dnf` / `pacman` / `brew` 等系统包管理；
   - 即使"只是补一个 import 缺的小包"也**不允许**——属于本节灰区，需停下问用户。
3. **改写持久化环境配置**：`~/.bashrc` / `~/.zshrc` / `~/.profile` / `~/.bash_profile` / `/etc/*` / DTS 给的 `set_env.sh` 原文 / 容器 entrypoint / 镜像层 / `~/.condarc` / `~/.pip/pip.conf` / 持久化的 `LD_LIBRARY_PATH` / `PYTHONPATH` 等。
4. **升降级核心组件**：torch / torch_npu / cann / driver / firmware / nccl / hccl / cuda / 容器镜像版本。
5. **重启 / 重建 / 删除 docker/podman 容器**：包括 `docker stop` / `docker rm` / `docker restart` / 重新 `start_train_container.sh`（以"环境刷新"为由），除非用户**显式**指令。
6. **代用户做 git 写操作**：在远端代码仓里 `git checkout` / `git pull` / `git reset` / `git stash` / 切分支 / 改 `git config`（"先把代码同步到 main 试试"也不行）。
7. **代用户清登录痕迹 / 跨账号操作**：`history -c` / 改 `~/.ssh/authorized_keys` 删除其它公钥 / `chown` / `chmod` 不属于本次任务产物的文件。

### 允许（白名单 — 不算"修改服务器环境"）

1. **在 pta 目录子目录内**创建 / 读 / 写 / 追加**助手自己生成的产物**：
   - `{opName}_pta_atk/` 下的 `分析结论.md` / `运行记录.md` / `analyze/*`；
   - `{opName}_pta_atk/运行日志/atk_full_*.log` 等 tee 输出；
   - `{opName}_pta_atk/atk_output/all_*_<ts>/`（ATK 自身落盘的产物目录，由 `atk task` 创建，不是助手手工预先 mkdir 的别人的目录）；
2. **当前 shell 进程内的临时 env 调整**（进程退出即消失，**不写入**任何 rc 文件 / 容器 entrypoint）：
   - `unset http_proxy https_proxy`（[op-atk-adapt · ATK 跑前环境基线](../tools/op-atk-adapt/SKILL.md#atk-跑前环境基线权威本仓库唯一展开点)）；
   - `export ATK_BIND_CPU_TYPE=2`（同上）；
   - `source <用户给的 set_env.sh>`（**只 source，不修改**该文件原文）；
   - 一次性 `export <临时变量>=...` 串到 `&&` 链中；
   - **切换到一个已存在的虚拟环境**：`conda activate <用户给的 env 名>` / `source <用户给的 venv 路径>/bin/activate` / `conda deactivate` / `pyenv shell <版本>`。与 `source set_env.sh` 同档——只改**当前 shell** 用哪个解释器，不落盘、不写 rc。**三个前置全满足才允许**：① 目标 env / venv **已存在**（先 `conda env list` / `ls <venv>` 只读确认）；② env 名 / venv 路径**由用户给出**（不靠猜，遵循上文「统一口径」不做语义推断）；③ 仅当前 shell 生效，**不**用 `conda init` 把它写进 `~/.bashrc` 设为默认（`conda init` 改写 rc，属上文「禁止」§3）。**「选**一个已有环境用 = 允许；**造 / 改 / 删**环境（`python -m venv <新>` / `conda env create` / `pip install` / 删 venv 目录）= 上文「禁止」§2，不变。**
3. **只读检查**：`whoami` / `hostname` / `pwd` / `ls` / `cat` / `head` / `tail` / `grep` / `find -name`（只查不删）/ `du -sh`（看占用） / `df -h`（看磁盘）/ `dmesg`（读 OOM 锚点）/ python 解析 xlsx（[references/atk-xlsx-parsing.md](../references/atk-xlsx-parsing.md)） / `python -c "import torch_npu, ..."` / `pip show torch-npu`（**只查不装**）。
4. **`mkdir -p <本次任务产物子目录>`**：仅限创建本 pta 目录下的子目录（如 `运行日志/` / `analyze/`），**不**对已有目录树动手。
5. **清理助手自己之前生成的产物**——**仅当全部满足**：
   - 用户**显式同意**清理；
   - 待清理路径**位于本pta 目录子目录内**且**完全是助手生成**（`atk_output/all_..._<ts>/` / `运行日志/atk_full_*.log` 等）；
   - 用户**给出具体路径**（不接受"清一下旧的"模糊指令）。
   - 反例：清理别人的任务的 `atk_output/`、清理 `/data/` / `/root/` 下文件、用 `rm -rf` 通配匹配、清理 `~/.cache/pip` —— 全部不允许。

### 灰区一律停手 + 写到 `运行记录.md` 让用户来动手

| 现象 | 错误做法 | 正确做法 |
|------|---------|---------|
| `import xxx` / pytest 找不到包 | 助手 `pip install xxx` | 把缺包名 + 推荐版本写进 `运行记录.md` 的「环境差异」段；停下问用户来装，并复述风险（影响其它任务 / 与 DTS 锁定版本不一致） |
| `torch_npu` 版本与 DTS 期望不一致 | 助手 `pip install torch-npu==<x>` 或卸载重装 | 把当前 `remote_torch_npu_version` 与 DTS 期望值写进 `运行记录.md`，由用户决定是否升降级；助手不动手 |
| 用户已给 env 名 / venv 路径，且该 env **已存在** | 当作"要新建/重装"去 `conda env create` / `pip install` | **直接** `conda activate <env>` / `source <venv>/bin/activate` 切过去即可（见「允许」§2），当前 shell 内激活，记一行到 `运行记录.md`——这是**允许**动作，不必停手问用户 |
| env **不存在** / 名字靠猜 / 需要新建环境 | 助手 `conda env create` / `conda activate <猜名>` / `python -m venv <新>` | 让用户给**准确的已存在 env 名**，或确认是否走容器；助手**不创建、不猜名** |
| 容器没起来 / `docker exec` 失败 | 助手 `docker restart` / 重新 `start_train_container.sh` | 让用户来起容器；助手等容器起来后再 SSH 进入 |
| 磁盘满 / inode 满 / 历史 `atk_output/` 占地大 | 助手 `rm -rf` 清旧产物 | 用 `du -sh` 给出占用排行，写进 `运行记录.md`，由用户决定清哪些 |
| 远端 git 仓库 dirty / 分支不对 | 助手 `git stash` / `git checkout main` | 让用户对齐分支；助手只读 `git status` / `git log` |
| `set_env.sh` 缺一行 / 路径不对 | 助手 `echo >> set_env.sh` 或改写原文 | 把 diff 建议写进 `运行记录.md`，由用户改；助手当次 shell 内可 `export` 临时补，但**不**修改文件 |

> **共同纪律**：灰区情况下，助手输出形如「**环境侧待办**：① 缺包 `xxx` —— 影响 `<具体步骤>`，建议 `pip install xxx==<版本>`，由你来跑；② ……」，而不是直接动手。每条待办都进 `运行记录.md` 留档。

### 失误恢复

如果助手已经误执行了上述禁止动作（典型：误 `pip install` / 误 `rm`），**立刻**：

1. **停下**所有后续动作，不再继续执行 / 分析；
2. 把已执行的命令、影响范围（pip 装了什么 / rm 删了什么）**完整**贴进 `运行记录.md` 的「失误恢复」段；
3. 告诉用户**不可逆的部分**（如装了的包不会自动回滚、删除的文件助手无法恢复）；
4. 等用户决定如何回滚（如 `pip uninstall` 还原、从备份恢复文件）；
5. 本次任务结论中明确写出"环境已被助手误改，结论可信度受影响"，避免把误改后的执行结果当作清白证据。

---

## 边界切换 — 只读分析态 ⇄ op-atk 适配态 ⇄ 本地可写修复态

> **适用**：[`flow-atk`](../skills/op-delimitation-flow-atk.md) 阶段 2 进入 op-atk-adapt 时 / [`flow-fix`](../skills/op-delimitation-flow-fix.md)（分支 C · 修复子流程）启动时本节生效。上文「远端只读边界」是 **状态相关** 的，不是绝对禁令。

### 本地同机约束（硬前提）

`auto_adapt.py` 要求 `project_root`（op-plugin 仓）所在机器**可写**（建 `{opName}_pta_atk/`）；`atk task` 要求 NPU 在同机。二者必须**同机**——否则本地建好 pta 目录后无法同步到远端 NPU 跑，或远端建目录违反远端只读边界。**因此 op-atk 场景默认且几乎必然本地**；远端仅剩「op-plugin 仓 + NPU 同在远端同一台机器」这一窄情形。**不支持**「本地建 pta + 远端跑 NPU」。

### 三个状态

| 状态 | 触发 | 远端只读边界 | 可写范围 |
|---|---|---|---|
| **只读分析态**（默认） | Gate 0 → flow-atk 行级判读 | 完全生效（上文 §禁止/允许/灰区 全部适用） | 仅pta 目录子目录内写产物 |
| **op-atk 适配态**（新增） | flow-atk 阶段 2 进入 op-atk-adapt | 仍生效（远端跑时）；默认本地不触发 | **本地 op-plugin 仓根下 `{opName}_pta_atk/`**（建目录/复制/改 executor/node.yaml；**不碰 op-plugin 源码**）|
| **本地可写修复态** | flow-fix 阶段1 声明 `boundary=READONLY_RELEASED` + `env=LOCAL_DEV_NPU` | **对远端仍生效**；但 flow-fix 操作 **本地 op-plugin 仓**，不碰远端 | 本地 op-plugin 仓可写（改 `op_plugin/ops/opapi/*.cpp`/yaml/meta + 编译）|

> 三态边界清晰：op-atk 适配态与本地可写修复态**同仓**（都写本地 op-plugin 仓根），但 op-atk 只写 `{opName}_pta_atk/` 副本子目录，**不碰源码**；flow-fix 才改源码。不混用。

### 切换规则

1. **切换由 op-delimitation 主导**：在 flow-fix 阶段 1 写入 `运行记录.md` 的「修复子流程 · 边界切换」段，声明 `boundary=READONLY_RELEASED` + `env=LOCAL_DEV_NPU` + 切换时刻 + 确认人。
2. **op-atk 适配态下**，`auto_adapt.py` 在 op-plugin 仓根建 `{opName}_pta_atk/`（本地可写）；**不改原 CANN 用例目录、不改 op-plugin 源码**（仅读 `_op_plugin_docs.py`/`CHECKLIST.md`）。远端只读边界仍生效（不在远端做 pip/rm/git 写操作）。
3. **本地可写修复态下**，上文「§禁止」中关于 **远端** 的禁令**仍禁止在远端执行**；但对 **本地 op-plugin 仓** 的 git 写操作 / 改码 / `pip install` 新编译 wheel / 编译 **允许**。改动**仅限 op-plugin 适配层**（yaml / `opapi/*.cpp` / meta / derivatives / test / schema）；**不动 aclnn/CANN 底层**、**不修精度**。
4. **修复红线与状态无关**：反 CPU 兜底、禁改上游 PyTorch、根因-修复一致性——**三种状态都生效**，见 [flow-fix §阶段 5 修复红线自检](../skills/op-delimitation-flow-fix.md)。
5. **不得借此回远端**：flow-fix 全程远端 **保持只读**。

> 一句话：**远端永远只读；op-atk 适配态的「可写」只针对 `{opName}_pta_atk/` 副本子目录；flow-fix 的「可写」只针对本地 op-plugin 源码**。
