# CI/CD 流水线逻辑分析报告（Nightly 流水线参考）

> 生成日期：2026-08-25
> 分析范围：`.gitcode/workflows/` 下全部 6 个 YAML 文件

---

## 一、文件总览

`.gitcode/workflows/` 目录下共 **6 个文件**，分为 **2 条主流水线 + 4 个可复用 Job 模板**：

| 文件 | 类型 | 用途 |
|---|---|---|
| `PR-pipeline_op-plugin.yml` | 主流水线 | 生产环境 PR 门禁流水线 |
| `PR-pipeline_op-plugin_test.yml` | 主流水线 | 测试环境 PR 门禁流水线 |
| `build_job.yml` | 可复用 Job | 生产环境编译 Job |
| `build_job_test.yml` | 可复用 Job | 测试环境编译 Job |
| `ut_job.yml` | 可复用 Job | 生产环境 UT Job |
| `ut_job_test.yml` | 可复用 Job | 测试环境 UT Job |

---

## 二、架构设计模式

采用 **「主流水线 + 可复用 Job 模板」** 的分层设计：

```
主流水线（编排层）                    可复用 Job（执行层）
PR-pipeline_op-plugin.yml ────────→ build_job.yml
（生产 PR 门禁）        ────────→ ut_job.yml

PR-pipeline_op-plugin_test.yml ──→ build_job_test.yml
（测试 PR 门禁）                  → ut_job_test.yml
```

**设计思想**：主流水线负责编排 stage 顺序、matrix 策略、条件判断；Job 模板封装具体的编译/测试逻辑，通过 `workflow_call` 被调用。

**Nightly 参考**：此分层结构可直接复用 —— 新建 `nightly-pipeline.yml` 作为主流水线，复用 `build_job.yml` / `ut_job.yml` 作为执行层。

---

## 三、PR-pipeline_op-plugin（生产流水线）详细分析

### 3.1 触发机制

```yaml
on:
  pull_request:
    types: [open, update, reopen]
    branches: ['master', '26.1.0', '26.0.0', '7.3.0']

  pull_request_comment:
    types: [created]
    branches: ['master', '26.1.0', '26.0.0', '7.3.0']
    comments: ['^compile$']
```

- **PR 事件触发**：open / update / reopen 时自动触发，仅限 4 个目标分支
- **评论触发**：在 PR 下评论 `compile` 时手动触发

### 3.2 并发控制

```yaml
concurrency:
  max: 5
  exceed-action: QUEUE
  enable: true
  preemption:
    enable: true
    events: [mr_id]
```

- 最大并发 5，超出排队（QUEUE）
- 开启抢占模式：同一 MR 的新触发会抢占旧的运行

### 3.3 环境变量

```yaml
env:
  repo: ${{ atomgit.event.repository.name }}
  pr_id: ${{ atomgit.event.pull_request.number }}
  org_name: "Ascend"
  # 镜像变量（14个，覆盖 master + v2.7.1 ~ v2.13.0 × arm + x86）
  img_master_arm: "pytorch_master_a2_aarch64_builder:20260704"
  img_2_7_1_arm: "pytorch_2.7.1_a2_aarch64_builder:20260704"
  img_2_9_0_arm: "pytorch_2.9.0_a2_aarch64_builder:20260704"
  img_2_10_0_arm: "pytorch_2.10.0_a2_aarch64_builder:20260704"
  img_2_11_0_arm: "pytorch_2.11.0_a2_aarch64_builder:20260704"
  img_2_12_0_arm: "pytorch_2.12.0_a2_aarch64_builder:20260704"
  img_2_13_0_arm: "pytorch_2.13.0_a2_aarch64_builder:20260709"
  img_master_x86: "pytorch_master_a1_x86_64_builder:20260704"
  img_2_7_1_x86: "pytorch_2.7.1_a1_x86_64_builder:20260704"
  img_2_9_0_x86: "pytorch_2.9.0_a1_x86_64_builder:20260704"
  img_2_10_0_x86: "pytorch_2.10.0_a1_x86_64_builder:20260704"
  img_2_11_0_x86: "pytorch_2.11.0_a1_x86_64_builder:20260704"
  img_2_12_0_x86: "pytorch_2.12.0_a1_x86_64_builder:20260704"
  img_2_13_0_x86: "pytorch_2.13.0_a1_x86_64_builder:20260709"
```

镜像仓库：`swr.cn-north-4.myhuaweicloud.com/frameworkptadapter/`

命名规则：`pytorch_<version>_<chip>_<arch>_builder:<date>`

- 芯片代号：`a1` = x86_64，`a2` = aarch64 (ARM)

### 3.4 Stage 编排

流水线共 3 个 Stage，**Stage 内 Jobs 并行，Stage 间串行**：

```
Stage 1: CodeCheck ──→ Stage 2: Build ──→ Stage 3: UT
```

#### Stage 1 - CodeCheck

| Job | 用途 | 运行环境 |
|---|---|---|
| `CodeCheck_pre_commit` | 代码风格检查 + gitleaks 密钥泄露检测 | `self-hosted, x86, cpu=8` |
| `Detect_file_changes` | 检测变更是否仅为文档，决定是否跳过后续阶段 | `self-hosted, x86, cpu=8` |

**CodeCheck_pre_commit 步骤**：

1. checkout PR 合并代码（`ref: refs/merge-requests/${{ env.pr_id }}/merge`）
2. 安装 Python 3.10（创建 `/python` 符号链接）
3. 从华为 OBS 下载 gitleaks
4. 执行 `openlibing-pre-commit-action`

**Detect_file_changes 步骤**：

1. checkout PR 合并代码（`fetch-depth: 0`）
2. 使用 `paths-changes-filter` Action 检测变更文件
3. 过滤规则：排除以下后缀的文件，若全部是这些文件则 `need_build=no`
   - `.md` `.rst` `.log` `.png` `.jpg` `.doc` `.eddx` `.svg`
4. 将变更文件列表写入 `modify_files.txt` 并上传为 artifact

**输出**：
- `need_build`（yes/no）→ 传给 Stage 2 和 Stage 3 的 `if` 条件
- `modify_files.txt` artifact → 传给 Stage 3 UT 使用

#### Stage 2 - Build

**条件**：`jobs.Detect_file_changes.outputs.need_build == 'yes'`

**Matrix 策略（4 组合）**：

| 索引 | PyTorch 分支 | 架构 | Docker 镜像 |
|---|---|---|---|
| 1 | master | x86 | pytorch_master_a1_x86_64_builder |
| 2 | v2.7.1 | x86 | pytorch_2.7.1_a1_x86_64_builder |
| 3 | master | arm | pytorch_master_a2_aarch64_builder |
| 4 | v2.7.1 | arm | pytorch_2.7.1_a2_aarch64_builder |

**传入参数**：`runs_on_spec: "cpu=32"`

#### Stage 3 - UT

**条件**：`jobs.Detect_file_changes.outputs.need_build == 'yes'`

**Matrix 策略（3 组合）**：

| 索引 | PyTorch 分支 | 镜像 | TEST_RANK | TEST_WORLD_SIZE | 说明 |
|---|---|---|---|---|---|
| 1 | master | aarch64_builder | 1 | 2 | 分布式 part1 |
| 2 | master | aarch64_builder | 2 | 2 | 分布式 part2 |
| 3 | v2.7.1 | aarch64_builder | 默认1 | 默认1 | 单卡 |

**设计意图**：master 版本 UT 用例较多，通过 `RANK/WORLD_SIZE` 拆分为 2 份并行执行，缩减耗时。

---

## 四、build_job.yml（编译 Job 模板）详细分析

### 4.1 输入参数

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `py_version` | 否 | `3.10` | Python 版本 |
| `REMOTE_URL` | 否 | `ATOMGIT_REPOSITORY_URL` | 仓库远程地址 |
| `pr_id` | 否 | PR number | PR 编号 |
| `CP_DOCKER_IMG` | **是** | - | Docker 镜像名 |
| `pytorch_branch` | **是** | - | PyTorch 分支 |
| `runs_on_arch` | 否 | `x64` | 运行架构 |
| `runs_on_spec` | 否 | `cpu=32` | 资源规格 |
| `TARGET_BRANCH` | 否 | PR base ref | 目标分支 |

### 4.2 执行步骤

```
步骤1: print env               → 打印所有输入参数
步骤2: check branch support    → 分支兼容性检查（核心逻辑）
步骤3: [仅master] checkout pytorch → 拉取 Ascend/pytorch 仓库
步骤4: [仅master] install torch   → 从华为 OBS 下载安装 torch whl
步骤5: checkout                → 拉取 PR 合并代码
步骤6: compile                 → bash ci/build.sh --python=3.10 --pytorch=<branch>
步骤7: upload artifact         → 上传 dist/torch_npu*.whl
```

### 4.3 分支兼容性检查（核心逻辑）

```bash
if [ "${TARGET_BRANCH}" = "master" ]; then
    # master 分支：构建所有版本
    need_build=yes
    pytorch_build_branch=${PYTORCH_BRANCH}
else
    # 版本分支只构建其支持的 PyTorch 版本
    support_730="v2.6.0 v2.7.1 v2.8.0 v2.9.0"              # 7.3.0 分支
    support_2600="v2.7.1 v2.8.0 v2.9.0 v2.10.0"            # 26.0.0 分支
    support_2610="v2.7.1 v2.9.0 v2.10.0 v2.11.0 v2.12.0"   # 26.1.0 分支
    # 通过动态变量名间接引用 support_${suffix}
fi
```

**输出**：
- `need_build`：是否需要构建（yes/no）
- `pytorch_build_branch`：实际编译分支名
  - master 对 master：`master`
  - master 对 v2.7.1：`master`（直接用 master 分支编译）
  - 26.1.0 对 v2.7.1：`v2.7.1-26.1.0`（拼接版本分支名）

### 4.4 编译分支名拼接规则

| TARGET_BRANCH | pytorch_branch | pytorch_build_branch |
|---|---|---|
| master | master | `master` |
| master | v2.7.1 | `master` |
| 26.0.0 | v2.7.1 | `v2.7.1-26.0.0` |
| 26.0.0 | v2.9.0 | `v2.9.0-26.0.0` |
| 26.1.0 | v2.7.1 | `v2.7.1-26.1.0` |
| 7.3.0 | v2.6.0 | `v2.6.0-7.3.0` |

### 4.5 Artifact 管理

- 上传路径：`dist/torch_npu*.whl`
- 命名规则：`op-plugin-<pytorch_branch>-<arch>`（如 `op-plugin-master-arm`）
- overwrite: true

---

## 五、ut_job.yml（UT Job 模板）详细分析

### 5.1 输入参数

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `py_version` | 否 | `3.10` | Python 版本 |
| `REMOTE_URL` | 否 | `ATOMGIT_REPOSITORY_URL` | 仓库远程地址 |
| `pr_id` | 否 | PR number | PR 编号 |
| `CP_DOCKER_IMG` | **是** | - | Docker 镜像名 |
| `pytorch_branch` | **是** | - | PyTorch 分支 |
| `TARGET_BRANCH` | 否 | PR base ref | 目标分支 |
| `TEST_RANK` | 否 | - | 测试 rank |
| `TEST_WORLD_SIZE` | 否 | - | 测试 world size |

### 5.2 运行环境

- runs-on: `[self-hosted, npu=0]`（需要 NPU 昇腾芯片）
- 同 build_job 一样包含分支兼容性检查逻辑

### 5.3 执行步骤

```
步骤1:  print env                    → 打印环境变量
步骤2:  check branch support         → 分支兼容性检查（同 build 逻辑）
步骤3:  [仅master] checkout pytorch  → 拉取 Ascend/pytorch
步骤4:  [仅master] install torch     → 从 OBS 安装 torch（硬编码 aarch64）
步骤5:  download artifact            → 下载 op-plugin-<branch>-arm 编译产物
步骤6:  verify + pip install         → 安装 torch_npu*.whl
步骤7:  checkout op-plugin           → 拉取 PR 代码
步骤8:  download modify_files.txt    → 下载变更文件列表
步骤9:  UT_op-plugin                 → 执行 op-plugin 自身 UT
步骤10: checkout pytorch（对应版本）  → 再次拉取 pytorch 含 submodules
步骤11: UT_pytorch                   → 执行 pytorch 仓库 UT
```

### 5.4 UT 详细说明

**UT_op-plugin（步骤 9）**：

```bash
source /etc/profile
export PTA_UT_EXEC_TIMEOUT=4800
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
python3.10 ci/access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}
```

- 超时 4800 秒（80 分钟）
- 需要 CANN 和 ATB 环境（NPU 驱动）

**UT_pytorch（步骤 11）**：

```bash
# 修改 pytorch 内的路径判断逻辑
sed -i "s/'op-plugin' in str(Path(ut_file))/str(NETWORK_OPS_DIR) in str(Path(ut_file))/g" pytorch/ci/split_by_time.py
sed -i "s/'op-plugin' in str(Path(ut_file))/str(NETWORK_OPS_DIR) in str(Path(ut_file))/g" pytorch/ci/access_control_test.py
# 执行 pytorch UT
python3.10 pytorch/ci/access_control_test.py --rank=${TEST_RANK} --world_size=${TEST_WORLD_SIZE}
```

- 修改 pytorch CI 脚本中的路径匹配逻辑，使其能正确识别 op-plugin 目录
- 测试 PyTorch 仓库中依赖 op-plugin 的算子

### 5.5 两个 UT 的区别

| 对比项 | UT_op-plugin | UT_pytorch |
|---|---|---|
| 测试对象 | op-plugin 仓库自身的算子 | PyTorch 仓库中依赖 op-plugin 的算子 |
| 执行脚本 | `ci/access_control_test.py` | `pytorch/ci/access_control_test.py` |
| 源码来源 | 当前 op-plugin 仓库 |Ascend/pytorch 仓库（对应分支） |
| 路径修改 | 不需要 | 需要 sed 修改路径匹配 |

---

## 六、数据流与 Artifact 传递

```
Stage 1: Detect_file_changes
  │
  ├─ output: need_build ──────────→ Stage 2 if 条件
  │                                → Stage 3 if 条件
  │
  └─ upload: modify_files.txt ────→ Stage 3 download（UT 用）

Stage 2: Build
  │
  └─ upload: op-plugin-<branch>-<arch>.whl ──→ Stage 3 download（UT 安装）

Stage 3: UT
  │
  ├─ download: op-plugin-<branch>-arm.whl（从 Build）
  ├─ download: modify_files.txt（从 Stage 1）
  └─ 执行 UT（op-plugin + pytorch）
```

---

## 七、生产与测试流水线差异对比

| 维度 | 生产流水线 | 测试流水线 |
|---|---|---|
| 触发方式 | PR 事件 + `/compile` 评论 | 仅 `/test_compile*` 评论 |
| 目标分支 | `master`/`26.1.0`/`26.0.0`/`7.3.0` | 任意分支 `*` |
| 组织名 | Ascend | ComputingActionTest |
| Build Stage | 活跃（4 组合） | **全部注释** |
| 变更检测方式 | paths-changes-filter Action | Only_Doc_Commit_Check Action |
| 编译脚本 | `ci/build.sh` | `cie/.ci/build/op-plugin_build.sh` + ccache |
| 编译产物 | `.whl` | `.tar.gz` |
| UT 脚本 | `access_control_test.py`（双 UT） | `op-plugin_ut_general.sh`（仅 op-plugin） |
| NPU 需求 | `npu=0` | `npuType=910b2, npuNum=1`（指定型号） |
| 分支兼容性检查 | 有 | 无 |
| PyTorch 版本覆盖 | master + v2.7.1 | v2.7.1 ~ v2.13.0（更广） |

---

## 八、外部依赖汇总

| 依赖 | 来源 | 用途 |
|---|---|---|
| torch whl | `pytorch-package.obs.cn-north-4.myhuaweicloud.com` | master 分支编译/UT 前安装 |
| gitleaks | 同上 OBS | 密钥泄露检测 |
| pytorch 源码 | `gitcode.com/Ascend/pytorch.git` | 编译依赖 + UT 来源 |
| CANN | `/usr/local/Ascend/cann/set_env.sh` | NPU 运行环境 |
| ATB | `/usr/local/Ascend/nnal/atb/set_env.sh` | NPU 运行环境 |
| pre-commit Action | `openlibing-pre-commit-action` | 代码风格检查 |
| paths-changes-filter | Action | 变更文件检测 |
| Only_Doc_Commit_Check | `WWQ129/.gitcode/actions/...@main` | 测试流水线文档检测 |

---

## 九、设计 Nightly 流水线的关键改动

### 9.1 需要改动的项

| 维度 | 当前 PR 流水线 | Nightly 应改为 |
|---|---|---|
| **触发** | `pull_request` + `pull_request_comment` | `schedule` (cron) + `workflow_dispatch` |
| **代码来源** | `refs/merge-requests/<pr_id>/merge` | `refs/heads/<branch>` |
| **变更检测** | 检测文档变更跳过构建 | 可去掉，Nightly 全量构建 |
| **Build Matrix** | 4 组合（master+v2.7.1 × x86+arm） | **14 组合**（全版本 × 全架构） |
| **UT Matrix** | 3 组合（仅 ARM） | 全版本 × 全架构，增加拆分份数 |
| **TARGET_BRANCH** | 从 `pull_request.base.ref` 获取 | 显式传入或从 matrix 定义 |
| **pr_id** | 有（PR 编号） | 无（可用 build_id 或日期替代） |
| **pre-commit/gitleaks** | 有 | 可去掉 |
| **modify_files.txt** | 有（PR 变更文件） | 可能需要替代方案或去掉 |

### 9.2 可直接复用的项

- `build_job.yml` 和 `ut_job.yml` 的步骤逻辑
- 分支兼容性检查逻辑（`check branch support`）
- Docker 镜像定义
- Artifact 命名规则
- 编译脚本 `ci/build.sh`
- UT 脚本 `ci/access_control_test.py`
- CANN / ATB 环境配置

### 9.3 建议的 Nightly 流水线结构

```yaml
name: nightly-pipeline_op-plugin

on:
  schedule:
    - cron: '0 2 * * *'        # 每天凌晨 2 点
  workflow_dispatch:            # 保留手动触发

env:
  # 镜像变量（从 PR 流水线复制，覆盖全版本×全架构）
  ...

stages:
  stage_1:
    name: Build
    jobs:
      Build:
        uses: .gitcode/workflows/build_job.yml
        strategy:
          matrix:
            include:
              # 全量：master ~ v2.13.0 × x86 / arm
              # 参考 PR-pipeline_op-plugin_test.yml 中注释掉的 Build Job
              ...

  stage_2:
    name: UT
    jobs:
      UT:
        uses: .gitcode/workflows/ut_job.yml
        strategy:
          matrix:
            include:
              # 全量版本 × 适当拆分（master 可拆 4 份）
              ...
```

### 9.4 注意事项

1. **ut_job.yml 中 torch 安装硬编码了 aarch64**（第 109 行），如需 x86 UT 需修改
2. **Nightly 没有 PR 事件上下文**，所有 `atomgit.event.pull_request.*` 引用需替换
3. **NPU 资源是瓶颈**，需确保凌晨时段 NPU 资源池可用
4. **镜像版本**建议固定日期（如当前 20260704），避免 nightly 因镜像更新导致不一致
5. **测试流水线中 Build Stage 全部注释**，但 UT 仍依赖 Build 产物，Nightly 必须保留 Build