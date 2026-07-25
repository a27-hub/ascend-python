# 反模式（Gotchas · 单点权威）

> **适用**：编排器、flow-atk 及各 `tools/*`。各 skill 正文可保留 **2–3 条本阶段特化** gotcha，完整清单以本节为准。
>
> **编排器入口**：[`skills/op-delimitation-orchestrator/SKILL.md`](../skills/op-delimitation-orchestrator/SKILL.md)

---

## 全局（orchestrator）

> **分析向陷阱**（精度/Dump/双标杆/确定性等）：[`gotchas.md`](gotchas.md)  
> **案例对照索引**：[`history.md`](history.md) → 

1. **跳过 / 绕过 Gate 0**：把 launcher **伪装成** ATK；或编造/拼接**不来源于用户 DTS 正文**的命令以硬套 ATK；不得为套分支捏造 case 编号或 heaven 段落。
2. **分支 A 违反 xlsx 纪律**：滚屏猜 `case_id`、忽视 `report/*.xlsx` 行级判读、混淆 `FAILED` 与 `SUCCESS+精度列=false`、把调度成功当业务全过 — 以 [op-delimitation-flow-atk](../skills/op-delimitation-flow-atk.md) 为准。
3. 在 skill / md 中写入真实密码或完整私钥。
4. 代查或编造 A2/A3 命中行；把 A5 表其它任务的记录混入结论。
5. 臆造 aclnn 名、条数、sheet 名、ATK 参数、`-wl` 下标、失败用例编号。
6. 写入广根目录（`home/` / `data/` / `root/`）而未先向用户确认风险。
7. **凭 URL 自造摘要**：未经用户粘贴原文，仅凭 DTS / heaven / wiki / CodeHub issue URL 归纳要点、步骤、命令、结论；违反 [hard-constraints · 统一口径](hard-constraints.md#统一口径--外部资源获取方式)。
8. **越界改远端**：在远端跑 `pip install` / `conda env create` / `apt` / `yum`、改写 rc / `set_env.sh`、`rm -rf` pta 目录 之外、`docker restart`、升降级 torch_npu 等 —— 全部命中 [hard-constraints · 远端只读边界](hard-constraints.md#远端只读边界)。**正确做法**：写进 `运行记录.md` 的「环境侧待办」，等用户来动手。
9. **末端症状当根因 / 内存踩踏万能归因**：见 `FileNotFoundError: input.bin not exists` / `accuracy dc failed` / `run opp task failed` 就下"celery 竞态 / 内存踩踏"，**未回溯 `atk_output/.../log/atk.log` 更早的 `EZ*` / `Unsupported_Operator` / `build op model failed`**。真实因果链：算子不支持某 dtype → 崩溃 → ATK `cleanup case artifacts` 正常清理（删 input.bin）→ 末端 `FileNotFoundError`。**凡判"内存踩踏"，分析结论 md 必附三项证据**：① 单跑的是失败 case 本身（不是旁证 case）② 该 case atk.log 无更早根因级 op error ③ 确定性 False 的 self dtype 分桶表（False 不聚集某 dtype）。任一缺失 → 不得下内存踩踏，回 3.2 查 atk.log 首个 error。参见 [single-op 证伪门](../tools/op-delimitation-analyze-single-op/SKILL.md#内存踩踏证伪门)、[functional 末端症状回溯](../tools/op-delimitation-analyze-functional/SKILL.md#末端症状回溯)。

---

## SSH 特化（manual-connect）

- **跳过 ① → ②，直接给 `sshpass` / `expect`**：命中 manual-connect §硬禁清单，回到认证策略三档自上而下重做。
