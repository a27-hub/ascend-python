---
name: op-adaptation
description: >-
  Integrate CANN (Ascend) operators into torch_npu backend. Two modes:
  (1) CREATE — new operators from .md + _def.cpp files;
  (2) MODIFY — existing operators with interface changes, new variants, backward supplements, or bug fixes.
  (2.1) M-adapt — Modify submode: add A5 (Ascend950) platform ACLNN path support to existing operators.
  Use when the user asks to add, adapt, modify, or fix CANN/aclnn operators.
---

# CANN Operator Integration

## Rules

- **project root** = op-plugin repo root. Resolve by walking up from CWD to find a directory named `op-plugin`; that directory is the root. If none is found, **ask the user to confirm the project root before creating any project file** — never silently fall back to the skill directory or CWD.
- **CHECKLIST.md location**: create at `<project root>/CHECKLIST.md` — the project root, NOT any subdirectory. Maintained throughout; update after every step.
- Always generate integration summary at the end.
- **Modify Meta Sync**: In modify mode, when `_out_size` / C++ shape logic changes, meta registration must be checked and updated if needed. Otherwise graph mode produces silent wrong results (see `meta-guide.md` Modifying Existing Meta Registration).
- **Graph Mode Quality Gates** (3-layer defense):
  1. meta-guide.md: Decision Table + Sentinel Patterns + Modification Workflow -> ensures "how to write correctly"
  2. Checklist: Mini Self-Check (5 items) -> ensures "nothing missed after writing"
  3. verify-guide.md: Shape/dtype/constraint cross-verification with Pass/Fail gates -> ensures "correctness verified"
  Modify mode follows the same 3-layer pattern: M-discover -> M-scope -> Meta Sync Decision -> C-verify cross-check
- **Scope Control**: Create mode — only what's in input files (see `parse-guide.md` §Scope Control). Modify mode — only M-scope confirmed files (see `modify-guide.md` §M-scope).
- **Scripts-First Protocol**: 每步若存在确定性脚本（parse / gen-aclgraph / verify / check-constraints / gen-baseline / gen-summary），**先跑脚本拿输出+草稿，再读 guide 补充语义**。禁止"读完 guide 再手写脚本已能生成的内容"。
- **ATK Files Are Subagent-Only**: If the user provides CANN ATK case files
  (4 件套 `atk.yaml` / `generate.py` / `executor.py` / `node.yaml`，或 3 件套
  `case.json` / `executor.py` / `node.yaml`——表示用例已生成） at any point —
  including in the very first message — do NOT Read or parse them in the main
  flow. Record the case-dir path in CHECKLIST.md (§ATK Case Dir) and pass the
  paths verbatim to the op-delimitation subagent at C-delimit. Only the
  subagent reads these files. If no ATK input is provided, C-delimit is
  skipped (not an error) — record the skip in C-summary.

→ Step-specific best practices are in each guide (`yaml-guide.md` §Best Practices, `cpp-guide.md` §Best Practices, etc.).

## Mode Selection

| User provides | Mode | Entry |
|---------------|------|-------|
| `.md` + `_def.cpp` files | **Create** | C-parse |
| Operator name + modification description | **Modify** | M-discover |
| Ambiguous | Ask: "New or modifying existing?" then branch |

**CRITICAL**: After each step, update `<project root>/CHECKLIST.md`. Always generate the integration summary (C-summary). Modify mode does NOT handle gen_opapi structured adapters.

> **📌 Modify submode: M-adapt (A5 platform adaptation)**
>
> When the user says "adapt an existing operator for A5", or M-discover auto-detects M-adapt conditions, **enter the M-adapt submode**:
> - Skip: C-parse / C-confirm / C-derivatives / C-constraints
> - Required: C-yaml / C-cpp / C-meta / C-verify / C-build / C-test / C-docs / C-baseline / C-summary
> - Each step layers §M-adapt rules (see `m-adapt-guide.md`)
>
> **M-adapt routes to one of 4 scenarios** based on `m-adapt-guide.md` §2 Step 0 state classification:
>
> | State | Signal | Scenario | Guide |
> |-------|--------|----------|-------|
> | **A** | YAML has `op_api:` + `opapi/<Op>.cpp` exists | (already adapted — exit) | — |
> | **B** | YAML has `gen_opapi:` sub-block | (gen_opapi handled elsewhere — exit) | — |
> | **C** | YAML only `acl_op:` + `aclops/<Op>.cpp` exists + 4-condition AND gate | Scenario 1 / 2 / 3 | `scenarios/scenario-{1,2,3}-*.md` |
> | **D** | No `aclops/<Op>.cpp` + no `op_api:` | (route to Create mode — exit) | — |
> | **E** | YAML only `op_api:` + no `aclops/<Op>.cpp` (A5-only) | **Scenario 4** (242 candidates in op-plugin) | `scenarios/scenario-4-a5-only.md` |
>
> **C++ implementation patterns**: `references/guides/m-adapt-cpp-patterns-guide.md` (boilerplate / 3-segment skeleton / runtime gating / segment details / backward hard-error template / file naming).
>
> **Full M-adapt orchestration**: `references/guides/m-adapt-guide.md`.

## Execution Order

> **Two orthogonal graph capabilities** (confirmed as two independent questions in C-confirm, see confirm-guide.md §Graph mode): **(Q1) torch.compile 入图** → governs `.default` meta registration (C-meta) + `--disable_torchair` + graph tests/docs; **(Q2) ACLGraph update** → governs the `.out`/`_get_max_workspace` pair（`_infer_output` 不在手工 update 范围）. Neither implies the other — `图模式 YES` below refers to Q1 only; `attention ACLGraph` refers to Q2 only.

### Create Mode
```
1. C-parse        → guides/parse-guide.md
2. C-confirm      → guides/confirm-guide.md
3. C-yaml         → guides/yaml-guide.md                 [conditional: attention ACLGraph → 额外生成 .out/_get_max_workspace 两条目（gen-aclgraph 辅助出草稿）, 见 aclgraph-attention-guide.md + yaml-guide.md §Attention ACLGraph Interfaces]
4. C-derivatives  → guides/derivatives-guide.md        [conditional: autograd; attention ACLGraph → 不为 .out 新增绑定]
5. C-cpp          → guides/cpp-guide.md                [conditional: attention ACLGraph → 额外实现 helper + .out/_get_max_workspace 共 3 个函数, 见 cpp-guide.md §Attention ACLGraph Variants]
6. C-constraints  → guides/cpp-guide.md                [create only]
7. C-meta         → guides/meta-guide.md               [conditional: 图模式 YES → 执行, NO → 跳过; attention ACLGraph 接口例外: 不新增 meta]
8. C-verify       → guides/verify-guide.md             [图模式 YES → include Meta Shape/Dtype cross-verify; attention ACLGraph → 结构检查 + 三参数表核对（verify-guide §ACLGraph）; 辅助接口 meta 缺失属预期, 不报 BLOCK]
9. C-build       → guides/build-guide.md。             [start build; flag conditional: 图模式 YES → remove --disable_torchair]
10. C-test        → op-test skill                      [steps 1-5 parallel with 9; 图模式 YES → add compile test cases]
11. C-docs        → op-apidoc skill                    [parallel with 9; 图模式 YES → add graph mode call example]
12. C-baseline    → guides/compatibility-guide.md      [gen parallel with 9; verify wait for 9]
13. C-delimit     → op-delimitation subagent           [wait for 10; conditional: ATK case dir provided → run; not provided → SKIP and note in C-summary]
14. C-summary     → templates/summary-template.md      [wait for 9+10]
```

### Modify Mode
```
1. M-discover     → guides/modify-guide.md (§ M-discover: grep + status report)
2. M-scope        → guides/modify-guide.md (§ M-scope: derive scope, user confirms)
3. M-gate         → guides/modify-guide.md (§ M-gate: git status check)
4. M-execute      → guides/modify-guide.md (§ M-execute: map files → steps, execute subset)
   └── Always execute: C-verify, C-build, C-summary
   └── Always skip: C-parse, C-confirm, C-constraints
   └── Conditional: C-meta [conditional: 图模式], C-build flag [conditional: 图模式],
       C-test graph cases [conditional: 图模式], C-docs graph example [conditional: 图模式]
   └── C-verify always executes; but Meta Shape/Dtype Logic Verification
       only when _meta_registrations.py or _out_size is in scope
```

## Steps

### C-parse: Parse Inputs

Parse `.md` + `_def.cpp` into the **Parameter Information Table** in `<project root>/CHECKLIST.md`  — the single source of truth for all later steps.

→ `references/guides/parse-guide.md` + `references/templates/checklist-template.md`

> Input scope = `.md` + `_def.cpp` only. If the user also provided a CANN ATK case dir, record its path in CHECKLIST.md §ATK Case Dir without reading the files (see Rules §ATK Files Are Subagent-Only).

### C-confirm: Confirm with User

Resolve all ⚠️ conflict items first (no unilateral decisions). Then standard questions (versions, SoC, .List, autograd, etc.). Present V2 analysis if applicable. Autograd question branches by operator type.

→ `references/guides/confirm-guide.md`

After C-confirm, check `references/examples/README.md` for cases matching the current operator type or scenario (e.g., quantization, autograd). Read matching cases for decision reference and pitfall warnings. If no matches, proceed normally.

### C-yaml: Generate YAML

Register operator(s) in `op_plugin_functions.yaml`. Create: new entry. Modify: edit or append variant. Autograd forward: register BOTH forward + backward entries. Backward/grad: single entry, no `exposed:`.

> **[conditional: attention ACLGraph]** — attention-class dynamic seq-len operators needing ACLGraph manual update support (confirmed in C-confirm Q2（含门禁）): besides `.default`, generate `.out` / `_get_max_workspace` entries（grad 算子按独立逻辑算子重复同一对）。`_infer_output` 不在手工 update 范围，不生成。**先跑 `op_adapt.py gen-aclgraph --op-name <name> --out-names <names>` 出草稿+C++骨架，再人工填版本字段/out 张量命名；`.default` 由 C-parse 参数表手写（脚本不生成）。** See `references/guides/aclgraph-attention-guide.md` and yaml-guide.md §Attention ACLGraph Interfaces.

→ `references/guides/yaml-guide.md`

### C-derivatives: Configure Derivatives (conditional)

Create mode: add new binding. Modify mode: rebind existing formula to new variant. Skip for backward/grad operators.

→ `references/guides/derivatives-guide.md`

### C-cpp: Generate C++

Implement in `op_plugin/ops/opapi/`. Create: new file. Modify: edit existing or add function (e.g., `_out_size` in `KernelNpuOutputSize.cpp`).

> **[conditional: attention ACLGraph]** — implement the 3 ACLGraph functions in order: shared output helper → `.out` 普通路径 → `_get_max_workspace` → `.out` workspace 路径（模板见 cpp-guide.md §Attention ACLGraph Variants；预处理限制转 TORCH_CHECK）。See cpp-guide.md §Attention ACLGraph Variants.

→ `references/guides/cpp-guide.md`

### C-constraints: Constraint Coverage (create only)

Verify every constraint from C-parse is implemented as `TORCH_CHECK` in C++. Run the checker script for a coverage DRAFT, then the LLM verifies it — the keyword matcher is heuristic (false positives + EN/ZH-gap false negatives).

→ `references/guides/cpp-guide.md` (§ Constraint Coverage)

### C-meta: Add Meta Registration [conditional: 图模式 = C-confirm Q1 torch.compile]

**Only execute when torch.compile graph capture (C-confirm Q1) is confirmed.** Skip if not. This is independent of ACLGraph update (Q2): Q2=YES does NOT force meta, and Q1 alone (without Q2) still requires meta.

Register in `_meta_registrations.py`. Meta registration is the **gate for graph mode** (torch.compile / npugraph_ex): no meta → Dynamo cannot infer shapes → graph compilation fails. Wrong meta → graph captures with wrong shapes → silent incorrect results.

> **[conditional: attention ACLGraph] exception**: the gate above applies to `.default`. The ACLGraph manual-update interfaces (`.out` / `_get_max_workspace`) need **no new meta** — they are invoked directly by the torch_npu ACLGraph handler and never go through Dynamo dispatch. Reuse `.default`'s existing meta. See aclgraph-attention-guide.md §Meta 与 Derivatives.

Create: new @impl. For new operators, classify into category via decision tree, match template, implement shape/dtype inference, add `torch._check` constraints, pass 10-item self-check. Modify: use the Modification Decision Table + Sentinel Patterns + 5-step Workflow in meta-guide.md.

→ `references/guides/meta-guide.md` (Modifying Existing Meta Registration for modify mode)

### C-verify: Signature Verification

Cross-check YAML ↔ C++ ↔ Meta ↔ Derivatives. Both modes always execute. Create: all files. Modify: affected files only. Run the verifier script, then the LLM interprets the mismatch report — the type-compat and formula-body checks are heuristic and can false-alarm. **Must produce coverage/reachability reports (see verify-guide). If BLOCK, fix before C-build.**

**When graph mode is supported**: also execute Meta Shape/Dtype Logic Verification (verify-guide.md): shape cross-verification, dtype cross-verification, constraint completeness, output order verification. This catches meta/C++ inconsistencies that parameter-name checks cannot — the #1 cause of graph mode silent errors.

> **[conditional: attention ACLGraph]**: execute verify-guide.md §ACLGraph Structural Verification — 结构检查 + 三条 Host API 参数表人工核对 + 预处理审查核对。Missing meta on `.out` / `_get_max_workspace`（及仓内已有 `_infer_output`）is **expected** — do not BLOCK on it.

→ `references/guides/verify-guide.md` (Meta Shape/Dtype Logic Verification for graph mode)

### C-build: Build

After C-verify is completed, ensure a complete or incremental build is executed. Handle any failures that occur during the build process and do not allow skipping.

> **⚠️ Shared host**: this machine runs many users' builds at once. To check if *your* build finished, capture its PID at launch (`nohup bash ci/build.sh ... & echo $!`) and poll that exact PID with `kill -0 <PID>` — NEVER `pgrep -f ci/build.sh` (matches everyone). Use absolute paths for the wheel/log.

→ `references/guides/build-guide.md`

### C-test: Test  [MUST invoke op-test skill — never hand-write tests]

**REQUIRED: invoke the `op-test` skill via the Skill tool.** Do NOT hand-write test
code inline as a substitute — the skill runs a mandatory coverage matrix + reconcile
gate that manual writing skips.

Deliverables (BOTH required, gate on all):
- Create: new UT file `test/test_custom_ops/test_npu_{op}.py` + FakeTensor class in
  `test/core_tests/test_fake_tensor.py`.
- Modify: append methods to the existing UT file + FakeTensor class.

> **📌 M-adapt test rules**: when adding A5 support to an existing operator, decorate 910A/910B legacy tests with `@SupportedDevices(['Ascend910A', 'Ascend910B'])` and new 950-specific tests with `@SupportedDevices(['Ascend950'])`. See `references/guides/m-adapt-guide.md` §10.4.

> **📌 ACLGraph test rules（conditional: attention ACLGraph）**: op-plugin 侧测试 = functional / `.out(workspace=None)` / `.out(workspace=max_ws)` 三条 eager 路径全量输出对齐（参照既有 `test_npu_fused_infer_attention_score*_with_workspace.py`）。torch_npu 侧 task-group update 测试不在本 skill 交付范围，C-summary 标注移交。

### C-docs: API Documentation

Use `op-apidoc` skill. Create: new doc page. Modify: remind user to manually update existing docs.

> **📌 M-adapt doc rules**: when adding A5 support, the doc product-support table MUST add an "Atlas A5 training series" row; any A5-specific constraints (input size, param limits) MUST be added to the constraints section. See `references/guides/m-adapt-guide.md`.

### C-baseline: Compatibility Baseline

Register the operator in `torch_npu_OpApi_schema_all.json` + `allowlist_for_publicAPI.json`. Run the generator script for entries (func key copied verbatim from YAML), then the LLM verifies — the `op_api:` form and version tags need confirmation.

→ `references/guides/compatibility-guide.md`

### C-delimit: Run ATK Cases & Locate/Delimit (create only, conditional)

**Gate**: check CHECKLIST.md §ATK Case Dir (or the user's messages) for a CANN
ATK case dir.

- **Provided** → invoke the `op-delimitation` subagent via the Agent tool —
  `Task(subagent_type="op-delimitation", prompt="<see below>")` (agent source
  `.claude/agents/op-delimitation.md`) — to execute the ATK cases and
  locate/delimit.

  ```
  Task(subagent_type="op-delimitation", prompt="""
  Handle operator <opName> with op-delimitation:
  CANN ATK case dir <dir> (mode: 4file — contains atk.yaml=<..> generate.py=<..> executor.py=<..> node.yaml=<..>;
                          mode: 3file — contains case.json=<..> executor.py=<..> node.yaml=<..>),
  op-plugin repo root <repo root>.
  """)
  ```

  提供的 ATK 文件路径（4 件套或 3 件套）MUST come from the prompt — **never fabricate**.
- **Not provided** → SKIP this step. Do not block, do not ask repeatedly.
  Record "无 ATK 输入，未执行 ATK 用例验证" in C-summary (summary-template
  §C-delimit + §5 Verification Results ⚠️).

- The main flow must NOT read the ATK files (4 件套或 3 件套) or run ATK itself — reading,
  adapting, and executing all happen inside the subagent's context. The main
  flow only assembles the prompt (paths from CHECKLIST.md §ATK Case Dir or
  the user's message) and launches the subagent.

### C-summary: Integration Summary

Create `notes/summary_{YYYY-MM-DD}.md` in Chinese, append if same date. Run `scripts/op_adapt.py gen-summary --op-name <name> [--parse-json <path>] [--verify-json <path>]` to write the **full template framework** with deterministic parts filled (identity, C-parse tables, C-verify results); if the script warns a block wasn't found (template drift), fill it manually. The LLM completes the semantic sections per `summary-template.md`, including the **test report from C-test** (C-test section). Mark CHECKLIST.md as completed.

If C-delimit was skipped (no ATK input), the summary MUST state "无 ATK 输入，未执行 ATK 用例验证" in the C-delimit section and mark §5 Verification Results with ⚠️ — a skip must never read as a pass.

**Case saving**: ask the user — "Save this integration as a case for future reference?" If yes, create `references/examples/{create-mode|modify-mode|m-adapt}/{operator_name}-{scenario}.md` from `references/templates/case-template.md` and add a row to the **matching mode's sub-index** (`references/examples/{create-mode|modify-mode|m-adapt}/README.md`, not the top-level nav README). Record the case file path in §7.

→ `references/templates/summary-template.md`

## Modify Mode Entry

Execute M-discover → M-scope → M-gate → M-execute in sequence. M-discover runs the discover script for a codebase status report; the LLM derives scope, asks the rebind question, and executes the affected steps.

→ `references/guides/modify-guide.md`

## Common Issues

→ `references/guides/troubleshooting-guide.md` for build, autograd, test, meta registration, and graph mode issues. Each guide also has its own `## Troubleshooting` section for step-specific problems.


