# AlltoAllv GMM V2 AIV PTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the official op-plugin so `torch_npu.npu_alltoallv_gmm(..., comm_mode="aiv")` eagerly calls the rewritten V2 ACLNN operator, then validate it with ep_test on EP2/EP4/EP8.

**Architecture:** Keep the optional PTA argument backward compatible: `None` dispatches V1 and a supplied mode dispatches V2. Add a focused schema and AIV runtime regression, build a PyTorch 2.9.0/Python 3.11 wheel, then add an isolated `a2avgmm_v2` operation package to ep_test with deterministic CPU reference generation.

**Tech Stack:** C++17, op-plugin codegen YAML, PyTorch/TorchNPU dispatcher, ACLNN, HCCL, Python 3.11, pandas/openpyxl, pytest/unittest.

## Global Constraints

- CANN environment: `/home/sangzhenguo/envs/0813/cann/set_env.sh` (CANN 9.2).
- Python environment: conda `szg_py311`, Python 3.11, PyTorch/TorchNPU 2.9.0.
- Device/operator scope: Ascend 910C, BF16, `comm_mode="aiv"`.
- Preserve `comm_mode=None` V1 behavior.
- Do not add chunk strategies or non-BF16 V2 cases.
- Run and check each batch before proceeding.

---

### Task 1: PTA schema and V2 dispatch regression

**Files:**
- Create: `test/test_custom_ops/test_alltoallv_gmm_aiv.py`
- Modify: `op_plugin/ops/opapi/AllToAllvGmmNpuOpApi.cpp`
- Modify: `codegen/templates/_op_plugin_docs.py`

**Interfaces:**
- Consumes: existing schema ending in `str? comm_mode=None` and V1/V2 ACLNN symbols.
- Produces: public eager call `torch_npu.npu_alltoallv_gmm(..., comm_mode="aiv")` with null-terminated mode forwarding.

- [ ] **Step 1: Write the schema regression test**

Add a test that asserts the installed dispatcher schema contains
`str? comm_mode=None` and accepts the keyword at dispatcher binding time.

- [ ] **Step 2: Verify RED against the current install**

Run:

```bash
python -m unittest test.test_custom_ops.test_alltoallv_gmm_aiv.TestAlltoAllvGmmAiv.test_schema_exposes_comm_mode -v
```

Expected: FAIL because installed `torch_npu 2.9.0.post2` has no `comm_mode`.

- [ ] **Step 3: Harden V2 string forwarding and update API docs**

Materialize `std::string comm_mode_value(comm_mode.value())` and pass
`comm_mode_value.c_str()` to V2. Update the generated API prototype, parameter
description, and example with `comm_mode="aiv"`.

- [ ] **Step 4: Run source checks**

Run `git diff --check`, the repository schema consistency test, and Python syntax
compilation for the new test.

- [ ] **Step 5: Commit**

Commit message: `feat: expose alltoallv gmm v2 aiv PTA path`.

### Task 2: Build and install matching TorchNPU wheel

**Files:**
- Generated: `dist/torch_npu-*.whl`

**Interfaces:**
- Consumes: op-plugin source from Task 1.
- Produces: an installed `torch_npu` in `szg_py311` whose schema includes `comm_mode`.

- [ ] **Step 1: Build against the matching versions**

Run:

```bash
source /home/sangzhenguo/envs/0813/cann/set_env.sh
conda activate szg_py311
bash ci/build.sh --python=3.11 --pytorch=v2.9.0
```

Expected: one Python 3.11 aarch64 wheel under `dist/`.

- [ ] **Step 2: Install without dependency replacement**

Run `pip install --force-reinstall --no-deps dist/torch_npu-*.whl`.

- [ ] **Step 3: Verify GREEN schema and binary symbols**

Run the Task 1 schema test, print `torch.ops.npu.npu_alltoallv_gmm._schemas`, and
check `libtorch_npu.so` for both V1 and V2 ACLNN names.

- [ ] **Step 4: Run two-rank BF16 AIV smoke**

Use `ASCEND_RT_VISIBLE_DEVICES=8,9`; initialize HCCL world size 2, pass
`comm_mode="aiv"`, and compare GMM/shared-MM/permute outputs with the independent
reference.

### Task 3: ep_test operation adapter

**Files:**
- Modify: `/home/sangzhenguo/code/ep_test/__main__.py`
- Modify: `/home/sangzhenguo/code/ep_test/core/launch/__init__.py`
- Modify: `/home/sangzhenguo/code/ep_test/core/launch/launch_op.py`
- Create: `/home/sangzhenguo/code/ep_test/core/launch/a2avgmm_v2.py`
- Create: `/home/sangzhenguo/code/ep_test/core/data/a2avgmm_v2.py`
- Create: `/home/sangzhenguo/code/ep_test/core/models/a2avgmm_v2.py`
- Create: `/home/sangzhenguo/code/ep_test/core/utils/a2avgmm_v2_cases.py`
- Create: `/home/sangzhenguo/code/ep_test/test/test_a2avgmm_v2_cpu.py`

**Interfaces:**
- Consumes: installed PTA call from Task 2 and the existing Case/Runner protocol.
- Produces: operation key `a2avgmm_v2`, deterministic per-rank inputs, CPU golden outputs, and normal ep_test CSV results.

- [ ] **Step 1: Write CPU-only failing tests**

Cover count transpose, expert-major/source-minor permutation, transposed weights,
zero-token experts, shared MM, and validation tolerance. Verify imports fail before
the new modules exist.

- [ ] **Step 2: Implement focused data and operation modules**

Define dataclass Input/Output/Golden/Attach objects, a custom Excel reader, a
deterministic generator, an eager model that always passes `comm_mode="aiv"`, and
an operation validator.

- [ ] **Step 3: Register the operation**

Add `a2avgmm_v2` to the top-level script map and launch registry. Let
`launch_op.py` select `op_class["reader_class"]` when present and otherwise retain
the existing `CaseReader`.

- [ ] **Step 4: Verify GREEN CPU suite and legacy imports**

Run the focused CPU tests, `python -m compileall core`, and import all four
operations. Confirm existing dispatch/combine/both registry entries remain.

- [ ] **Step 5: Commit**

Commit message: `feat: add alltoallv gmm v2 ep_test adapter`.

### Task 4: ep_test cases and runtime verification

**Files:**
- Create: `/home/sangzhenguo/code/ep_test/cases/xlsx/a2avgmm_v2_cases.xlsx`
- Create: `/home/sangzhenguo/code/ep_test/config_a2avgmm_v2_ep2.yaml`
- Create: `/home/sangzhenguo/code/ep_test/config_a2avgmm_v2_ep4.yaml`
- Create: `/home/sangzhenguo/code/ep_test/config_a2avgmm_v2_ep8.yaml`
- Modify: `/home/sangzhenguo/code/ep_test/README.md`

**Interfaces:**
- Consumes: operation key and reader from Task 3.
- Produces: reproducible EP2/EP4/EP8 accuracy commands and result CSVs.

- [ ] **Step 1: Generate and validate the workbook**

Create BF16 cases covering balanced/skewed/zero-token distributions, both weight
layouts, optional shared MM, and optional permute output. Validate required
columns and render/inspect the workbook.

- [ ] **Step 2: Run EP2**

Use physical devices 8-9 and confirm every rank passes GMM, shared-MM when
enabled, and permute when enabled.

- [ ] **Step 3: Run EP4**

Use physical devices 8-11 and repeat the accuracy checks.

- [ ] **Step 4: Run EP8**

Use physical devices 8-15 and repeat the accuracy checks.

- [ ] **Step 5: Run invalid-mode regression**

Call the PTA interface with `comm_mode="ai_cpu"` and confirm the AIV-only V2
returns a nonzero error.

- [ ] **Step 6: Final verification and commit**

Re-run schema, CPU adapter, EP2 smoke, `git diff --check`, and inspect both repo
working trees. Commit message: `test: add alltoallv gmm v2 ep cases`.

