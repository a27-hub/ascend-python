# Build Guide

> Build execution guide. Used after verify is complete to decide and run a full / incremental build, and to handle build failures.

## When to Run

After C-verify is complete and all CHECKLIST files have been modified, enter the C-build phase.

## Pre-checks

**Before starting, ask**: "Build in the current environment?"

- If yes: continue
- If no: continue after the user provides the environment activation command

**Additional confirmations**:

- Free disk ≥ 10G (full build includes pytorch clone + build artifacts)
- The `--pytorch` argument matches the current branch of `build/pytorch` (a mismatch triggers `git clean -fdx`, see the full-build warning)

**⚠️ cgroup / MAX_JOBS (mandatory pre-check)** — easy to get wrong, hard to recover from

`setup.py build` (line 369) reads `MAX_JOBS` from env, falling back to `multiprocessing.cpu_count()`. On a 128-core box inside a 32G cgroup (typical container), the default **silently OOM-kills `cc1plus` ~50+ times in a tight loop** and the build either hangs or exits deceptively. The error is `c++: fatal error: Killed signal terminated program cc1plus` — looks like a compiler bug, actually memory pressure.

Pre-check + mitigation, in order:

1. **Detect cgroup limit**:
   ```bash
   cat /sys/fs/cgroup/memory.max 2>/dev/null \
     || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
     || echo "no cgroup"
   ```
2. **Detect CPUs**: `python3 -c "import os; print(len(os.sched_getaffinity(0)))"` (bare `nproc` is wrong when `OMP_NUM_THREADS` is set).
3. **Set `MAX_JOBS` before invoking build.sh** (env passes through to `setup.py build`):
   - Conservative default: `export MAX_JOBS=4` (each `cc1plus` peaks at ~3-5 GB for typical op-plugin TUs).
   - Raise cautiously if memory headroom allows (e.g. 64G cgroup → `MAX_JOBS=8`).
4. **Pass `--jobs N` only if supported** by the underlying `build_ext`. `ci/build.sh` does NOT currently forward `-j`; use the env var.
5. **If you see `Killed signal terminated program cc1plus`** during build: STOP. Do not let it retry. Lower `MAX_JOBS` and restart (the previously-cloned `build/pytorch` is reusable; only `setup.py build bdist_wheel` re-runs).

Heuristic: `MAX_JOBS = min(nproc, cgroup_mem_GB / 4)`.

**`--pytorch` version self-check (mandatory)**

`--pytorch` is the branch/tag of ascend/pytorch (the torch_npu source). The torch version part is independent of the op-plugin branch, **but the branch name usually carries a release suffix that tracks the op-plugin/CANN release line** — the naming scheme is `v<torch_ver>-<release>` (e.g. `v2.10.0-26.0.0`), and a bare `v<torch_ver>` (e.g. `v2.10.0`) frequently **also exists as a different branch**. Passing the bare form checks out the wrong tree and fails during PyTorch codegen.

Derive both parts — never hardcode, and never assume the bare `v<torch_ver>`:

```bash
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")   # e.g. 2.10.0
RELEASE=$(git -C <op-plugin-root> rev-parse --abbrev-ref HEAD)                  # op-plugin branch, e.g. 26.0.0
PYTORCH_TAG="v${TORCH_VER}-${RELEASE}"                                          # e.g. v2.10.0-26.0.0
# Verify the tag actually exists on ascend/pytorch BEFORE building:
git ls-remote https://gitcode.com/ascend/pytorch.git "${PYTORCH_TAG}" | grep -q . \
  && echo "OK: --pytorch=${PYTORCH_TAG}" \
  || echo "MISSING — fall back to v${TORCH_VER} only if ls-remote confirms it exists"
```

- **Never use `--pytorch=master`**: master is a rolling branch with an unstable version; use it only when the user explicitly asks to track mainline, and double-check version consistency.
- **Never assume `--pytorch=v${TORCH_VER}` (bare)**: on suffixed release lines this points to the wrong branch. Prefer the suffixed `v${TORCH_VER}-${RELEASE}`, confirmed via `git ls-remote`.
- `ci/build.sh` defaults to `master`, so you must explicitly pass the derived, verified `--pytorch` value.
- Write the fully derived command into CHECKLIST and copy it verbatim during C-build.
- Also record `torch_npu.__version__` (`python -c "import torch_npu; print(torch_npu.__version__)"`) for C-baseline version-tag reference.

---

## Build Type Detection

### Step 1: Get the list of changed files

```bash
git diff --name-only <base-commit>  # or git status --short
```

### Step 2: Decide build type from file paths

| Changed file | Build type | Reason |
|--------------|-----------|--------|
| `op_plugin/config/*.yaml` (the 4 codegen inputs) | **Full** | yaml → gencode → Functions.h → 350+ cpp recompiled |
| `op_plugin/OpApiInterface.h`, `op_api_common.h` | **Full** | Common header included by 300+ cpp |
| `ops-transformer/common/include/**/*.h` | **Full** | Operator-side common header, all op_host depend on it |
| `torch_npu/csrc/aten/npu_native_functions.yaml` | **Full** | torchnpugen input, triggers full backend stubs regeneration |
| `op_plugin/ops/opapi/*.cpp` added/removed | **Incremental** | `ci/build.sh` re-copies the source tree and re-runs CMake configure every time, so GLOB rescans automatically — no manual reconfigure needed |
| `ops-transformer/<op>/` new operator directory | **Incremental** | Same: `ci/build.sh`'s re-configure picks up the new subdirectory |
| Existing `op_plugin/ops/opapi/*.cpp` modified | **Incremental** | Single cpp recompiled + libtorch_npu relinked |
| `ops-transformer/<op>/op_host/*.cpp` modified | **Incremental** | Single operator recompiled |
| `op_plugin/python/meta/_meta_registrations.py` | **No compile** | Pure Python, takes effect on process restart |
| `torch_npu/csrc/aten/ops/*.cpp` | **Incremental** | Single translation unit |

### Step 3: Decision tree

```
Do the changes include any full-build trigger file (config/*.yaml, common headers, native_functions.yaml)?
├── Yes → Full build     (rm -rf build && ci/build.sh)
└── No  → Are the changes pure Python (_meta_registrations.py, docstrings)?
    ├── Yes → No compile  (restart the process; the .py change takes effect on import)
    └── No  → Incremental (keep build/, re-run ci/build.sh — covers both "cpp modified" and "cpp added/removed")
```

> **Always drive both build types through `ci/build.sh`** — never hand-run `cp -rf op_plugin && cmake --build`. See § Common Pitfalls #4: bypassing `ci/build.sh` skips gencode, which desyncs the dvm (`lazy_fusion::*`) call sites and produces `undefined reference` at link time.

---

## Build Execution

### Full build

```bash
rm -rf build                          # After deletion, ci/build.sh re-clones pytorch (a few GB over the network)
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
RELEASE=$(git rev-parse --abbrev-ref HEAD)
bash ci/build.sh --python=3.10 --pytorch=v${TORCH_VER}-${RELEASE} 2>&1 | tee build.log
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

### Incremental build

Same command as full, but **do not delete `build/`**. Because `build/pytorch` already exists, `ci/build.sh` skips the git clone (saving the multi-GB download); it still re-copies the source tree, re-runs gencode, and lets ccache + ninja recompile only what actually changed (ccache keys on file **content**, not mtime, so the internal `cp -rf` does not invalidate the cache). This one path covers every incremental case — cpp modified, cpp added/removed, new operator directory.

```bash
# Keep build/ — do NOT rm it.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
RELEASE=$(git rev-parse --abbrev-ref HEAD)
bash ci/build.sh --python=3.10 --pytorch=v${TORCH_VER}-${RELEASE} 2>&1 | tee build.log
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

**⚠️ `--pytorch` must match the branch already checked out in `build/pytorch`.** Derive it from the local `torch.__version__` (never hardcode / never `master`) — the same value as the full build. If it differs, `ci/build.sh` runs `git clean -fdx` inside `build/pytorch`, wiping your incremental artifacts and silently degrading to a full rebuild (and deleting other untracked files there). The top-level `op-plugin` source is a separate repo and is unaffected.

---

## Handling Build Failures

### Principle

**Do not skip.** A build failure must be analyzed and resolved — bypassing it is not allowed.

### Diagnostic flow

1. **Read build.log**
   ```bash
   less build.log  # or tail -100 build.log
   ```

2. **Locate the error type**

**First isolate the source**: stash the current changes and rebuild once. If the error disappears it came from your changes; otherwise it is a pre-existing/environmental problem (e.g., ci/build.sh overwriting the copy with the top-level source causing a version rollback).

| Error pattern | Likely cause | Fix |
|---------------|-------------|-----|
| `undefined reference to op_api::npu_xxx` | YAML signature does not match the C++ implementation | Check that the YAML function signature matches the C++ implementation |
| `error: 'xxx' was not declared in this scope` | Missing header or wrong namespace | Check include paths and namespace |
| `No rule to make target 'xxx'` | A new file was not picked up by GLOB | Trigger a reconfigure (touch CMakeLists.txt) |
| `fatal error: OpApiInterface.h: No such file` | gencode.sh was not run after editing yaml | Run `bash gencode.sh` first, then build |
| `undefined reference to aclnnXxx` | Operator not linked correctly, or the aclnn interface does not exist | Check the ACLNN version and confirm the interface is available |

### Special handling for "function not found"

When the error is `undefined reference to xxx function`, **identify the root cause first**:

| Root cause | Fix |
|-----------|-----|
| The target function is genuinely missing (not implemented) | **Implement it** |
| A caller references a stale/renamed symbol | **Fix the caller** |
| Version dispatch is needed (different CANN versions expose different interfaces) | Use version dispatch — this is a **legitimate pattern** |

**Forbidden**:
- ❌ Deleting/skipping functionality to "bypass" a compile error
- ❌ Referencing a non-existent symbol just to make the code compile
- ❌ Changing the call site without verification to mask the problem

**Allowed**:
- ✅ Version dispatch: choosing `aclnnXxxV2` vs `aclnnXxxV3` by CANN/chip version is a legitimate pattern

### Fix loop

1. Analyze the error → locate the root cause
2. Fix the code
3. Rebuild
4. If it still fails, go back to step 1

**Not allowed**: skipping ahead to later steps after a build failure.

---

## Common Pitfalls

### 1. GLOB pitfall

After adding a new `op_plugin/ops/opapi/*.cpp`, CMake's `FILE(GLOB)` does not automatically notice the new file.

**Symptom**: the new cpp is not compiled (link-time `undefined reference`, or `No rule to make target 'xxx'`)

**Fix**: trigger a reconfigure (`touch build/pytorch/third_party/op-plugin/CMakeLists.txt`)

### 2. Codegen ordering pitfall

If you build immediately after editing `op_plugin_functions.yaml`, compilation uses the stale `OpApiInterface.h`, causing link errors.

**Symptom**: `undefined reference to op_api::npu_xxx`

**Fix**: under `ci/build.sh`, codegen is idempotent within the pipeline and needs no manual step; only when bypassing the script and running `cmake --build` manually, run `bash gencode.sh` first, then build.

### 3. Misjudging the cost of a common header

`OpApiInterface.h` (344 opapi cpp) and `op_api_common.h` (348 cpp) are the highest-impact files.

**Risk**: changing a single line = 300+ translation units recompiled + minutes of link time

**Suggestion**: if you only need to add an interface for a single operator, consider putting the declaration inside that operator's cpp.

### 4. Bypassing ci/build.sh for an incremental build (dvm link failure)

**Symptom**: at link time, a batch of `undefined reference to lazy_fusion::abs / mul_ / exp / silu / where ...` from `OpInterface.cpp`, ending in `final link failed`.

**Root cause**: the dvm ("lazy fusion") module's call sites are **generated by gencode into `OpInterface.cpp`**, not hand-written. If you sync source manually (`cp -rf op_plugin build/...`) and then run `cmake --build` directly, you skip gencode: the regenerated callers and the `lazy_fusion_*.cpp` definitions fall out of sync (or the new sources are not consistently pulled into the link), so the linker can't resolve those symbols.

**Fix**: always run the incremental build through `ci/build.sh` (keep `build/`, don't rm it). It re-copies the tree, runs gencode, and configures consistently, so the dvm callers and definitions always match.

**Note on ccache/mtime**: `ci/build.sh`'s internal `cp -rf` refreshes file mtimes, which makes ninja *re-examine* every file — but ccache keys on preprocessed **content**, not mtime, so unchanged translation units hit the cache and cost ~milliseconds each. The real incremental cost is the single relink of `libtorch_npu.so`, not recompilation. So the `cp -rf` does **not** slow the build via cache misses.

---

## Graph Mode Build (torch.compile / npugraph_ex)

When the operator supports graph mode (confirmed in C-confirm), the build must include torchair. By default, `ci/build.sh` uses `--disable_torchair` to reduce package size, but this causes `torch.compile` to silently fall back to eager mode.

### Enable torchair for graph mode

```bash
# Check if --disable_torchair is present (usually around line 137)
grep -n "disable_torchair" ci/build.sh

# Remove the flag before building
sed -i 's/--disable_torchair//g' ci/build.sh
```

### Verify torchair was compiled

```bash
# Check build log for torchair target
grep "Built target _torchair" build.log

# Check installed package
python -c "import torch_npu; print(torch_npu.dynamo.torchair.__file__)"
```

If `Built target _torchair` is NOT in build.log → torchair was not compiled → `torch.compile(backend="npu")` will silently fall back to eager.

For most operator testing, use `backend="npugraph_ex"` (always available, pre-initialized, faster first invocation). `backend="npu"` requires torchair compiled in and is slower on first call.

## After a Successful Build

```bash
pip install --force-reinstall build/pytorch/dist/torch_npu-*.whl
```

After confirming the installation succeeded, proceed to the C-test step.
