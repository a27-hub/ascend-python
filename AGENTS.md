# Project Notes (op-plugin)

## Build

- `MAX_JOBS=64` for `ci/build.sh` — this host has 256 cores / ~1.9TB RAM (no cgroup limit). 64 is a safe value that does not starve other users on this shared host. Set via env before launching the build:
  ```bash
  MAX_JOBS=64 bash ci/build.sh --python=3.10 --pytorch=v2.10.0
  ```
- Environment: `source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh` before any build/test/run.
- Graph mode (torch.compile) builds: remove `--disable_torchair` from `ci/build.sh` line ~137.
- Test deps that may be missing: `numpy`, `decorator`, `tornado`, `expecttest`.

## Adaptation Workflow (op-adaptation skill)

- **C-test must run in parallel with the build, not after it.** Per the skill execution order, C-test steps 1-5 (scaffold_init → supplement parameter semantics → coverage matrix → generate UT file → generate FakeTensor class) are pure generation and do NOT depend on torch_npu being installed. Run them while the build (C-build, step 9) is compiling. Only C-test step 6 (execute) needs the built+installed wheel. Concretely: after launching `ci/build.sh` in the background, immediately invoke the op-test skill to drive steps 1-5; by the time the build finishes, only step 6 (execution + triage) remains. Serializing test generation after the build wastes ~the full build duration.
- Same rule applies to C-docs (op-apidoc) and C-baseline gen — all parallel with the build; only their verify/run phases wait for it.
- Pitfall to re-apply when patching meta after a build: the installed package's `_meta_registrations.py` is a copy — sync the source fix to the installed path (or rebuild) or the meta change won't take effect.
