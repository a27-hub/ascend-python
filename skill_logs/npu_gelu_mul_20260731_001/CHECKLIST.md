# CHECKLIST — npu_gelu_mul (Create Mode)

run_dir: /home/llxxyy/op-plugin/skill_logs/npu_gelu_mul_20260731_001

## Project Constants
- OP_PLUGIN_ROOT_PATH: /home/llxxyy/op-plugin
- OP_PLUGIN_SCRIPT_PATH: /home/llxxyy/.traecli/skills/op-adaptation/scripts
- OP_PLUGIN_SKILL_HASH: non-git (skill dir 不是 git 仓库)
- OP_PLUGIN_SKILL_BRANCH: non-git
- OP_PLUGIN_BUILD_PATH: /home/llxxyy/op-plugin/build
- CCACHE_STATUS: OK_6332G
- op-plugin branch: add-gelu-mul

## Input Files
- md: /home/llxxyy/aclnn_input/aclnnGeluMul.md
- def: /home/llxxyy/aclnn_input/gelu_mul_def.cpp
- parse JSON: {run_dir}/notes/npu_gelu_mul.parse.json

## ATK Case Dir
- 无 ATK 输入（C-delimit 将跳过）

## C-parse: Parameter Information Table

| Parameter | Source | Type | Category | Optionality | Default | Notes |
|-----------|--------|------|----------|-------------|---------|-------|
| input | both | Tensor | input | REQUIRED | - | BF16/F16/F32, ND, dim 2-8, 支持空Tensor, 最后一维偶数且≤1024, 其他维度乘积≤200000 |
| approximate | both | str | attr | OPTIONAL | "none" | 取值 "none"/"tanh"，空指针="none" |
| output | _def.cpp | Tensor | output | REQUIRED | - | shape: 最后一维=input最后一维/2，其他维度一致；dtype=与input一致 |

### Conflicts Resolved
- **out (md) vs output (def)**: .md 的 `out` 是 aclnn 两段式接口的 C 风格输出 tensor 参数；_def.cpp 的 `output` 是 op_def 注册的算子输出。两者语义一致，以 _def.cpp 为权威——torch_npu 接入无 out 参数变体，输出名为 `output`。Resolution: exclude_def_cpp_is_authoritative ✓

### Constraints (需在 C++ 实现 TORCH_CHECK)
1. input 最后一维值为偶数
2. input 最后一维值 ≤ 1024
3. input 其他维度乘积 ≤ 200000
4. input dim 范围 2-8
5. input dtype ∈ {BF16, F16, F32}
6. approximate ∈ {"none", "tanh"} （已在 npu_gelu_approximate_str 中校验）
7. 输出 dtype 与 input 一致
8. 输出 shape: 最后一维 = input 最后一维 / 2

### Op Type
- Forward (无 backward/grad，无 autograd)

### Gen Mode
- **gen_opapi** (codegen 自动生成 C++ 实现)，YAML 已含 gen_opapi 块

## 现有代码状态（关键发现）
- YAML 注册: ✓ 已存在 (op_plugin_functions.yaml:5624-5634)
- 辅助函数 npu_gelu_mul_output_size: ✓ 已存在 (KernelNpuOutputSize.cpp:2146)
- 辅助函数 npu_gelu_approximate_str: ✓ 已存在 (KernelNpuNewParams.cpp:26)
- Meta 注册: ✓ 已存在 (_meta_registrations.py:2471-2478)
- C++ 实现 op_plugin/ops/opapi/: ✗ 已被用户删除（需通过 codegen 重新生成）
- 测试文件: ✓ 已存在 (test/test_custom_ops/test_npu_gelu_mul.py)

## Step Status
- [x] C-parse
- [ ] C-confirm
- [ ] C-yaml (已存在，需验证)
- [ ] C-derivatives (forward only, skip)
- [ ] C-cpp (codegen 生成)
- [ ] C-constraints
- [ ] C-meta (已存在，需验证)
- [ ] C-verify
- [ ] C-codegen-check
- [ ] C-build
- [ ] C-test
- [ ] C-docs
- [ ] C-baseline
- [ ] C-delimit (无 ATK, skip)
- [ ] C-summary
