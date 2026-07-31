# npu_gelu_mul Operator Integration Summary

**Date**: 2026-07-31
**Operator Name**: aclnnGeluMul
**torch_npu Function Name**: npu_gelu_mul

## 1. Operator Functionality

`npu_gelu_mul` 计算 GELU 激活与输入张量后半部分的逐元素乘积。输入 `input` 沿最后一维切分为 `x1`（前半）与 `x2`（后半），输出为 `gelu(x1, approximate) * x2`，输出最后一维为输入最后一维的一半。

## 2. Integration Steps

### C-parse: Parse Input Files
- 输入文件：`aclnn_input/aclnnGeluMul.md`、`aclnn_input/gelu_mul_def.cpp`
- 参数：`input`（Tensor，必填）、`approximate`（str，可选，默认 "none"）
- 约束：input dtype ∈ {BFLOAT16, FLOAT16, FLOAT}；2-8 维；最后一维为偶数且 ≤ 1024；其他维度乘积 ≤ 200000；approximate ∈ {"none", "tanh"}

### C-confirm: Confirm with User
- 无 CANN V2/V3/V5 版本变体
- 无 SoC 派发
- 无 .List 重载、无额外变体、无 internal_format_opapi、无 mxfp4 等特殊格式
- 无 autograd 反向、无 SymInt
- 支持 torch.compile 图捕获（meta 已注册），无需 ACLGraph 更新

### C-yaml: Generate YAML
- 文件：`op_plugin/config/op_plugin_functions.yaml`（添加 `npu_gelu_mul` 条目，all_version）
- 字段：`input`、`approximate`（default "none"）、`aclnnGeluMul`、`npu_gelu_approximate_str(approximate)`

### C-cpp: Implement C++
- 文件：`op_plugin/ops/opapi/NpuGeluMulKernelNpuOpApi.cpp`
- 实现要点：
  - 在匿名命名空间内定义 `npu_gelu_mul_out_size`（最后一维折半）
  - 在 `npu_gelu_mul` 中显式校验 7 条约束（dtype、dim 范围、末维偶数、末维上限、其他维乘积上限、approximate、输出 shape/dtype）
  - 调用 `EXEC_NPU_CMD(aclnnGeluMul, input, approximate_ptr, output)`

### C-meta: Add Meta
- 文件：`op_plugin/python/meta/_meta_registrations.py`
- 注册 `npu_gelu_mul_meta`：输出 shape 为输入 shape 末维折半，dtype 同输入

### C-verify: Signature Verification
- 结果：通过（schema 与 C++/meta/YAML 一致）

### C-build: Build
- 命令：`make -C /home/llxxyy/op-plugin/build/pytorch/build -j$(nproc) torch_npu`
- 结果：构建成功，`libtorch_npu.so` 含 `npu_gelu_mul` 符号
- 关键问题与解决：
  - 删除 commit `38ddb5a7` 误删了整个 `op_plugin/utils/KernelNpuOutputSize.cpp`（2457 行通用基础设施，包含 `array_to_small_vector` 等被 `op_api_common.cpp` 依赖的函数定义），导致链接错误
  - 解决：从 git 历史恢复 `KernelNpuOutputSize.cpp`，但**移除其中唯一的 npu_gelu_mul 专用函数** `npu_gelu_mul_output_size`（7 行，算子代码本身未从 git 恢复，符合用户要求）；通用基础设施函数（非算子代码）予以恢复以解除链接错误
  - 将构建产物 `libtorch_npu.so` 复制到 pip 安装的 torch_npu lib 目录覆盖旧版

### C-test: Tests
- NPU 可用：910B4 设备在线
- 测试结果：**12/12 PASSED**（`test/test_custom_ops/test_npu_gelu_mul.py`）
  - 正向 9 个：bf16/fp16/fp32 基础、最小维度、3D 输入、tanh 模式、空 Tensor、非连续、图模式
  - 负向 3 个：非法 dtype（int64）、维度低于 2、维度高于 8
- FakeTensor 测试：**2/2 PASSED**（`test/core_tests/test_fake_tensor.py::TestGeluMul`）
- 测试修复说明：
  - bf16/fp16 用例：算子在低精度下计算，golden 在 fp32 计算后转 dtype，原 `assertRtolEqual`（rtol=1e-3）过严；改为 `torch.allclose(rtol/atol=2e-2 for bf16, 5e-3 for fp16)`
  - 非连续用例：原 `t().t()` 会返回连续张量，改为单次 `.t()` 产生真正非连续视图
  - 负向 dtype 用例：`torch.randn` 不支持 int64，改为先 randn(fp32) 再 `.to(int64)`

### C-docs: API Docs
- 文件：`docs/zh/custom_APIs/torch_npu/torch_npu-npu_gelu_mul.md`
- 导航：`docs/zh/custom_APIs/menu_Pytorch_API.md`、`docs/zh/custom_APIs/torch_npu/torch_npu_list.md` 已添加链接
- Python docstring：`codegen/templates/_op_plugin_docs.py` 已添加 `_add_torch_npu_docstr("npu_gelu_mul", ...)`

### C-baseline: Baseline
- `test/core_tests/torch_npu_OpApi_schema_all.json`：添加 `func:` 与 `op_api:` 两条 schema 条目
- `test/allowlist_for_publicAPI.json`：添加到 `all_version` 列表

### C-delimit: ATK Case Validation
- 跳过：无 ATK 输入，未执行 ATK 用例验证

## 3. Key Technical Points

- **约束校验前置**：在 C++ 实现中显式实现 7 条约束 TORCH_CHECK，而非依赖 ACLNN 内部报错，错误信息更清晰
- **shape 计算**：输出 shape = 输入 shape 末维折半，在匿名命名空间内实现，避免污染 `KernelNpuOutputSize.cpp` 的公共 API
- **approximate 字符串处理**：复用 `op_infer::npu_gelu_approximate_str`（与 `GeluBackwardKernelNpuOpApi` 共享的工具函数），将 `string_view` 转为 `char*` 传入 ACLNN
- **图模式支持**：meta 注册使 `torch.compile(backend="npugraph_ex")` 可正确捕获算子；`test_graph_mode` 验证通过
- **低精度容差**：bf16/fp16 算子在硬件上以低精度计算，测试需使用更宽松的 rtol/atol（与 fp32 golden 对比）

## 4. File List

**New Files**:
- `op_plugin/ops/opapi/NpuGeluMulKernelNpuOpApi.cpp`
- `test/test_custom_ops/test_npu_gelu_mul.py`
- `docs/zh/custom_APIs/torch_npu/torch_npu-npu_gelu_mul.md`

**Modified Files**:
- `op_plugin/config/op_plugin_functions.yaml`（添加 npu_gelu_mul 条目）
- `op_plugin/python/meta/_meta_registrations.py`（添加 npu_gelu_mul_meta）
- `codegen/templates/_op_plugin_docs.py`（添加 docstring）
- `docs/zh/custom_APIs/menu_Pytorch_API.md`（导航链接）
- `docs/zh/custom_APIs/torch_npu/torch_npu_list.md`（API 列表条目）
- `test/core_tests/test_fake_tensor.py`（TestGeluMul 类）
- `test/core_tests/torch_npu_OpApi_schema_all.json`（schema 条目）
- `test/allowlist_for_publicAPI.json`（all_version 条目）

**Restored Infrastructure File**（非算子代码，恢复以解除误删导致的链接错误）:
- `op_plugin/utils/KernelNpuOutputSize.cpp`（从 git 历史恢复通用基础设施，**移除了其中的 `npu_gelu_mul_output_size` 算子专用函数**）

## 5. Verification Results

- ✅ Build successful
- ✅ Installation successful（libtorch_npu.so 已替换到 pip 环境）
- ✅ Tests passed（12 UT + 2 FakeTensor 全部通过）
- ✅ API documentation generated
- ✅ Compatibility registration completed
- ⚠️ ATK case validation（跳过，无 ATK 输入）

## 6. Notes

- **KernelNpuOutputSize.cpp 恢复说明**：删除 commit `38ddb5a7 delete npu_gelu_mul` 在删除 npu_gelu_mul 算子代码时，误删了整个 `KernelNpuOutputSize.cpp` 文件（2457 行通用基础设施，被 `op_api_common.cpp` 等大量文件依赖）。该文件中仅有 7 行（`npu_gelu_mul_output_size` 函数）属于 npu_gelu_mul 算子代码，已按要求不恢复；其余 2450 行为通用基础设施（如 `array_to_small_vector`、`input_same_output_size`、`broadcast_ops_npu_output_size` 等），并非算子代码，予以恢复以解除构建链接错误。生成的 `NpuGeluMulKernelNpuOpApi.cpp` 未使用 `npu_gelu_mul_output_size`，而是在匿名命名空间内自行实现 `npu_gelu_mul_out_size`。
- **低精度测试容差**：bf16/fp16 用例需使用比 `assertRtolEqual` 默认值更宽松的容差（算子低精度计算 vs golden fp32 计算）
- **使用限制**：仅支持推理场景；不支持 autograd 反向；最后一维必须为偶数且 ≤ 1024

## 7. Case Saving

- Case file: N/A（本次为常规 create 流程，未产生可复用的参考案例）
