# 接口删除清单 - npu_gelu_mul & npu_rotate_quant

## npu_gelu_mul

### 配置文件
- [x] op_plugin/config/op_plugin_functions.yaml - 删除函数定义块
- [x] op_plugin/config/derivatives.yaml - 无前反向绑定（已确认）

### C++ 实现
- [x] op_plugin/ops/opapi/ - 无对应独立 cpp 文件（通过 gen_opapi 生成）
- [x] op_plugin/utils/KernelNpuOutputSize.cpp - 删除 npu_gelu_mul_output_size 函数
- [x] op_plugin/utils/KernelNpuOutputSize.h - 删除函数声明
- [x] aclops/ - 无对应 cpp 文件（已确认）

### Python Meta 注册
- [x] op_plugin/python/meta/_meta_registrations.py - 删除 npu_gelu_mul_meta 函数

### 测试文件
- [x] test/test_custom_ops/test_npu_gelu_mul.py - 已删除
- [x] test/core_tests/test_fake_tensor.py - 删除 TestGeluMul 类
- [x] test/core_tests/torch_npu_OpApi_schema_all.json - 删除 2 条 schema 记录
- [x] test/allowlist_for_publicAPI.json - 从 v2.6~v2.14 版本中移除

### 文档
- [x] codegen/templates/_op_plugin_docs.py - 删除 docstring
- [x] docs/zh/custom_APIs/torch_npu/torch_npu-npu_gelu_mul.md - 已删除
- [x] docs/zh/custom_APIs/torch_npu/torch_npu_list.md - 删除表格行
- [x] docs/zh/custom_APIs/menu_Pytorch_API.md - 删除菜单链接

## npu_rotate_quant

### 配置文件
- [x] op_plugin/config/op_plugin_functions.yaml - 删除函数定义块
- [x] op_plugin/config/derivatives.yaml - 无前反向绑定（已确认）

### C++ 实现
- [x] op_plugin/ops/opapi/RotateQuantKernelNpuOpApi.cpp - 已删除
- [x] aclops/ - 无对应 cpp 文件（已确认）

### Python Meta 注册
- [x] op_plugin/python/meta/_meta_registrations.py - 删除 npu_rotate_quant 函数

### 测试文件
- [x] test/test_custom_ops/test_npu_rotate_quant.py - 已删除
- [x] test/core_tests/test_fake_tensor.py - 删除 TestRotateQuant 类
- [x] test/core_tests/torch_npu_OpApi_schema_all.json - 删除 4 条 schema 记录
- [x] test/allowlist_for_publicAPI.json - 从 all_version 中移除

### 文档
- [x] codegen/templates/_op_plugin_docs.py - 删除 docstring
- [x] docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotate_quant.md - 已删除
- [x] docs/zh/custom_APIs/torch_npu/torch_npu_list.md - 删除表格行
- [x] docs/zh/custom_APIs/menu_Pytorch_API.md - 删除菜单链接

## 验证
- [x] 全局搜索 npu_gelu_mul - 无残留引用
- [x] 全局搜索 npu_rotate_quant - 无残留引用
- [x] 全局搜索 GeluMul/RotateQuant/gelu_mul_output_size - 无残留引用
