# atb_cpp_interface 样例说明

## 概述

本样例演示如何通过 C++ extension 调用 ATB C++ 接口 `at_npu::native::atb::_npu_quantize_per_tensor`。

## 目录结构

```text
.
├── build_and_run.sh
├── setup.py
├── README.md
├── csrc
│   └── atb_quantize_per_tensor.cpp
├── atb_cpp_interface
│   └── __init__.py
└── test
    └── test_atb_quantize_per_tensor.py
```

## 运行依赖

`at_npu::native::atb::_npu_quantize_per_tensor` 的对外符号位于 `torch_npu` 安装目录中的 `libop_plugin_atb.so`。`setup.py` 编译扩展时会链接该库。运行时还需要真实 `libatb.so`，样例通过 `ATB_HOME_PATH` 查找该库。

## 运行样例

运行前请确保：

1. `torch` 可正常导入。
2. `torch_npu` 可正常导入。
3. `ATB_HOME_PATH` 已可用。

`ATB_HOME_PATH` 通常可通过 `source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh` 设置。

在样例目录下执行：

```bash
bash build_and_run.sh
```

脚本会依次完成以下步骤：

1. 执行 `python setup.py build bdist_wheel`，编译 wheel 包。
2. 执行 `python -m pip install --force-reinstall dist/*.whl`，安装 wheel 包。
3. 进入 `test/` 目录，执行 `test_atb_quantize_per_tensor.py`。
4. 测试脚本分别调用 `torch.ops.atb._npu_quantize_per_tensor` 和 `atb_cpp_interface.quantize_per_tensor`，并比较两条路径的输出结果。

用例通过判定方式如下：

- `build_and_run.sh` 执行完成。
- 测试输出中出现 `Ran 1 test` 和 `OK`。
