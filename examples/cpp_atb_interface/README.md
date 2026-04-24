# C++ ATB Wrapper 样例

本样例演示如何在 `torch_npu` 同构运行环境中，通过 Python C++ extension 调用一个 ATB public C++ wrapper。
当前样例只保留一个接口：

- `at_npu::native::atb::_npu_quantize_per_tensor`

## 前置条件

运行样例前，请确认以下依赖已经准备完成：

1. `python3` 可以正常导入 `torch_npu`。
2. `torch_npu` 安装目录的 `lib/` 下存在 `libop_plugin_atb.so`。
3. `ATB_HOME_PATH` 已设置，且其目录树下可以搜索到真实 `libatb.so`。

本样例固定使用 `torch_npu` 安装目录中的 `libop_plugin_atb.so`，不依赖仓库本地 build 树中的该 so。

`build_and_run.sh` 和 `setup.py` 只会从 `ATB_HOME_PATH` 下递归搜索 `libatb.so`。
因此，`ATB_HOME_PATH` 可以直接指向任意包含 `libatb.so` 的目录树。

## 目录结构

```text
.
├── build_and_run.sh
├── setup.py
├── README.md
├── csrc
│   └── atb_quantize_per_tensor.cpp
├── cpp_atb_interface
│   └── __init__.py
└── test
    └── test_atb_quantize_per_tensor.py
```

## 运行方式

执行如下命令：

```bash
bash build_and_run.sh
```

脚本会完成以下步骤：

1. 构建样例 wheel 包。
2. 安装 wheel 包。
3. 设置 `libop_plugin_atb.so` 和 `libatb.so` 的运行时路径。
4. 进入 `test/` 目录执行单接口测试用例。

## 成功判定

当 `test/test_atb_quantize_per_tensor.py` 执行通过时，说明样例运行成功。

## 说明

本样例用于演示如何在 `torch_npu` 同构环境中消费 ATB public C++ wrapper，不是独立 SDK 形态的样例。
