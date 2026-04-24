# atb_cpp_interface 样例说明

## 前提条件

运行本样例前，请先确认以下环境已准备完成：

1. Pytorch框架已安装，`torch`，`torch_npu` 组件可正常导入。
2. CANN软件已安装且环境变量 `ATB_HOME_PATH` 已配置。

环境变量 `ATB_HOME_PATH` 通常可通过如下命令进行配置，`{CANN_INSTALL_PATH}` 是CANN软件的安装目录。

```bash
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh
```

## 适配文件结构

```text
├── build_and_run.sh                      // 编译、安装并执行测试
├── csrc                                  // C++ 适配代码目录
│   └── atb_quantize_per_tensor.cpp       // ATB C++ 接口封装
├── atb_cpp_interface                     // Python 侧代码目录
│   └── __init__.py                       // Python 侧导出接口
├── setup.py                              // wheel 构建文件
└── test                                  // 测试用例目录
    └── test_atb_quantize_per_tensor.py   // 测试用例
```

## 操作步骤

1. 进入样例目录：

   ```bash
   cd examples/atb_cpp_interface
   ```

2. 执行样例脚本：

   ```bash
   bash build_and_run.sh
   ```

该脚本会完成样例的编译、wheel 包安装以及测试执行。

## 成功描述

满足以下条件时，可认为样例执行成功：

- `build_and_run.sh` 执行结束且未报错。
- 测试输出中出现 `Ran 1 test`。
- 测试输出中出现 `OK`。
