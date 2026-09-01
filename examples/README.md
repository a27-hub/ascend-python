# 适配说明

## 概述

C++ extensions插件提供了将自定义算子映射到昇腾AI处理器的功能，为使用PyTorch框架的开发者提供了便捷的NPU算子库调用能力，基于PyTorch原生提供的自定义算子扩展功能，用户可以编译安装自定义算子wheel包并运行。本章节基于C++ extensions的方式介绍如何在昇腾NPU上完成自定义算子开发和适配。PyTorch官方的C++ extensions功能具体可参考[C++ extensions官方文档](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

## 适配样例目录

此目录下展示了C++ extensions构建自定义算子库的所有样例。

```text
├── aclnn_extension                   // aclnn算子适配样例
├── cpp_extension_base                 // 基础样例
├── cpp_extension_full                 // 完整样例
│   ├── module                         // 前反向绑定通过python实现
│   └── torch_lib_impl                // 前反向绑定通过c++实现
├── cpp_extension_structured           // 结构化适配实现
├── cpp_extension_asc                  // 算子通过AscendC实现
├── cpp_extension_pybind               // 通过pybind绑定对外接口
├── framwork_cpp_extension             // 框架级适配样例
└── kernel_extension_aclgraph          // AscendC算子+aclgraph样例
    ├── pybind                         // 通过pybind绑定对外接口
    └── torch_library                  // 通过torch_library绑定对外接口
```

## 适配方法

提供了九种适配样例，用户可根据下表快速选择适合自己的适配方法。

**如何选择**：先确认算子来源（已有 aclnn 接口 / 自己用 AscendC 编写），再根据是否需要反向、图模式等缩小范围。

| 适配方法 | 算子来源 | 反向 | 图模式 | 适配代码 | 适用场景 | 更多信息 |
| --- | :---: | :---: | :---: | :---: | --- | --- |
| cpp_extension_base | aclnn | ✗ | 仅eager | 手写C++ | 最简上手，仅需前向调用 | [基础样例](./cpp_extension_base/README.md) |
| cpp_extension_pybind | aclnn | ✗ | 仅eager | 手写C++ | pybind绑定，不走ATen IR，灵活Python调用 | [pybind样例](./cpp_extension_pybind/README.md) |
| cpp_extension_full（module） | aclnn | ✓ | 仅eager | 手写C++ + Python类 | 需前反向，用Python类注册forward/backward | [完整样例-module](./cpp_extension_full/module/README.md) |
| cpp_extension_full（`TORCH_LIBRARY_IMPL`） | aclnn | ✓ | 仅eager | 手写C++ | 需前反向，用C++ `TORCH_LIBRARY_IMPL` 注册 | [完整样例-impl](./cpp_extension_full/torch_lib_impl/README.md) |
| aclnn_extension | aclnn | ✓ | 仅eager | YAML自动生成 | aclnn与ATen语义一致，gen.sh生成，无需手写C++ | [aclnn样例](./aclnn_extension/README.md) |
| cpp_extension_structured | aclnn | ✓ | 仅eager | YAML自动生成 | 同上，op-plugin内置YAML方式 | [结构化样例](./cpp_extension_structured/README.md) |
| cpp_extension_asc | AscendC | ✗ | 仅eager | 手写`.asc` | 算子用AscendC编写，通过`.asc`文件实现 | [AscendC样例](./cpp_extension_asc/README.md) |
| framwork_cpp_extension | aclnn | ✓ | eager + torch.compile | 手写C++ | 需eager和torch.compile双模式，含图模式converter注册 | [框架级样例](./framwork_cpp_extension/README.md) |
| kernel_extension_aclgraph | AscendC | ✗ | eager + aclgraph | 手写`.asc` | 需要aclgraph支持，含pybind和torch_library两种绑定 | [aclgraph样例](./kernel_extension_aclgraph/pybind/README.md) |

> [!NOTE]
> 
> cpp_extension_structured（结构化实现）使用前提：opapi对应的aclnn与ATen IR语义一致，适配层除申请output tensor外无其他逻辑。
