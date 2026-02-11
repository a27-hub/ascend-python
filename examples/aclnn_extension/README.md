运行依赖torch、torch_npu和CANN。具体安装步骤参考[torch_npu文档](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85)

## 适配文件结构

```
npu_cpp_extension
├── gen.sh                            # 调用torchnpugen工具脚本，用于生成算子适配代码
├── add_custom.yaml                   # 自定义算子示例yaml（无需绑定前反向ATen IR定义）
├── npu_fast_gelu.yaml                # 自定义算子示例yaml（需要绑定前反向ATen IR定义）
├── npu_fast_gelu_derivatives.yaml    # 算子前反向绑定配置yaml
├── op_plugin                         # 生成的算子适配代码目录
│   ├── config                        # 算子配置文件目录
│   │   └── v2r7                      # Torch版本
│   │       ├── derivatives.yaml      # 自动生成清洗后的算子反向传播yaml
│   │       └── op_plugin_functions.yaml  # 自动生成清洗后的ATen IR定义yaml
│   ├── OpInterface.h                 # 自动生成op_plugin对外接口的头文件，用于框架侧调用算子
│   ├── OpInterface.cpp               # 自动生成op_plugin对外接口路由实现，内部实现不同类型算子分支选择代码
│   ├── OpApiInterface.h              # 自动生成aclnn算子插件适配所对应头文件
│   ├── AclOpsInterface.h             # 自动生成aclop算子插件适配所对应头文件 
│   ├── ASparseOpsInterface.h         # 稀疏算子专用适配头文件
│   ├── ops                           # 算子适配文件目录
│   │   └── opapi                     # aclnn算子适配目录
│   │       └── StructKernelNpuOpApi.cpp  # 结构化适配自动生成的适配文件
├── setup.py                          # 项目构建脚本，用于编译生成whl包
└── test                              # 测试脚本目录
    ├── test_add_custom.py            # add_custom算子测试脚本
    └── test_npu_fast_gelu_custom.py  # npu_fast_gelu_custom算子测试脚本
```

注：1、需要手写的文件add_custom.yaml、setup.py，若自定义前反向绑定算子则需要npu_fast_gelu.yaml以及对应的npu_fast_gelu_derivatives.yaml。
2、op_plugin下的文件均由gen.sh生成

## 操作步骤

说明：以下流程以npu_fast_gelu_custom算子适配为例进行说明。

### 1. 自定义算子ATEN IR yaml编写

在npu_fast_gelu.yaml里的custom进行结构化适配。

判断是否可结构化依据：opapi对应的aclnn算子与Aten IR的语义对齐，适配层除申请output tensor，无其他适配逻辑。

```yaml
custom:
  - func: npu_fast_gelu_custom(Tensor self) -> Tensor
    op_api: all_version
    gen_opapi:
      out:
        size: self
        dtype: self
      exec: aclnnFastGelu
  - func: npu_fast_gelu_custom_backward(Tensor grad, Tensor self) -> Tensor
    op_api: all_version
    gen_opapi:
      out:
        size: grad
        dtype: grad
      exec: aclnnFastGeluBackward
```

### 2. 前反向绑定算子配置yaml编写

在npu_fast_gelu_derivatives.yaml里进行前反向绑定算子配置。

PyTorch的算子自动反向微分依赖于算子的前反向绑定，即前向函数和反向函数的绑定。对于原生的算子，官方已有前反向绑定逻辑，插件侧有对应前向算子和反向算子适配即可（只需要在op_plugin_functions.yaml里面配置）。对于自定义算子，则需要在插件侧配置前反向自动绑定。

针对需要绑定前反向的算子（包括自定义算子和前反向绑定逻辑与原生不一致的原生算子）提供自动绑定前向算子和反向算子的功能。

```yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10, v2.11]
backward:
- name: npu_fast_gelu_custom(Tensor self) -> Tensor
  self: npu_fast_gelu_custom_backward(grad, self)
  version: all_version
```

### 3. 启动gen.sh，生成算子适配代码

```sh
bash gen.sh npu_fast_gelu.yaml npu_fast_gelu_derivatives.yaml
```

## 运行自定义的算子

运行流程：

1. 运行setup脚本，编译生成whl包。
   
   ```bash
   python setup.py build bdist_wheel
   ```
2. 安装whl包
   
   ```bash
   cd dist
   pip install *.whl
   ```
3. 运行样例
   
   ```bash
   cd test
   python test_npu_fast_gelu_custom.py
   ```
