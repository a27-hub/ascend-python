# aclnn_extension 样例说明

本样例是 **「自定义算子接入 - 结构化代码生成」** 的示例工程，演示如何通过 YAML 描述 + 代码生成工具，将自定义 aclnn 算子接入 PyTorch NPU 生态，并生成可参与 torch_npu 编译的适配代码。

## 适用场景

本样例适用于以下情况：

- **需要接入自定义 aclnn 算子**：CANN 已提供 aclnn 接口（如 `aclnnFastGelu`），希望以 PyTorch 自定义算子的形式在 `torch_npu` 中暴露给用户。
- **算子满足「结构化适配」条件**：  
  aclnn 算子与目标 ATen IR 的语义一致，适配层**除申请 output tensor（shape/dtype 由输入推导）外，无其他复杂逻辑**。  
  满足时即可用 YAML 做结构化描述，由工具自动生成 C++ 适配代码，无需手写适配层。
- **可选：需要自动求导**：若希望自定义算子参与 autograd，需在 YAML 中配置前向与反向的对应关系，由工具生成 derivatives 配置。

不适用或需手写适配的情况：算子语义与 ATen 不一致、需要复杂 shape/type 推导、或需在适配层写额外逻辑时，应使用非结构化方式（如 acl_op 或手写 opapi）。

## 环境与依赖

- **运行/编译依赖**：PyTorch、torch_npu、CANN。  
  安装与版本要求见 [torch_npu 安装说明](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85)。


## 目录与文件结构

### 运行前
```
aclnn_extension/
├── gen.sh                            # 一键生成脚本：调用 torchnpugen 生成算子适配代码
├── setup.py                          # 项目构建脚本，用于编译生成 whl 包
├── npu_fast_gelu.yaml                # 自定义算子 YAML（含前向/反向 ATen IR 与 aclnn 映射）
├── npu_fast_gelu_derivatives.yaml    # 前向/反向绑定配置（用于 autograd）
├── test_native_functions.yaml        # NPU backend 声明（生成 stub 等时会使用）
├── test/
│   └── test_npu_fast_gelu_custom.py  # 自定义算子测试脚本
└── README.md
```



### 运行后

先执行 `gen.sh` 生成适配代码，再执行 `setup.py` 构建后，在 `./dist/`下得到 **`custom_ops.whl`**：

```
aclnn_extension/
├── ...（运行前已有文件）
├── op_plugin/
├── csrc/
├── autograd/
├── utils/
├── dist/
│   └── custom_ops.whl
└── ...
```

使用方式：`pip install custom_ops.whl` 安装后，即可在 Python 中调用本样例接入的自定义算子（如 `torch_npu.npu_fast_gelu_custom`）。


## 一键运行样例

若仅想快速跑通本样例（不修改算子），在 `aclnn_extension` 目录下**按顺序**执行：

```bash
# 1. 先执行 gen.sh，生成适配代码
bash gen.sh npu_fast_gelu.yaml npu_fast_gelu_derivatives.yaml

# 2. 再执行 setup.py 构建 whl 包并安装
python setup.py build bdist_wheel
cd dist
pip install custom_ops.whl

# 3. 运行测试验证
cd test && python test_npu_fast_gelu_custom.py
```

测试通过即表示样例已跑通，可直接调用 `torch_npu.npu_fast_gelu_custom` 算子。

---

**改成接入自己的算子时**：只需把 YAML 里的内容换成自己算子的定义（见下方「使用步骤」），然后同样先跑 `gen.sh`（传入你的 YAML 文件名），再跑 `setup.py` 即可。

## 使用步骤

### 1. 编写自定义算子 YAML（结构化描述）

在 `npu_fast_gelu.yaml` 的 `custom` 段中，按「ATen IR ↔ aclnn」一一对应的方式填写。  
是否可结构化的判断标准：**opapi 对应的 aclnn 与 ATen IR 语义一致，适配层除申请 output tensor 外无其他逻辑**。

示例（前向 + 反向）：

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

- `func`：PyTorch 侧暴露的算子签名（ATen IR 形式）。
- `gen_opapi.out`：输出 tensor 的 shape/dtype 由哪个输入推导（如 `self` / `grad`）。
- `exec`：实际调用的 aclnn 接口名。

### 2. 可选：编写前反向绑定 YAML（用于 autograd）

若需要自动求导，在 `npu_fast_gelu_derivatives.yaml` 中配置前向对反向的绑定，例如：

```yaml
all_version: [v2.1, v2.2, v2.3, v2.4, v2.5, v2.6, v2.7, v2.8, v2.9, v2.10, v2.11]

backward:
  - name: npu_fast_gelu_custom(Tensor self) -> Tensor
    self: npu_fast_gelu_custom_backward(grad, self)
    version: all_version
```

### 3. 执行代码生成

在 `aclnn_extension` 目录下执行：

```bash
# 仅生成主算子（无前反向绑定）
bash gen.sh npu_fast_gelu.yaml

# 生成主算子 + 前反向绑定
bash gen.sh npu_fast_gelu.yaml npu_fast_gelu_derivatives.yaml
```

脚本会根据当前环境的 PyTorch 版本生成适配代码。

### 4. 构建 whl 包并运行测试

在步骤 3 完成（gen.sh 已执行）后，再执行 `setup.py` 构建，在 `./dist/`下得到 **`custom_ops.whl`**：

```bash
python setup.py build bdist_wheel
cd dist
pip install custom_ops.whl
```

安装完成后即可在代码中调用接入的自定义算子（如 `torch_npu.npu_fast_gelu_custom`）。例如用样例自带的测试用例验证：

```bash
cd test
python test_npu_fast_gelu_custom.py
```

测试通过即说明算子已正确接入、结果与参考实现一致。

## 自用时的替换与扩展

改成接入自己的算子时，**核心是把 YAML 里的内容换成自己算子的定义**：

- **算子 YAML**（如 `npu_fast_gelu.yaml` 或自建 `my_op.yaml`）：在 `custom` 段中写上自己的 `func`、`gen_opapi.out`、`exec`（aclnn 接口名）等，格式参考本样例。
- **前反向绑定**（可选）：若需自动求导，在 derivatives YAML 中配置前向与反向的对应关系；若不需要，可不提供该文件，`gen.sh` 只传算子 YAML 即可。
- **gen.sh 参数**：若使用新文件名（如 `my_op.yaml`），则执行 `bash gen.sh my_op.yaml` 或 `bash gen.sh my_op.yaml my_op_derivatives.yaml`。
- **测试脚本**：在 `test/` 下修改或新增测试，调用你暴露的算子名做数值验证。

流程不变：先执行 `gen.sh`（传入你的 YAML），再执行 `setup.py` 构建得到 `custom_ops.whl`，`pip install` 后即可调用。

## 小结

- **适用**：自定义 aclnn 算子、语义与 ATen 对齐、适配层仅做 output 申请的结构化场景。  
- **执行流程**：先执行 `gen.sh npu_fast_gelu.yaml npu_fast_gelu_derivatives.yaml` 生成适配代码，再执行 `setup.py` 构建得到 `custom_ops.whl`，`pip install` 后即可调用接入的算子。  
- **自用**：把 YAML 里的内容换成自己算子的定义，gen.sh 传入对应 YAML 文件名，同样先 gen.sh 再 setup.py 即可。
