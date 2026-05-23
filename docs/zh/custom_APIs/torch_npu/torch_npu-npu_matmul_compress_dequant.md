# torch_npu.npu_matmul_compress_dequant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

对压缩存储的权重重建后进行矩阵乘与反量化计算。即使用压缩索引(compress_index)对压缩权重(x2)解压，与输入(x1)做矩阵乘，加上偏置(bias)，再按 scale 做反量化，得到 float16 结果。适用于 8x8 块压缩的权重量化推理场景。

## 函数原型

```python
torch_npu.npu_matmul_compress_dequant(x1, x2, compress_index, bias, scale, *, offsetW=None, offsetX=0) -> Tensor
```

## 参数说明

- **x1**（`Tensor`）：必选参数，矩阵乘左输入，2维，形状[M, K]。数据类型支持int8。M需小于等于256，K为64的整数倍且大于256.
- **x2**（`Tensor`）：必选参数，压缩后的权重重建矩阵，1维，形状[C]，与 compress_index 配合使用，数据类型支持int8。shape大小c的值大于0且不大于k * n。压缩算法为msmodelslim.pytorch.weight_compression
- **compress_index**（`Tensor`）：必选参数，压缩索引，用于从压缩格式还原权重，1维，数据类型支持int8。
- **bias**（`Tensor`）：必选参数，偏置，2维，形状与输出的第二维 N 一致，例如[1, N]。数据类型支持int32。n为64的整数倍且大于等于128。
- **scale**（`Tensor`）：必选参数，反量化 scale，2维，第二维 n 与矩阵乘输出的 N 对应，例如[1, n]。用于将整型结果反量化为 float16，数据类型支持uint64。
- **offsetW**（`Tensor`）：可选参数，权重量化偏移。当前仅支持传入 None，请勿传入具体张量。
- **offsetX**（`int64_t`）：可选参数，输入量化偏移。当前仅支持 0，默认值为 0。

## 返回值说明

- **out**（`Tensor`）：形状 [M, N]，其中 M 为 x1 的第 0 维，N 为 bias 的第 1 维。数据格式为 ND，数据类型为 float16。

## 约束说明

- <term>Atlas 推理系列产品</term>：
  - 该接口支持推理场景下使用。
  - 该接口支持图模式。

## 调用示例

- **对数据进行预处理**

  - **原始权重通过msModelSlim压缩工具生成压缩后的x2、compressIndex以及compressInfo**
  
    使用以下接口时，需对CANN包中msModelSlim压缩工具进行编译，具体操作参考[Gitee msit仓](https://gitee.com/ascend/msit/tree/master/msmodelslim)中msmodelslim/pytorch/weight_compression目录下的README.md。

    ```python
    import torch
    import torch_npu
    from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

    x2 = torch.zeros((k, n), dtype=torch.int8).to("cpu").numpy()
    weight_path = os.path.join(
            os.getcwd(),
            f"weight_2.npy")
    np.save(weight_path, {'weight': x2})

    compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1)
    compressor = Compressor(compress_config, weight_path=weight_path)

    compress_weight, compress_index, compress_info = compressor.run()
    ```

  - **将原始float类型的反量化参数deqscale进行转换， 得到接口需要的uint64数据**

    deqScale原始为float类型，以int32读取并转换为int64

    ```python
    import torch
    import torch_npu
    deq_scale_scale = torch.ones((1, n), dtype=torch.float32)
    deq_scale_uint64 = torch_npu.npu_trans_quant_param(deq_scale_scale.npu()).to(torch.uint64)
    ```

- **调用接口进行计算**

  - **单算子模式调用**

    ```python
    import torch
    import torch_npu

    m, k, n = 16, 256, 128
    x1 = torch.ones((m, k), dtype=torch.int8)
    x2_compressed = torch.from_numpy(compress_weight)
    x2_compressed_idx = torch.from_numpy(compress_index)
    bias = torch.zeros((1, n), dtype=torch.int32)
    custom_output = torch_npu.npu_matmul_compress_dequant(x1.npu(),
                                                        x2_compressed.npu(),
                                                        x2_compressed_idx.npu(),
                                                        bias.npu(),
                                                        deq_scale_uint64)
    ```

  - **图模式调用**

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    device=torch.device(f'npu:0')

    torch_npu.npu.set_device(device)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x1, x2_compressed, x2_compressed_idx, bias, deq_scale_uint64):
            output = torch_npu.npu_matmul_compress_dequant(x1, x2_compressed, x2_compressed_idx, bias, deq_scale_uint64)
            return output

    m, k, n = 16, 256, 128
    x1 = torch.ones((m, k), dtype=torch.int8).to("npu")
    x2_compressed = torch.from_numpy(compress_weight).to("npu")
    x2_compressed_idx = torch.from_numpy(compress_index).to("npu")
    bias = torch.zeros((1, n), dtype=torch.int32).to("npu")

    model = MyModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    output = model(x1, x2_compressed, x2_compressed_idx, bias, deq_scale_uint64)
    print(output)
    ```
