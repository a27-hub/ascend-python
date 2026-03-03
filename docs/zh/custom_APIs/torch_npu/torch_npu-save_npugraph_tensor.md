# torch_npu.save_npugraph_tensor

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |
|<term>Atlas 推理系列加速卡产品</term> | √   |

## 功能说明

aclgraph模式下，由于Python原生的print函数包含同步处理，在`torch.compile时`会触发断图，导致图模式下无法使用print观察aclgraph模式执行过程中的tensor值。

当前接口提供了类似原生print特性且不影响aclgraph replay的tensor dump能力，允许将ACLGraph中间节点的tensor数据、数据类型、shape信息保存到指定的pt或bin文件中以便用户观察aclgraph的执行过程，以快速定位问题。

## 函数原型

- 单个tensor dump：
    ```
    torch_npu.save_npugraph_tensor(tensor, save_path=None) -> None
    ```

- 多个tensor dump (tensorlist/tuple)：
    ```
    torch_npu.save_npugraph_tensor(tensors, save_name, save_dir=None, suffix=None) -> None
    ```

## 参数说明

- 单个tensor dump：
    - **tensor** (`Tensor`)：必选参数，用于保存的tensor。
    - **save_path** (`str`)：可选参数，文件保存路径。


- 多个tensor dump：
    - **tensors** (`Tensor`)：必选参数，用于保存的tensorlist/tuple。
    - **save_name** (`str`)：必选参数，tensors的统一保存名称。
    - **save_dir** (`str`)：可选参数，文件保存路径，默认为当前路径。
    - **suffix** (`str`)：可选参数，文件保存后缀，默认为".pt"。

## 返回值说明

`None`

无返回值

## 约束说明

该接口支持在Eager模式和aclgraph模式下使用。

实际的文件路径会加上device序号作为多设备场景下的区分。并且每一个文件拥有一个**唯一标识**，作为多个同名文件的区分。例如当多次尝试写入到"tensor.pt"时，会生成多个文件，分别为"tensor_device_x_0.pt"，"tensor_device_x_1.pt"等。

保存的文件可以配合`torch.load()`接口读取并重建tensor。

- 单个tensor dump接口：
    - **save_path**参数的可选文件后缀为".pt"或".bin"，如果路径不存在会优先尝试创建。

    - 如果不指定文件保存路径则在当前路径创建，默认命名格式为"tensor_" + 当前时间戳的".pt"文件，例如"tensor_2026010100000_device_0.pt"。

- 多个tensor dump：
    - 必须传入tensorlist/tuple，并指定统一保存名称。

    - 被保存的tensors将共用此统一保存名称，并且会带上index作为区分。例如当指定save_name为"inputs"时，每个文件的保存名称为"inputs_0_device_x_0.pt"，"inputs_1_device_x_0.pt"等。

    - 如果未指定"save_dir"时，会使用当前路径。如果路径不存在会尝试创建。

    - "suffix"支持指定文件后缀为".pt"或".bin"，如果未指定文件后缀时，默认后缀为".pt"。


## 单个tensor dump接口调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
    >>> y = torch.add(x, x)
    >>> torch_npu.save_npugraph_tensor(y)
    ```

- 基于`torch.npu.graph`调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor(x, save_path = "x.pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path = "./output.pt")

    graph1.replay()
    ```

- 基于`torch.compile`调用

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = torch.add(x, x)
            torch_npu.save_npugraph_tensor(x, save_path = "/home/dump/output.pt")
            x = torch.add(x, 2)
            torch_npu.save_npugraph_tensor(x)
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- 保存文件解析

    ```python
    >>> x = torch.load("./x_device_0_0.pt")
    >>> print(x)
    >>> print(x.shape)
    ```

## 多个tensor dump接口调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> y = torch.randn([5, 5]).npu()
    >>> z = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor([x, y, z], "tensors", save_dir="./", suffix=".bin")
    ```

- 基于`torch.npu.graph`调用

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor((x, y, z), "tensors")

    graph1.replay()
    ```

- 基于`torch.compile`调用

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, arg1, arg2, arg3):
            model_args = (arg1, arg2, arg3)
            torch_npu.save_npugraph_tensor(model_args, "model_args")
            add_1 = torch.add(arg1, arg2)
            torch_npu.save_npugraph_tensor(x)
            add_2 = torch.add(add_1, arg3)
            return add_2

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model(x, y, z)
    ```

- 保存文件解析：与单tensor dump接口一致。