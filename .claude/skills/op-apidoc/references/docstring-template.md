# Python Docstring Template for _op_plugin_docs.py

Append a `_add_torch_npu_docstr` call to `codegen/templates/_op_plugin_docs.py`.

## Template

```python
_add_torch_npu_docstr(
    "npu_{operator_name}",
    """
功能描述:
算子功能: {one sentence describing the operator's core function}
计算公式: (omit if no formula)
{LaTeX formula converted to plain text}

接口原型:
torch_npu.npu_{operator_name}({parameter_list_with_types}) -> {return_type}

参数说明:
{param1_name}: {type}类型, {description}, {required/optional}, 数据类型支持{dtypes}, 数据格式支持{formats}. {additional constraints}.
{param2_name}: ...
*(position-keyword separator, omit description if not needed)

输出说明：
{output1_name}: {type}类型, {description}. {shape rule}. {dtype rule}.
{output2_name}: ... (omit if single return)

约束说明:
该接口支持{inference/training}场景下使用.
该接口{支持/不支持}图模式.
{additional constraints}

支持的PyTorch版本
{list supported versions, e.g. PyTorch 2.5, PyTorch 2.4, etc.}

支持的型号:
{list supported products}

调用示例:
单算子模式调用
{example description}
import torch
import torch_npu

{setup code}
{operator call}
{output}

图模式调用
{example description}
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

{setup code}

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, {params}):
        return torch_npu.npu_{operator_name}({params})

model = Model().npu()
model = torch.compile(model, backend=npu_backend, dynamic=False)
{output} = model({inputs})
"""
)
```

## Key Differences from Markdown Documentation

- **No Markdown syntax**: Use plain text instead of `##`, `$...$`, tables, or `` `code` ``
- **Section labels**: Use `label:` format (e.g., `功能描述:`, `参数说明:`, `约束说明:`)
- **Extra sections**: Include `支持的PyTorch版本` and `支持的型号` which are not in the Markdown template
- **No interactive prompt**: Call examples use plain code blocks, not `>>>` style
- **Tables not supported**: Convert any table content to text descriptions

## Placement

Append before the last existing `_add_torch_npu_docstr` call in the file.
