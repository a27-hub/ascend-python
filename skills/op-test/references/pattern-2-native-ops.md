# Pattern 2: cpu/npu Dual-Path Comparison

For NPU implementations of `torch.xxx` native operators — directly compare CPU and NPU outputs.

## Basic Structure

```python
from torch_npu.testing.common_utils import create_common_tensor

class TestOperatorName(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.numpy()

    def npu_op_exec(self, input1):
        output = torch.op_name(input1)
        return output.to("cpu").numpy()

    def test_op_fp16(self, device="npu"):
        format_list = [0, 3]
        shape_list = [[5], [5, 10], [1, 3, 2]]
        shape_format = [[np.float16, i, j] for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)  # use fp32 for CPU reference of fp16 ops
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
```

## Reference Examples

- `test/test_base_ops/test_abs.py`
- `test/test_base_ops/test_npu_addmm.py`
