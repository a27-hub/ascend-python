import torch
import torch_npu
from torch_npu.op_plugin.dvm import graph_fusion

def test_basic():
    class MyModule(torch.nn.Module):
        def forward(self, x, y):
            z = (x + y) * x
            return torch.sum(z, 0)
    m = MyModule()
    m2 = torch.compile(m, backend="inductor")
    a = torch.normal(0, 0.01, size=(4, 4096), dtype=torch.float).npu()
    b = torch.normal(0, 0.01, size=(4, 4096), dtype=torch.float).npu()
    with torch.no_grad():
        expect = m(a, b)
        output = m2(a, b)
        torch.testing.assert_close(output, expect, rtol=1e-3, atol=1e-3)