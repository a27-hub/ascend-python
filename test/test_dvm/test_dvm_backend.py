import torch
import torch_npu

from torch._dynamo.backends.registry import register_backend
from torch.testing._internal.common_utils import TestCase

from torch_npu.op_plugin.dvm.dvm_build import DvmBackend


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        a = torch.broadcast_to(a, b.shape)
        add = a + b
        sub = c - a
        add = torch.acos(add)
        mul = add * sub
        mul = mul + 3
        mul = mul.type(torch.float32)
        return mul


register_backend(name="dvm", compiler_fn=DvmBackend())


class TestAtbFakeTensor(TestCase):
    def test_basic_partitioning(self):
        a = torch.normal(0, 0.01, size=(4096, 1), dtype=torch.float16).npu()
        b = torch.normal(0, 0.01, size=(4096, 20480),
                         dtype=torch.float16).npu()
        c = torch.normal(0, 0.01, size=(1, 20480), dtype=torch.float16).npu()
        model = TestModule()
        dvm_compiled_model = torch.compile(model, backend="dvm")
        with torch.no_grad():
            expect = model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
