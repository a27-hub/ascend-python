import os
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
import torch
import torch_npu

from torch.testing._internal.common_utils import TestCase
from torch_npu.inductor.ascend_npu_ir.ascned_npu_ir import config

config.GENERATE_LIST = [
    aten.mul,
    aten.add,
    aten.sub,
]

class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        add = a + b
        sub = c - a
        add = torch.acos(add)
        mul = add * sub
        mul = mul + 3
        return mul

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
