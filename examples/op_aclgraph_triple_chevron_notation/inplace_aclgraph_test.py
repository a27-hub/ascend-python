import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch.library import Library

# Load the custom operator library
torch.ops.load_library("libcustom_ops.so")

# Register Meta function
m = Library("ascendc_ops", "IMPL", "Meta")
@torch.library.impl(m, "ascendc_trig")
def ascendc_trig_meta(x, sin_out, cos_out):
    return torch.empty_like(x)

# Define a simple model using the custom operation
class Model(torch.nn.Module):
    def forward(self,x, sin_out, cos_out):
        tan_out = torch.ops.ascendc_ops.ascendc_trig(x, sin_out, cos_out)
        return tan_out

class TestCustomAdd(TestCase):

    # Test single custom operator call
    def test_trig_inplace_ops(self):
        length = [8, 2048]
        x = torch.rand(length, device='npu', dtype=torch.float32)
        sin_out = torch.empty_like(x)
        cos_out = torch.empty_like(x)
        tan_out = torch.ops.ascendc_ops.ascendc_trig(x, sin_out, cos_out)
        self.check_res(x, sin_out, cos_out, tan_out)

    # torch.npu.NPUGraph()使能方式
    def test_npugraph(self):
        length = [8, 2048]
        static_x =  torch.rand(length, device='npu', dtype=torch.float32)
        static_sin_out = torch.rand(length, device='npu', dtype=torch.float32)
        static_cos_out = torch.rand(length, device='npu', dtype=torch.float32)
        static_tan_out = torch.rand(length, device='npu', dtype=torch.float32)

        g = torch.npu.NPUGraph()
        model = Model()
        with torch.npu.graph(g):
            static_tan_out = model(static_x, static_sin_out, static_cos_out)

        real_x = torch.rand_like(static_x)
        real_sin_out = torch.rand_like(static_sin_out)
        real_cos_out = torch.rand_like(static_cos_out)

        static_x.copy_(real_x)
        static_sin_out.copy_(real_sin_out)
        static_cos_out.copy_(real_cos_out)
        # replay
        g.replay()
        self.check_res(real_x, static_sin_out, static_cos_out, static_tan_out)
        
    # Test using torch.npu.NPUGraph
    def test_make_graphed_callables(self):
        length = [8, 2048]
        model = Model().npu()
        x = torch.rand(length, device='npu', dtype=torch.float32)
        sin_out = torch.empty_like(x)
        cos_out = torch.empty_like(x)
        model = torch.npu.make_graphed_callables(model,(x, sin_out, cos_out))

        real_x = torch.rand_like(x)
        real_tan_out = model(real_x, sin_out, cos_out)
        self.check_res(real_x, sin_out, cos_out, real_tan_out)

    # Test using make_graphed_callables
    def test_npugraph_ex_backend(self):
        length = [8, 2048]
        model = Model().npu()
        compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
        x = torch.rand(length, device='npu', dtype=torch.float32)
        sin_out = torch.empty_like(x)
        cos_out = torch.empty_like(x)
        tan_out = compiled_model(x, sin_out, cos_out)
        self.check_res(x, sin_out, cos_out, tan_out)
     
    # Test using the npugraph_ex backend for model compilation
    def check_res(self,x, sin_out, cos_out, tan_out):
        cpu_sin_out = torch.sin(x)
        cpu_cos_out = torch.cos(x)
        cpu_tan_out = torch.tan(x)
        self.assertRtolEqual(sin_out, cpu_sin_out)
        self.assertRtolEqual(cos_out, cpu_cos_out)
        self.assertRtolEqual(tan_out, cpu_tan_out)


if __name__ == "__main__":
    run_tests()