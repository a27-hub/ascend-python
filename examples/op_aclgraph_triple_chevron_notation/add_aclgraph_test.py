import sys
import os
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch.library import Library

# Load the custom operator library
torch.ops.load_library("libcustom_ops.so")

# Register Meta function
m = Library("ascendc_ops", "IMPL", "Meta")
@torch.library.impl(m, "ascendc_add")
def ascendc_add_meta(a, b):
    return torch.empty_like(a)

# Define a simple model using the custom operation
class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.ascendc_ops.ascendc_add(x, y)

class TestCustomAdd(TestCase):

    # Test single custom operator call
    def test_add_custom_ops(self):
        length = [8, 2048]
        x = torch.randint(low=1, high=100,size=length, device='cpu', dtype=torch.int)
        y = torch.randint(low=1, high=100,size=length, device='cpu', dtype=torch.int)
        output = torch.ops.ascendc_ops.ascendc_add(x.npu(), y.npu()).cpu()
        cpuout = torch.add(x, y)
        self.assertRtolEqual(output, cpuout)

    # Test using torch.npu.NPUGraph
    def test_npugraph(self):
        length = [8, 2048]
        static_x = torch.randint(low=1, high=100,size=length, device='npu:0', dtype=torch.int)
        static_y = torch.randint(low=1, high=100,size=length, device='npu:0', dtype=torch.int)
        static_target = torch.randint(low=1, high=100,size=length, device='npu:0', dtype=torch.int)

        g = torch.npu.NPUGraph()
        model = Model()
        with torch.npu.graph(g):
            static_target = model(static_x, static_y)

        real_x = torch.randint_like(static_x, low=1, high=100)
        real_y = torch.randint_like(static_y, low=1, high=100)
        static_x.copy_(real_x)
        static_y.copy_(real_y)
        # replay
        g.replay()
        cpuout = torch.add(real_x, real_y)
        self.assertRtolEqual(static_target, cpuout)

    # Test using make_graphed_callables
    def test_make_graphed_callables(self):
        length = [8, 2048]
        model = Model().npu()
        x = torch.randint(low=1, high=100,size=length, device='npu', dtype=torch.int)
        y = torch.randint(low=1, high=100,size=length, device='npu', dtype=torch.int)
        model = torch.npu.make_graphed_callables(model,(x, y))

        real_x = torch.randint_like(x, low=1, high=100)
        real_y = torch.randint_like(y, low=1, high=100)
        output = model(real_x, real_y)
        cpuout = torch.add(real_x, real_y)
        self.assertRtolEqual(output, cpuout)

    # Test using the npugraph_ex backend for model compilation
    def test_npugraph_ex_backend(self):
        length = [8, 2048]
        model = Model().npu()
        compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
        x = torch.randint(low=1, high=100,size=length, device='npu', dtype=torch.int)
        y = torch.randint(low=1, high=100,size=length, device='npu', dtype=torch.int)
        output = compiled_model(x, y)
        cpuout = torch.add(x, y)
        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()