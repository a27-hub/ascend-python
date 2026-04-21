import copy
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import cpp_extension_base

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


class TestCustomAdd(TestCase):
    def test_add_custom(self):
        x_cpu = torch.randn([8, 2048], dtype=torch.float16)
        y_cpu = torch.randn([8, 2048], dtype=torch.float16)
        x_npu, y_npu = copy.deepcopy(x_cpu).npu(), copy.deepcopy(y_cpu).npu()
        x_cpu.requires_grad = True
        y_cpu.requires_grad = True
        x_npu.requires_grad = True
        y_npu.requires_grad = True

        # calculate on npu
        output = torch.ops.cpp_extension_base.add_custom(x_npu, y_npu)
        output.backward(output)

        # calculate on cpu
        cpuout = torch.add(x_cpu, y_cpu)
        cpuout.backward(cpuout)

        # compare result
        self.assertRtolEqual(output, cpuout)
        self.assertRtolEqual(x_npu.grad, x_cpu.grad)
        self.assertRtolEqual(y_npu.grad, y_cpu.grad)

    def test_add_custom_ops(self):
        x_cpu = torch.randn([8, 2048], dtype=torch.float16)
        y_cpu = torch.randn([8, 2048], dtype=torch.float16)
        x_npu, y_npu = copy.deepcopy(x_cpu).npu(), copy.deepcopy(y_cpu).npu()
        x_cpu.requires_grad = True
        y_cpu.requires_grad = True
        x_npu.requires_grad = True
        y_npu.requires_grad = True

        # calculate on npu
        output = cpp_extension_base.ops.add_custom(x_npu, y_npu)
        output.backward(output)

        # calculate on cpu
        cpuout = torch.add(x_cpu, y_cpu)
        cpuout.backward(cpuout)

        # compare result
        self.assertRtolEqual(output, cpuout)
        self.assertRtolEqual(x_npu.grad, x_cpu.grad)
        self.assertRtolEqual(y_npu.grad, y_cpu.grad)

if __name__ == "__main__":
    run_tests()
