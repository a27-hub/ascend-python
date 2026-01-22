import torch
import torch_npu
import os
from torch_npu.testing.testcase import TestCase, run_tests

功能UT
class TestLogSpace(TestCase):

    def supported_op_exec(self, start, end, steps, base, dtype=torch.float32):
        # linspace 是标准算子
        exponents = torch.linspace(start, end, steps, dtype=dtype)
        output = torch.pow(base, exponents)
        return output.cpu().detach()

    def custom_op_exec(self, start, end, steps, base, dtype=torch.float32):
        output = torch.logspace(
            start=start,
            end=end,
            steps=steps,
            base=base,
            device="npu",
            dtype=dtype
        )
        return output.cpu().detach()

    def test_logspace(self):
        start = 0
        end = 3
        steps = 10
        base = 10.0

        supported_output = self.supported_op_exec(start, end, steps, base)
        custom_output = self.custom_op_exec(start, end, steps, base)

        self.assertRtolEqual(supported_output, custom_output)

    # onnx测试用例
    def test_wrapper_npu_logspace(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return torch.logspace(
                    start=0,
                    end=3,
                    steps=10,
                    base=10.0,
                    device="npu"
                )

        model = Model().to("npu")

        dummy = torch.tensor(0).npu()
        onnx_model_name = "model_npu_logspace.onnx"

        torch.onnx.export(
            model,
            dummy,                  # 必须传入 dummy
            onnx_model_name,
            opset_version=13,
            input_names=["input"],
            output_names=["output"]
        )

        assert os.path.isfile(onnx_model_name)


    # script用例
    def test_script_logspace(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self):
                return torch.logspace(
                    start=1,
                    end=4,
                    steps=8,
                    base=2.0,
                    device="npu"
                )

        model = NpuModel().to("npu")

        output1 = model()

        script_model = torch.jit.script(model)
        output2 = script_model()

        self.assertRtolEqual(
            output1.cpu().detach(),
            output2.cpu().detach()
        )

    # trace 用例
    def test_trace_logspace(self):
        class NpuModel(torch.nn.Module):
            def __init__(self):
                super(NpuModel, self).__init__()

            def forward(self, x):
                return torch.logspace(
                    start=-2,
                    end=2,
                    steps=5,
                    base=10.0,
                    device="npu"
                )

        model = NpuModel().to("npu")

        dummy = torch.tensor(0).npu()
        output1 = model(dummy)  # 这里必须传入 dummy

        trace_model = torch.jit.trace(model, dummy)

        output2 = trace_model(dummy)

        self.assertRtolEqual(
            output1.cpu().detach(),
            output2.cpu().detach()
        )

if name == "main":
    run_tests()