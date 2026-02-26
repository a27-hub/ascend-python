import copy
import torch
from torch import _VF
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.npu.utils import _is_gte_cann_version


class TestLstmCell(TestCase):

    def test_lstm_cell(self):
        # Check CANN version and set compile mode accordingly
        # aclnn version requires CANN >= 9.0.0, otherwise use aclop
        use_aclnn = _is_gte_cann_version("9.0.0", module="CANN")
        # Save original compile mode
        original_jit_compile = torch.npu.is_jit_compile_false()
        try:
            if use_aclnn:
                # Use aclnn (op_api) mode for CANN >= 9.0.0
                torch.npu.set_compile_mode(jit_compile=False)
            else:
                # Use aclop mode for older CANN versions
                torch.npu.set_compile_mode(jit_compile=True)
            # shape_format:[[dtype, (batch_size, input_size), input_size, hidden_size]
            shape_format = [
                [[np.float16, (32, 64)], 64, 32],
                [[np.float16, (2560, 512)], 512, 256],
                [[np.float32, (32, 64)], 64, 64],
                [[np.float32, (33, 128)], 128, 64],
            ]
            for item in shape_format:
                cpu_lstm = torch.nn.LSTMCell(input_size=item[1], hidden_size=item[2])
                npu_lstm = copy.deepcopy(cpu_lstm).npu()
                cpu_lstm2 = torch.nn.LSTMCell(input_size=item[1], hidden_size=item[2], bias=False)
                npu_lstm2 = copy.deepcopy(cpu_lstm2).npu()

                input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)
                cpu_input1 = torch.from_numpy(input1)
                npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()

                cpu_output_h, cpu_output_c = cpu_lstm(cpu_input1)
                npu_output_h, npu_output_c = npu_lstm(npu_input1)
                cpu_output_h2, cpu_output_c2 = cpu_lstm2(cpu_input1)
                npu_output_h2, npu_output_c2 = npu_lstm2(npu_input1)

                self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                     npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
                self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                     npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

                self.assertRtolEqual(cpu_output_h2.detach().numpy(),
                                     npu_output_h2.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
                self.assertRtolEqual(cpu_output_c2.detach().numpy(),
                                     npu_output_c2.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
        finally:
            # Restore original compile mode
            torch.npu.set_compile_mode(jit_compile=not original_jit_compile)


if __name__ == "__main__":
    run_tests()
