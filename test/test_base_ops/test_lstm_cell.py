import copy
import torch
from torch import _VF
import numpy as np
import torch_npu

from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests


class TestLstmCell(TestCase):

    def test_lstm_cell(self):
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

    def _cpu_to_exec(self, input_data, weight_ih, weight_hh, h0_data, c0_data, bias_ih, bias_hh):
        """ cpu lstm cell """
        weight_ih = weight_ih.transpose(0, 1)
        weight_hh = weight_hh.transpose(0, 1)
        h_out, c_out = _VF.lstm_cell(input_data, (h0_data, c0_data),
                                     weight_ih, weight_hh, bias_ih, bias_hh)
        return h_out, c_out

    def _npu_to_exec(self, input_data, weight_ih, weight_hh, h0_data, c0_data, bias_ih, bias_hh):
        """ npu lstm cell """
        ig = torch.matmul(input_data, weight_ih.t())
        hg = torch.matmul(h0_data, weight_hh.t())
        h_out, c_out, _ = torch.ops.aten._thnn_fused_lstm_cell(ig, hg, c0_data, bias_ih, bias_hh)
        return h_out, c_out

    def test_thnn_fused_lstm_cell(self):
        """ test thnn_fused_lstm_cell"""
        input_size = 8
        hidden_size = 7
        batch_size = 3

        input_shape = (batch_size, input_size)
        h0_shape = (batch_size, hidden_size)
        c0_shape = (batch_size, hidden_size)

        _, input_data = create_common_tensor(["float16", 29, input_shape], -1, 1)
        _, h0_data = create_common_tensor(["float16", 29, h0_shape], -1, 1)
        _, c0_data = create_common_tensor(["float16", 29, c0_shape], -1, 1)
        _, weight_ih = create_common_tensor(["float16", 2, (input_size, 4 * hidden_size)], -1, 1)
        _, weight_hh = create_common_tensor(["float16", 2, (hidden_size, 4 * hidden_size)], -1, 1)
        _, bias_ih = create_common_tensor(["float16", 2, (4 * hidden_size)], -1, 1)
        _, bias_hh = create_common_tensor(["float16", 2, (4 * hidden_size)], -1, 1)
        # cpu
        cpu_h_out, cpu_c_out = self._cpu_to_exec(input_data, weight_ih, weight_hh, h0_data, c0_data,
                                                 bias_ih, bias_hh)
        cpu_h_out = cpu_h_out.cpu().detach().numpy()
        cpu_c_out = cpu_c_out.cpu().detach().numpy()

        npu_h_out, npu_c_out = self._npu_to_exec(input_data, weight_ih, weight_hh, h0_data, c0_data,
                                                 bias_ih, bias_hh)
        npu_h_out = npu_h_out.cpu().detach().numpy()
        npu_c_out = npu_c_out.cpu().detach().numpy()
        self.assertRtolEqual(cpu_h_out, npu_h_out)
        self.assertRtolEqual(cpu_c_out, npu_c_out)

if __name__ == "__main__":
    run_tests()
