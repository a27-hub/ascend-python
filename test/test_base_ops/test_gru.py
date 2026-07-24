import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    def test_gru_fp32(self):
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            # single layer, unidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            # single layer, unidirectional, with bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, True],
            # single layer, bidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, False],
            # two layers, unidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (2, 2, 3)], 4, 3, 2, False, True, False],
            # two layers, bidirectional, with bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
            # single layer, unidirectional, without bias, batch_first=False
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, False, False],
            # single layer, bidirectional, without bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, False, True],
            # two layers, unidirectional, without bias, batch_first=True
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 2, False, False, True],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            # Ascend fp32 isn't enough precision, relaxation of precision requirement temporary
            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy(), prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy(), prec=1.e-1)

    def test_gru_fp16(self):
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, True, False],
            [[np.float16, (2, 3, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, True, True],
            [[np.float16, (3, 2, 4)], [np.float16, (2, 2, 3)], 4, 3, 1, True, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (2, 2, 3)], 4, 3, 2, False, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (4, 2, 3)], 4, 3, 2, True, True, False],
            [[np.float16, (3, 2, 4)], [np.float16, (1, 2, 3)], 4, 3, 1, False, False, False],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0.astype(np.float32))
            npu_h0 = torch.from_numpy(h0).npu()

            # CPU keeps fp32 for higher precision reference; NPU uses fp16
            npu_gru = npu_gru.to(torch.float16)
            npu_input1 = npu_input1.to(torch.float16)
            npu_h0 = npu_h0.to(torch.float16)

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            self.assertRtolEqual(cpu_output_y.detach().numpy().astype(np.float16),
                                 npu_output_y.cpu().detach().numpy())
            self.assertRtolEqual(cpu_output_h.detach().numpy().astype(np.float16),
                                 npu_output_h.cpu().detach().numpy())

    def test_gru_eval_mode(self):
        # Test inference mode (train=False)
        shape_format = [
            # [input, h0, input_size, hidden_size, num_layers, bidirectional, bias, batch_first]
            [[np.float32, (3, 2, 4)], [np.float32, (1, 2, 3)], 4, 3, 1, False, True, False],
            [[np.float32, (2, 3, 4)], [np.float32, (2, 2, 3)], 4, 3, 1, True, True, True],
            [[np.float32, (3, 2, 4)], [np.float32, (4, 2, 3)], 4, 3, 2, True, True, False],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            cpu_gru.eval()
            npu_gru.eval()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            with torch.no_grad():
                cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
                npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy(), prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy(), prec=1.e-1)


if __name__ == "__main__":
    run_tests()
