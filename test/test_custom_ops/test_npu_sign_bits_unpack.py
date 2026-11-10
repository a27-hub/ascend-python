import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuSignBitsUnpack(TestCase):
    def cpu_op_exec(self, input_np, size, dtype):
        unpack_data = np.unpackbits(input_np, bitorder="little").astype(dtype)
        unpack_data = (unpack_data - 0.5) * 2.0
        return unpack_data.reshape(size, unpack_data.shape[0] // size)

    def npu_op_exec(self, input_npu, size, dtype):
        return torch_npu.npu_sign_bits_unpack(input_npu, size, dtype).cpu().numpy()

    @SupportedDevices(["Ascend950"])
    def test_npu_sign_bits_unpack(self):
        case_list = [
            (1, 1),
            (8, 1),
            (8, 2),
            (17, 1),
            (17, 8),
            (32, 4),
            (33, 3),
            (64, 16),
        ]
        dtype_list = [
            (torch.float16, np.float16),
            (torch.float32, np.float32),
        ]
        for numel, size in case_list:
            input_np = np.random.randint(0, 256, numel, dtype=np.uint8)
            input_npu = torch.from_numpy(input_np).npu()
            for dtype, np_dtype in dtype_list:
                cpu_output = self.cpu_op_exec(input_np, size, np_dtype)
                npu_output = self.npu_op_exec(input_npu, size, dtype)
                self.assertEqual(npu_output.shape, (size, numel * 8 // size))
                self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_sign_bits_unpack_empty(self):
        input_npu = torch.empty(0, dtype=torch.uint8).npu()
        npu_output = torch_npu.npu_sign_bits_unpack(input_npu, 2, torch.float32)
        self.assertEqual(tuple(npu_output.shape), (2, 0))
        self.assertEqual(npu_output.dtype, torch.float32)
        self.assertEqual(npu_output.numel(), 0)

    @SupportedDevices(["Ascend950"])
    def test_npu_sign_bits_unpack_invalid_args(self):
        with self.assertRaisesRegex(RuntimeError, "1-dimensional"):
            torch_npu.npu_sign_bits_unpack(torch.zeros(4, 2, dtype=torch.uint8).npu(), 2, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "only supports torch.uint8"):
            torch_npu.npu_sign_bits_unpack(torch.zeros(8, dtype=torch.int8).npu(), 2, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "less than or equal to zero"):
            torch_npu.npu_sign_bits_unpack(torch.zeros(8, dtype=torch.uint8).npu(), 0, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "multiple of size"):
            torch_npu.npu_sign_bits_unpack(torch.zeros(2, dtype=torch.uint8).npu(), 3, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "must be torch.float32 or torch.float16"):
            torch_npu.npu_sign_bits_unpack(torch.zeros(8, dtype=torch.uint8).npu(), 2, torch.int32)


if __name__ == "__main__":
    run_tests()
