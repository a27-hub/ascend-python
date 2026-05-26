import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinalgQr(TestCase):
    def test_linalg_qr_functional_strides(self):
        shape_modes = [
            ((5, 3), "reduced"),
            ((2, 5, 3), "complete"),
        ]

        for shape, mode in shape_modes:
            cpu_input = torch.randn(shape, dtype=torch.float32)
            npu_input = cpu_input.npu()

            cpu_q, cpu_r = torch.linalg.qr(cpu_input, mode=mode)
            npu_q, npu_r = torch.linalg.qr(npu_input, mode=mode)

            self.assertEqual(cpu_q.shape, npu_q.shape)
            self.assertEqual(cpu_r.shape, npu_r.shape)
            self.assertEqual(cpu_q.stride(), npu_q.stride())
            self.assertEqual(cpu_r.stride(), npu_r.stride())

    def test_linalg_qr_r_mode_empty_q_and_r_strides(self):
        cpu_input = torch.randn(4, 6, dtype=torch.float32)
        npu_input = cpu_input.npu()

        cpu_q, cpu_r = torch.linalg.qr(cpu_input, mode="r")
        npu_q, npu_r = torch.linalg.qr(npu_input, mode="r")

        self.assertEqual(cpu_q.shape, npu_q.shape)
        self.assertEqual(cpu_q.stride(), npu_q.stride())
        self.assertEqual(cpu_r.shape, npu_r.shape)
        self.assertEqual(cpu_r.stride(), npu_r.stride())

    def test_linalg_qr_rejects_non_float_or_complex_input(self):
        npu_input = torch.ones(3, 3, dtype=torch.int32, device="npu")

        with self.assertRaisesRegex(RuntimeError, "Expected a floating point or complex tensor"):
            torch.linalg.qr(npu_input)

    def test_linalg_qr_invalid_mode_error_matches_native(self):
        npu_input = torch.randn(3, 3, dtype=torch.float32, device="npu")

        with self.assertRaisesRegex(RuntimeError, "qr received unrecognized mode 'invalid'"):
            torch.linalg.qr(npu_input, mode="invalid")

    def test_linalg_qr_out_with_non_meta_strides(self):
        cpu_input = torch.randn(2, 5, 3, dtype=torch.float32)
        npu_input = cpu_input.npu()

        q_out = torch.empty((2, 5, 3), dtype=torch.float32, device="npu")
        r_out = torch.empty((2, 3, 3), dtype=torch.float32, device="npu")

        q, r = torch.linalg.qr(npu_input, mode="reduced", out=(q_out, r_out))

        self.assertEqual(q.data_ptr(), q_out.data_ptr())
        self.assertEqual(r.data_ptr(), r_out.data_ptr())
        self.assertRtolEqual(torch.matmul(q, r).cpu(), cpu_input, prec=1e-3)


if __name__ == "__main__":
    run_tests()
