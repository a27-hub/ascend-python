import unittest

import torch
import torch_npu


class TestAllGatherQuantMmHiF4Meta(unittest.TestCase):
    def test_packed_layouts(self):
        for n in (192, 128):
            for transposed in (False, True):
                for gather_output in (False, True):
                    for y_dtype, expected_dtype in ((5, torch.float16), (15, torch.bfloat16)):
                        for scale_dtype in (torch.int32, torch.float32):
                            with self.subTest(n=n, transposed=transposed, gather=gather_output,
                                              y_dtype=y_dtype, scale_dtype=scale_dtype):
                                x1 = torch.empty((64, 128), dtype=torch.uint8, device="meta")
                                x2 = torch.empty((n, 128), dtype=torch.uint8, device="meta")
                                if transposed:
                                    x2 = x2.t()
                                # Square shapes follow TorchAir's (K/2, N) convention.
                                scale2_shape = (4, n, 1) if x2.size(0) == 128 else (n, 4, 1)
                                y, gathered, amax = torch.ops.npu.npu_all_gather_quant_mm(
                                    x1, x2, "meta_test", 2,
                                    x1_scale=torch.empty((64, 4, 1), dtype=scale_dtype, device="meta"),
                                    x2_scale=torch.empty(scale2_shape, dtype=scale_dtype, device="meta"),
                                    x1_dtype=298, x2_dtype=298,
                                    x1_scale_dtype=299, x2_scale_dtype=299,
                                    y_dtype=y_dtype, gather_output=gather_output)
                                self.assertEqual(tuple(y.shape), (128, n))
                                self.assertEqual(y.dtype, expected_dtype)
                                self.assertEqual(tuple(gathered.shape), (128, 128) if gather_output else (0,))
                                self.assertEqual(gathered.dtype, torch.uint8)
                                self.assertEqual(tuple(amax.shape), (0,))
                                self.assertEqual(amax.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
