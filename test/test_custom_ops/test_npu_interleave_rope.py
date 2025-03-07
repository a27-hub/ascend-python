import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuInterleaveRope(TestCase):
    def interleave_rope_numpy(self, x: torch.tensor, cos: torch.tensor, sin: torch.tensor) -> tuple:
        dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
            sin = sin.to(dtype=torch.float32)
            cos = cos.to(dtype=torch.float32)

        x_numpy = x.numpy()
        cos_numpy = cos.numpy()
        sin_numpy = sin.numpy()

        x_numpy = x_numpy.reshape([32, 32, 1, 64])
        cos_numpy = cos_numpy.reshape([32, 1, 1, 64])
        sin_numpy = sin_numpy.reshape([32, 1, 1, 64])

        q_numpy = x_numpy.reshape(32, 32, 1, 32, 2).transpose(0, 1, 2, 4, 3).reshape(32, 32, 1, 64)
        q_embed_numpy = (q_numpy * cos_numpy) + (self.rotate_half(q_numpy) * sin_numpy)

        q_embed = torch.tensor(q_embed_numpy, dtype=dtype)

        return q_embed.cpu()

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate((-x2, x1), axis=-1)
        # return torch.cat((-x2, x1), dim=-1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_interleave_rope_1(self, device="npu"):
        print("************************** test_npu_interleave_rope_1 begin **************************\n")
        x = torch.randn(32, 32, 1, 64, dtype = torch.float16)
        cos = torch.randn(32, 1, 1, 64, dtype = torch.float16)
        sin = torch.randn(32, 1, 1, 64, dtype = torch.float16)

        # benchmark
        out_cpu = self.interleave_rope_numpy(x, cos, sin)
        print("************************** golden begin**************************\n")
        print(f"out is: {out_cpu}, dtype is: {out_cpu.dtype}")
        print("************************** golden end **************************\n\n\n")

        # test
        out_npu = torch_npu.npu_interleave_rope(x.npu(), cos.npu(), sin.npu())
        print("************************** tets begin**************************\n")
        print(f"out is: {out_npu.cpu()}, dtype is: {out_npu.cpu().dtype}")
        print("************************** test end **************************\n\n\n")

        # compare
        self.assertRtolEqual(out_cpu, out_npu.cpu())


if __name__ == "__main__":
    run_tests()
