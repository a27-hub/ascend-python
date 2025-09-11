import unittest
import math
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def npu_attention_update_golden(lse_list, local_out_list, update_type=0):
    assert isinstance(lse_list, (list, tuple)) and isinstance(local_out_list, (list, tuple))
    assert len(lse_list) > 0 and len(local_out_list) > 0
    assert len(lse_list) == len(local_out_list)
    assert update_type == 0

    N = None
    H = None
    lse_cpu = []
    out_cpu = []
    for i, (lse_i, out_i) in enumerate(zip(lse_list, local_out_list)):
        assert isinstance(lse_i, torch.Tensor) and isinstance(out_i, torch.Tensor)
        assert lse_i.dtype == torch.float32, f"lse[{i}] must be float32"
        assert out_i.dtype == torch.float32, f"local_out[{i}] must be float32"
        assert lse_i.dim() == 1, f"lse[{i}] must be 1D [N], got {list(lse_i.shape)}"
        assert out_i.dim() == 2, f"local_out[{i}] must be 2D [N, H], got {list(out_i.shape)}"

        if N is None:
            N = lse_i.size(0)
            H = out_i.size(1)
        else:
            assert lse_i.size(0) == N, f"lse[{i}].size(0) must be {N}"
            assert out_i.size(0) == N and out_i.size(1) == H, f"local_out[{i}] must be [{N}, {H}]"

        lse_cpu.append(lse_i.detach().to("cpu", dtype=torch.float32))
        out_cpu.append(out_i.detach().to("cpu", dtype=torch.float32))

    lse_stack = torch.stack(lse_cpu, dim=0)
    out_stack = torch.stack(out_cpu, dim=0)

    lse_max, _ = torch.max(lse_stack, dim=0)
    exp_terms = torch.exp(lse_stack - lse_max.unsqueeze(0))
    lse_sum = torch.sum(exp_terms, dim=0)
    lse_m = lse_max + torch.log(lse_sum + 1e-20)

    weights = torch.exp(lse_stack - lse_m.unsqueeze(0))
    O = torch.sum(out_stack * weights.unsqueeze(-1), dim=0)
    return O.contiguous()


class TestNpuAttentionUpdate(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def compare(self, a: torch.Tensor, b: torch.Tensor, benchmark: float) -> bool:
        a = a.reshape(-1).cpu()
        b = b.reshape(-1).cpu()
        diff_abs = torch.abs(a - b)
        if diff_abs.numel() == 0:
            return True
        max_diff_abs = diff_abs.max().item()
        if max_diff_abs == 0:
            return True

        rel_error = 0
        abs_error = 0
        for i in range(a.numel()):
            ai = float(a[i].item())
            bi = float(b[i].item())
            diff = abs(ai - bi)

            if ai == 0.0 and bi == 0.0:
                diff_rel_item = 0.0
            elif ai == 0.0 or bi == 0.0:
                diff_rel_item = 1.0
            else:
                diff_rel_item = diff / abs(ai)

            if abs(ai) < 1 and diff > benchmark:
                abs_error += 1
            elif abs(ai) >= 1 and diff_rel_item > benchmark:
                rel_error += 1

            if (rel_error + abs_error) > 10:
                break

        return (rel_error + abs_error) == 0

    def _run_and_check(self, N=4, H=32, K=2, update_type=0, atol=1e-4, rtol=1e-4, benchmark=2**-10):
        assert isinstance(N, int) and isinstance(H, int) and isinstance(K, int)
        assert N > 0 and H > 0 and K > 0
        assert H % 8 == 0 and H < 512

        dtype = torch.float32
        device = 'npu'

        lse_list = [torch.randn(N, dtype=dtype, device=device) for _ in range(K)]
        local_out_list = [torch.randn(N, H, dtype=dtype, device=device) for _ in range(K)]

        out_npu = torch_npu.npu_attention_update(lse_list, local_out_list, update_type)

        out_cpu = npu_attention_update_golden([t.to("cpu") for t in lse_list],
                                              [t.to("cpu") for t in local_out_list],
                                              update_type=update_type)

        self.assertEqual(tuple(out_npu.shape), (N, H))
        self.assertEqual(tuple(out_cpu.shape), (N, H))
        self.assertEqual(out_npu.dtype, dtype)
        self.assertEqual(out_cpu.dtype, dtype)

        out_npu_cpu = out_npu.to("cpu", dtype=torch.float32)
        out_cpu_f32 = out_cpu.to(torch.float32)
        self.assertTrue(torch.allclose(out_npu_cpu, out_cpu_f32, atol=atol, rtol=rtol),
                        f"allclose failed, max_abs={float((out_npu_cpu - out_cpu_f32).abs().max())}")

        self.assertTrue(self.compare(out_cpu_f32, out_npu_cpu, benchmark))

        for t in [out_npu_cpu, out_cpu_f32]:
            self.assertFalse(torch.isnan(t).any().item())
            self.assertFalse(torch.isinf(t).any().item())

    @unittest.skip("skip until CANN is updated to support aclnnAttentionUpdate")
    @SupportedDevices(['Ascend910B'])
    def test_forward_min_case(self):
        cases = [
            (4, 32, 2),
            (8, 64, 2),
            (16, 128, 3),
            (32, 256, 2),
            (64, 64, 4),
            (120, 32, 2),
            (240, 128, 2),
        ]
        for (N, H, K) in cases:
            self._run_and_check(N=N, H=H, K=K, update_type=0)

if __name__ == "__main__":
    run_tests()