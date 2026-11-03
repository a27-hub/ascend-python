import unittest

import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


# torch.linalg.polar was introduced in torch 2.14; this repo also maintains the v2.1~v2.13
# branches where the API does not exist, so skip at class level via hasattr.
@unittest.skipIf(not hasattr(torch.linalg, "polar"), "requires torch>=2.14")
class TestLinalgPolar(TestCase):
    def _check_polar(self, a, u, h, prec=2e-4):
        # a, u, h are CPU tensors: u must have orthonormal columns, h must be Hermitian positive
        # semi-definite, and the reconstruction must give back a.
        n = a.size(-1)
        eye = torch.eye(n, dtype=a.dtype).expand(a.shape[:-2] + (n, n)).contiguous()
        torch.testing.assert_close(eye, u.mH() @ u, rtol=prec, atol=prec)
        torch.testing.assert_close(h, h.mH(), rtol=prec, atol=prec)
        torch.testing.assert_close(a, u @ h, rtol=prec, atol=prec)
        torch.testing.assert_close(a.mH() @ a, h @ h, rtol=prec, atol=prec)

    def test_linalg_polar(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            for shape in ((1, 1), (2, 2), (4, 4), (5, 3), (2, 5, 3)):
                a_cpu = torch.randn(shape, dtype=dtype)
                u_cpu, h_cpu = torch.linalg.polar(a_cpu)
                u_npu, h_npu = torch.linalg.polar(a_cpu.npu())
                self.assertEqual(u_npu.shape, u_cpu.shape)
                self.assertEqual(h_npu.shape, h_cpu.shape)
                self.assertTrue(u_npu.is_contiguous())
                self.assertTrue(h_npu.is_contiguous())
                torch.testing.assert_close(u_cpu, u_npu.cpu(), rtol=2e-4, atol=2e-4)
                torch.testing.assert_close(h_cpu, h_npu.cpu(), rtol=2e-4, atol=2e-4)
                self._check_polar(a_cpu, u_npu.cpu(), h_npu.cpu())

    def test_linalg_polar_out(self):
        torch.manual_seed(1)
        a_cpu = torch.randn(4, 3)
        a_npu = a_cpu.npu()
        u_out = torch.empty(4, 3, dtype=torch.float32, device="npu")
        h_out = torch.empty(3, 3, dtype=torch.float32, device="npu")
        torch.linalg.polar(a_npu, out=(u_out, h_out))

        u_aten = torch.empty(4, 3, dtype=torch.float32, device="npu")
        h_aten = torch.empty(3, 3, dtype=torch.float32, device="npu")
        torch.ops.aten.linalg_polar.out(a_npu, U=u_aten, H=h_aten)

        self.assertEqual(u_out, u_aten)
        self.assertEqual(h_out, h_aten)
        self._check_polar(a_cpu, u_out.cpu(), h_out.cpu())

    def test_linalg_polar_empty(self):
        for shape in ((2, 0, 0), (3, 0)):
            a_npu = torch.empty(shape, device="npu", dtype=torch.float32)
            u_npu, h_npu = torch.linalg.polar(a_npu)
            expected_h = list(shape)
            expected_h[-2] = shape[-1]
            self.assertEqual(list(u_npu.shape), list(shape))
            self.assertEqual(list(h_npu.shape), expected_h)
            self.assertEqual(u_npu.numel(), 0)
            self.assertEqual(h_npu.numel(), 0)

    def test_linalg_polar_non_contiguous(self):
        torch.manual_seed(2)
        a_cpu = torch.randn(3, 4).t()
        self.assertFalse(a_cpu.is_contiguous())
        u_npu, h_npu = torch.linalg.polar(a_cpu.npu())
        self._check_polar(a_cpu, u_npu.cpu(), h_npu.cpu())

    def test_linalg_polar_dtype(self):
        with self.assertRaisesRegex(RuntimeError, "Expected a floating point or complex tensor"):
            torch.linalg.polar(torch.randint(0, 5, (3, 3)).npu())
        # The adapted aclnnSvd has no kernel for the low precision dtypes.
        with self.assertRaisesRegex(RuntimeError, "only float32, float64, complex64 and complex128"):
            torch.linalg.polar(torch.randn(3, 3).half().npu())

    def test_linalg_polar_invalid_shape(self):
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.polar(torch.randn(3, device="npu"))
        with self.assertRaisesRegex(RuntimeError, "at least as many rows as columns"):
            torch.linalg.polar(torch.randn(2, 3, device="npu"))
        with self.assertRaisesRegex(RuntimeError, "only supports 2 to 8 dimensions"):
            torch.linalg.polar(torch.randn(1, 1, 1, 1, 1, 1, 1, 2, 2, device="npu"))

    def test_linalg_polar_backward(self):
        # Compare with the CPU reference (eigh based formula) for both cotangents and for each of
        # them alone, so that both branches of the VJP are covered.
        torch.manual_seed(3)
        for shape in ((4, 4), (5, 3), (2, 4, 3)):
            a_src = torch.randn(shape)
            u_src, h_src = torch.linalg.polar(a_src)
            g_u = torch.randn(u_src.shape)
            g_h = torch.randn(h_src.shape)
            for gu, gh in ((g_u, g_h), (g_u, None), (None, g_h)):
                a_cpu = a_src.clone().requires_grad_(True)
                u_cpu, h_cpu = torch.linalg.polar(a_cpu)
                ref = torch.autograd.grad((u_cpu, h_cpu), a_cpu, (gu, gh))[0]

                a_npu = a_src.npu().requires_grad_(True)
                u_npu, h_npu = torch.linalg.polar(a_npu)
                got = torch.autograd.grad(
                    (u_npu, h_npu),
                    a_npu,
                    (None if gu is None else gu.npu(), None if gh is None else gh.npu()))[0]

                self.assertTrue(torch.isfinite(got).all().item())
                self.assertRtolEqual(ref.numpy(), got.cpu().numpy(), prec=1e-3)

    def test_linalg_polar_complex_singleton_backward(self):
        torch.manual_seed(5)
        a_src = torch.randn((1, 1), dtype=torch.complex64)
        g_u = torch.randn((1, 1), dtype=torch.complex64)
        g_h = torch.randn((1, 1), dtype=torch.complex64)

        a_cpu = a_src.clone().requires_grad_(True)
        u_cpu, h_cpu = torch.linalg.polar(a_cpu)
        ref = torch.autograd.grad((u_cpu, h_cpu), a_cpu, (g_u, g_h))[0]

        a_npu = a_src.npu().requires_grad_(True)
        u_npu, h_npu = torch.linalg.polar(a_npu)
        got = torch.autograd.grad((u_npu, h_npu), a_npu, (g_u.npu(), g_h.npu()))[0]

        torch.testing.assert_close(ref, got.cpu(), rtol=1e-3, atol=1e-3)

    def test_linalg_polar_forward_ad(self):
        # Exercises the derivatives.yaml "U, H" entry (the JVP formula).
        torch.manual_seed(4)
        a_cpu = torch.randn(4, 3)
        d_cpu = torch.randn(4, 3)
        _, tangents_cpu = torch.func.jvp(torch.linalg.polar, (a_cpu,), (d_cpu,))
        _, tangents_npu = torch.func.jvp(torch.linalg.polar, (a_cpu.npu(),), (d_cpu.npu(),))
        self.assertRtolEqual(tangents_cpu[0].numpy(), tangents_npu[0].cpu().numpy(), prec=1e-3)
        self.assertRtolEqual(tangents_cpu[1].numpy(), tangents_npu[1].cpu().numpy(), prec=1e-3)


if __name__ == "__main__":
    run_tests()
