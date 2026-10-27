import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRmsNorm(TestCase):
    # normalized_shape equals the trailing dims of these shapes
    shape_format = [
        [np.float32, 0, (64, 10)],
        [np.float32, 0, (256, 2048, 7, 7)],
        [np.float32, 0, (32, 1, 3, 3)],
        [np.float32, 0, (10, 128)],
    ]

    def cpu_op_exec(self, input1, normalized_shape, weight=None, eps=None):
        output = F.rms_norm(input1, normalized_shape, weight=weight, eps=eps)
        return output

    def npu_op_exec(self, input1, normalized_shape, weight=None, eps=None):
        weight_npu = weight.npu() if weight is not None else None
        output = F.rms_norm(input1, normalized_shape, weight=weight_npu, eps=eps)
        return output.cpu()

    def test_rms_norm_fp32_shape_format(self):
        for item in self.shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            normalized_shape = item[2][1:]
            cpu_weight = torch.rand(normalized_shape, dtype=torch.float32)
            for eps in (None, 1e-5):
                cpu_output = self.cpu_op_exec(cpu_input, normalized_shape, cpu_weight, eps)
                npu_output = self.npu_op_exec(npu_input, normalized_shape, cpu_weight, eps)
                self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_rms_norm_float16(self):
        shape_format = [
            [np.float16, 0, (64, 10)],
            [np.float16, 0, (10, 128)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            normalized_shape = item[2][1:]
            cpu_weight = torch.rand(normalized_shape, dtype=torch.float32)
            # fp32 golden: CPU half-precision math is not accurate enough as
            # the reference for the NPU fused kernel (fp32 accumulation)
            cpu_output = self.cpu_op_exec(cpu_input.float(), normalized_shape, cpu_weight)
            npu_output = self.npu_op_exec(npu_input, normalized_shape, cpu_weight.half())
            torch.testing.assert_close(
                npu_output.float(), cpu_output, rtol=2e-3, atol=1e-2)

    def test_rms_norm_bfloat16(self):
        cpu_input = torch.rand(16, 128, dtype=torch.bfloat16) - 0.5
        cpu_weight = torch.rand(128, dtype=torch.bfloat16)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        npu_output = self.npu_op_exec(npu_input, (128,), cpu_weight)
        torch.testing.assert_close(
            npu_output.float(), cpu_output.float(), rtol=2e-2, atol=2e-2)

    def test_rms_norm_without_weight(self):
        cpu_input, npu_input = create_common_tensor([np.float32, 0, (16, 128)], 1, 100)
        cpu_output = self.cpu_op_exec(cpu_input, (128,))
        npu_output = self.npu_op_exec(npu_input, (128,))
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_rms_norm_1d_input(self):
        # 1-D input is folded to 2-D in the fused kernel
        cpu_input = torch.rand(128, dtype=torch.float32)
        cpu_weight = torch.rand(128, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        npu_output = self.npu_op_exec(npu_input, (128,), cpu_weight)
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_rms_norm_multi_dim_normalized_shape(self):
        # multi-dim normalized_shape is folded to 2-D in the fused kernel
        cpu_input = torch.rand(2, 5, 3, 4, dtype=torch.float32)
        cpu_weight = torch.rand(3, 4, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, (3, 4), cpu_weight)
        npu_output = self.npu_op_exec(npu_input, (3, 4), cpu_weight)
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_rms_norm_noncontiguous_input(self):
        cpu_input = torch.rand(8, 128, dtype=torch.float32).t()
        cpu_weight = torch.rand(8, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, (8,), cpu_weight)
        npu_output = self.npu_op_exec(npu_input, (8,), cpu_weight)
        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_rms_norm_weight_dtype_mismatch(self):
        # input/weight dtype mismatch falls back to the composite path
        cpu_input = torch.rand(16, 128, dtype=torch.float16)
        cpu_weight = torch.rand(128, dtype=torch.float32)
        npu_input = cpu_input.npu()
        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        npu_output = self.npu_op_exec(npu_input, (128,), cpu_weight)
        self.assertRtolEqual(
            cpu_output.detach().float().numpy(), npu_output.detach().float().numpy())

    def test_rms_norm_backward_fp32(self):
        cpu_input = torch.rand(16, 128, dtype=torch.float32, requires_grad=True)
        cpu_weight = torch.rand(128, dtype=torch.float32, requires_grad=True)
        npu_input = cpu_input.detach().clone().npu().requires_grad_(True)
        npu_weight = cpu_weight.detach().clone().npu().requires_grad_(True)

        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        # keep the NPU output on device: backward runs on the NPU graph and
        # the incoming grad must be on the same device as the output
        npu_output = F.rms_norm(npu_input, (128,), npu_weight)
        grad = torch.rand_like(cpu_output)
        cpu_output.backward(grad)
        npu_output.backward(grad.npu())

        self.assertRtolEqual(cpu_input.grad.detach().numpy(), npu_input.grad.detach().cpu().numpy())
        self.assertRtolEqual(cpu_weight.grad.detach().numpy(), npu_weight.grad.detach().cpu().numpy())

    def test_rms_norm_backward_input_only(self):
        # weight does not require grad, covers output_mask[1] = False branch
        cpu_input = torch.rand(16, 128, dtype=torch.float32, requires_grad=True)
        cpu_weight = torch.rand(128, dtype=torch.float32)
        npu_input = cpu_input.detach().clone().npu().requires_grad_(True)
        npu_weight = cpu_weight.npu()

        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        npu_output = F.rms_norm(npu_input, (128,), npu_weight)
        grad = torch.rand_like(cpu_output)
        cpu_output.backward(grad)
        npu_output.backward(grad.npu())

        self.assertRtolEqual(cpu_input.grad.detach().numpy(), npu_input.grad.detach().cpu().numpy())

    def test_rms_norm_backward_float16(self):
        cpu_input = torch.rand(16, 128, dtype=torch.float32, requires_grad=True)
        cpu_weight = torch.rand(128, dtype=torch.float32, requires_grad=True)
        npu_input = cpu_input.detach().clone().half().npu().requires_grad_(True)
        npu_weight = cpu_weight.detach().clone().half().npu().requires_grad_(True)

        cpu_output = self.cpu_op_exec(cpu_input, (128,), cpu_weight)
        npu_output = F.rms_norm(npu_input, (128,), npu_weight)
        grad = torch.rand_like(cpu_output)
        cpu_output.backward(grad)
        npu_output.backward(grad.half().npu())

        # fp16 graph vs fp32 golden: same tolerance policy as the fp16
        # forward test (fp16 rounding + fp32 accumulation differences)
        torch.testing.assert_close(
            npu_input.grad.detach().cpu().float(), cpu_input.grad,
            rtol=2e-3, atol=1e-2)
        torch.testing.assert_close(
            npu_weight.grad.detach().cpu().float(), cpu_weight.grad,
            rtol=2e-3, atol=1e-2)

    def test_fused_rms_norm_rstd(self):
        # aten::_fused_rms_norm directly: check y/rstd shape and dtype
        cpu_input, npu_input = create_common_tensor([np.float32, 0, (16, 128)], 1, 100)
        cpu_weight = torch.rand(128, dtype=torch.float32)
        npu_weight = cpu_weight.npu()

        npu_y, npu_rstd = torch.ops.aten._fused_rms_norm(npu_input, [128], npu_weight, None)
        cpu_y = self.cpu_op_exec(cpu_input, (128,), cpu_weight)

        self.assertEqual(npu_y.dtype, torch.float32)
        self.assertEqual(npu_rstd.dtype, torch.float32)
        self.assertEqual(list(npu_rstd.shape), [16, 1])
        self.assertRtolEqual(cpu_y.detach().numpy(), npu_y.cpu().detach().numpy())

    def test_rms_norm_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            npu_x = torch.randn(16, 128, dtype=torch.float32).npu()
            npu_weight = torch.randn(128, dtype=torch.float32).npu()

            y, rstd = torch.ops.aten._fused_rms_norm(npu_x, [128], npu_weight, None)
            self.assertEqual(y.shape, torch.Size([16, 128]))
            self.assertEqual(y.dtype, torch.float32)
            self.assertEqual(rstd.shape, torch.Size([16, 1]))
            self.assertEqual(rstd.dtype, torch.float32)

            # 1-D input: rstd is stat_shape (input dims with the trailing
            # normalized dims set to 1) — aligned with CUDA, no fold special case
            npu_x_1d = torch.randn(128, dtype=torch.float32).npu()
            y_1d, rstd_1d = torch.ops.aten._fused_rms_norm(npu_x_1d, [128], npu_weight, None)
            self.assertEqual(y_1d.shape, torch.Size([128]))
            self.assertEqual(rstd_1d.shape, torch.Size([1]))

            out = F.rms_norm(npu_x, (128,), npu_weight)
            self.assertEqual(out.shape, torch.Size([16, 128]))
            self.assertEqual(out.dtype, torch.float32)

    def test_rms_norm_backward_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            npu_x = torch.randn(16, 128, dtype=torch.float32).npu()
            npu_weight = torch.randn(128, dtype=torch.float32).npu()
            npu_x.requires_grad = True
            npu_weight.requires_grad = True

            out = F.rms_norm(npu_x, (128,), npu_weight)
            out.sum().backward()
            self.assertEqual(npu_x.grad.shape, torch.Size([16, 128]))
            self.assertEqual(npu_weight.grad.shape, torch.Size([128]))
            self.assertEqual(npu_weight.grad.dtype, torch.float32)

            # direct backward primitive: exercises the registered backward
            # Meta (shapes/dtype and the output_mask[0]=False undefined branch)
            y, rstd = torch.ops.aten._fused_rms_norm(npu_x, [128], npu_weight, None)
            dy = torch.randn(16, 128, dtype=torch.float32).npu()
            dx, dgamma = torch.ops.aten._fused_rms_norm_backward(
                dy, npu_x, [128], rstd, npu_weight, [True, True])
            self.assertEqual(dx.shape, torch.Size([16, 128]))
            self.assertEqual(dgamma.shape, torch.Size([128]))
            self.assertEqual(dgamma.dtype, torch.float32)
            dx2, dgamma2 = torch.ops.aten._fused_rms_norm_backward(
                dy, npu_x, [128], rstd, npu_weight, [False, True])
            self.assertTrue(dx2 is None or not dx2.defined())
            self.assertEqual(dgamma2.shape, torch.Size([128]))

    def test_rms_norm_fp64_unsupported(self):
        # fp64: aclnnRmsNorm/Grad do not support DOUBLE (CANN docs) — the fused
        # path rejects fp64 with a readable error at the forward shell, so
        # training never reaches an unusable backward.
        # NOTE: build-dependent — torch builds without the derivatives formula
        # decompose F.rms_norm into small ops (which support fp64) and the
        # shell is never reached; this case asserts the fused-kernel behavior.
        npu_input = torch.randn(4, 8, dtype=torch.float64).npu()
        npu_weight = torch.randn(8, dtype=torch.float64).npu()
        with self.assertRaisesRegex(RuntimeError, "does not support DOUBLE"):
            F.rms_norm(npu_input, (8,), npu_weight)

    def test_rms_norm_empty_input(self):
        # M=0: the shell guard skips the kernel — forward returns an empty
        # Tensor, backward dgamma must be zeros (no gradient pollution)
        npu_input = torch.randn(0, 128, dtype=torch.float32).npu().requires_grad_(True)
        npu_weight = torch.randn(128, dtype=torch.float32).npu().requires_grad_(True)
        npu_output = F.rms_norm(npu_input, (128,), npu_weight)
        self.assertEqual(npu_output.shape, torch.Size([0, 128]))
        npu_output.sum().backward()
        self.assertEqual(npu_weight.grad.shape, torch.Size([128]))
        self.assertTrue(torch.all(npu_weight.grad == 0).item())

    def test_rms_norm_backward_fold(self):
        # backward fold branches: multi-dim ns and 1-D input (shell
        # rstd.reshape({M,1}) + dx/dgamma shape-restore paths)
        for shape, ns in (((2, 5, 3, 4), (3, 4)), ((128,), (128,))):
            cpu_input = torch.randn(shape, dtype=torch.float32, requires_grad=True)
            cpu_weight = torch.randn(ns, dtype=torch.float32, requires_grad=True)
            npu_input = cpu_input.detach().clone().npu().requires_grad_(True)
            npu_weight = cpu_weight.detach().clone().npu().requires_grad_(True)
            cpu_output = F.rms_norm(cpu_input, ns, cpu_weight)
            npu_output = F.rms_norm(npu_input, ns, npu_weight)
            grad = torch.rand_like(cpu_output)
            cpu_output.backward(grad)
            npu_output.backward(grad.npu())
            self.assertEqual(npu_input.grad.shape, cpu_input.grad.shape)
            self.assertRtolEqual(
                npu_input.grad.detach().cpu().numpy(), cpu_input.grad.numpy())
            self.assertEqual(npu_weight.grad.shape, cpu_weight.grad.shape)
            self.assertRtolEqual(
                npu_weight.grad.detach().cpu().numpy(), cpu_weight.grad.numpy())

    def test_rms_norm_backward_output_mask_eager(self):
        # eager direct primitive: the output_mask contract is enforced by the
        # real shell (dx undefined when mask[0]=False, dgamma undefined when
        # mask[1]=False) — not just by the Meta registration.
        # NOTE: build-dependent — requires the fused backward kernel to be
        # registered (current wheel decomposes; this asserts fused-kernel
        # behavior).
        npu_input = torch.randn(4, 8, dtype=torch.float32).npu()
        npu_weight = torch.randn(8, dtype=torch.float32).npu()
        y, rstd = torch.ops.aten._fused_rms_norm(npu_input, [8], npu_weight, None)
        dy = torch.randn(4, 8, dtype=torch.float32).npu()
        dx, dgamma = torch.ops.aten._fused_rms_norm_backward(
            dy, npu_input, [8], rstd, npu_weight, [False, True])
        self.assertTrue(dx is None or not dx.defined())
        self.assertEqual(dgamma.shape, torch.Size([8]))
        dx2, dgamma2 = torch.ops.aten._fused_rms_norm_backward(
            dy, npu_input, [8], rstd, npu_weight, [True, False])
        self.assertEqual(dx2.shape, torch.Size([4, 8]))
        self.assertTrue(dgamma2 is None or not dgamma2.defined())


if __name__ == "__main__":
    run_tests()
