import torch
import torch_npu
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests


class TestNPURenormBackward(TestCase):

    def test_basic_forward(self):
        """基础功能：直接调用 npu_renorm_backward"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        result = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, x.dtype)

    def test_compare_with_autograd_p_2(self):
        """对比自动微分：p=2"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 2.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_1(self):
        """对比自动微分：p=1"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 1.0, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 1.0, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_p_0_5(self):
        """对比自动微分：p<1"""
        x = torch.randn(4, 8, device='npu', requires_grad=True)
        grad = torch.randn(4, 8, device='npu')
        y = torch.renorm(x, 0.5, 0, 1.0)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 0.5, 0, 1.0)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))

    def test_compare_with_autograd_3d(self):
        """3D 张量测试"""
        x = torch.randn(3, 4, 5, device='npu', requires_grad=True)
        grad = torch.randn(3, 4, 5, device='npu')
        y = torch.renorm(x, 2.0, 1, 0.8)
        y.backward(grad)
        grad_auto = x.grad.clone()
        x.grad.zero_()
        grad_custom = torch_npu.npu_renorm_backward(grad, x, 2.0, 1, 0.8)
        self.assertTrue(torch.allclose(grad_auto, grad_custom, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    run_tests()
    