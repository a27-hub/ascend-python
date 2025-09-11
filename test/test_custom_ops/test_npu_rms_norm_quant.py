import unittest
import math

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNPURmsNormQuant(TestCase):

    def compare(self, a, b, benchmark):
        diff_abs = torch.abs(a - b)
        max_diff_abs, _ = torch.max(diff_abs, dim=0)

        if max_diff_abs.item() == 0:
            return True
        else:
            rel_error = 0
            abs_error = 0
            for i in range(a.shape[0]):
                yes_no = (a[i] == 0 and b[i].item() != 0)
                no_yes = (a[i] != 0 and b[i].item() == 0)
                if a[i].item() == 0 and b[i].item() == 0:
                    diff_rel_item = 0
                elif yes_no or no_yes:
                    diff_rel_item = 1
                elif a[i] != 0 and b[i].item() != 0:
                    diff_rel_item = diff_abs[i].item() / abs(a[i].item())
                        
                if abs(a[i].item()) < 1 and diff_abs[i].item() > benchmark:
                    abs_error += 1
                elif abs(a[i].item()) >= 1 and diff_rel_item > benchmark:
                    rel_error += 1
                if (rel_error + abs_error) > 10:
                    break
            if (rel_error + abs_error) > 0:
                return False
            else:
                return True

    def npu_rms_norm_quant_golden(self, input_x: torch.Tensor, input_gamma: torch.Tensor, input_beta: torch.Tensor, input_scale: torch.Tensor, input_offset: torch.Tensor, epsilon: float = 1e-6):
        torchType32 = torch.float32
    
        input_gamma = input_gamma.reshape(-1)
        input_beta  = input_beta.reshape(-1)
    
        D = input_gamma.numel()
    
        x_fp32      = input_x.to(torchType32)
        gamma_fp32  = input_gamma.to(torchType32)
        beta_fp32   = input_beta.to(torchType32)
    
        variance = torch.mean(torch.pow(x_fp32, 2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + epsilon)
        y_mid = x_fp32 * rstd  
    
        bcast_shape = [1] * (y_mid.dim() - 1) + [D]
        gamma_b = gamma_fp32.view(*bcast_shape)
        beta_b  = beta_fp32.view(*bcast_shape)
        quant_in = y_mid * gamma_b + beta_b  
    
        scale_val = float(input_scale.reshape(()).to(torchType32).item())

        offset_t = input_offset.reshape(())
        if offset_t.dtype.is_floating_point:
            zp_val = int(round(float(offset_t.to(torchType32).item())))
        else:
            zp_val = int(offset_t.to(torch.int32).item())
    
        z = quant_in * scale_val + float(zp_val)    
        z_rao = torch.sign(z) * torch.floor(torch.abs(z) + 0.5)
        z_i8 = torch.clamp(z_rao, -128, 127).to(torch.int8)
        y_int8 = z_i8.reshape(input_x.shape)
        return y_int8

    @unittest.skip("skip test_npu_rms_norm_quant due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rms_norm_quant(self):
        shape_list = [
            [[16,],      [16,]],    
            [[16,],      [1, 16]],   
            [[1, 16],    [16,]],     
            [[1, 16],    [1, 16]],   
            [[1, 1, 16], [16,]], 
            [[1, 1, 16], [1, 16]],
            [[1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 1, 1, 16], [1, 16]],     
        ]  

        benchmark_int8 = 1              

        for x_shape, quant_shape in shape_list:
            D = x_shape[-1]
            x = torch.randn(x_shape, dtype=torch.float16)
  
            if quant_shape == [D,]:
                gamma = torch.randn(D, dtype=torch.float16)
                beta  = torch.randn(D, dtype=torch.float16)
            elif quant_shape == [1, D]:
                gamma = torch.randn(1, D, dtype=torch.float16)
                beta  = torch.randn(1, D, dtype=torch.float16)

            scale = (torch.rand(1, dtype=torch.float16) * 0.8 + 0.2)  
            offset = torch.randint(-5, 6, (1,), dtype=torch.int8)

            x_npu       = x.npu()     
            gamma_npu   = gamma.npu() 
            beta_npu    = beta.npu() 
            scale_npu   = scale.npu() 
            offset_npu  = offset.npu() 
 
            y_ref = self.npu_rms_norm_quant_golden(x, gamma, beta, scale, offset, epsilon=1e-6)
            y_npu = torch_npu.npu_rms_norm_quant(x_npu, gamma_npu, beta_npu, scale_npu, offset_npu, epsilon=1e-6)

            y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
            y_npu_flat = y_npu.reshape(1, y_npu.numel())[0].cpu()
            self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_int8))

    @unittest.skip("skip test_npu_rms_norm_quant due to low version of cann")
    @SupportedDevices(['Ascend910B'])
    def test_npu_rms_norm_quant_bf16(self):
        shape_list = [
            [[16,],      [16,]],    
            [[16,],      [1, 16]],   
            [[1, 16],    [16,]],     
            [[1, 16],    [1, 16]],   
            [[1, 1, 16], [16,]], 
            [[1, 1, 16], [1, 16]],
            [[1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 1, 16], [1, 16]],
            [[1, 1, 1, 1, 1, 1, 16], [16,]], 
            [[1, 1, 1, 1, 1, 1, 16], [1, 16]],     
        ]
  
        benchmark_int8 = 1              

        for x_shape, quant_shape in shape_list:
            D = x_shape[-1]
            x = torch.randn(x_shape, dtype=torch.bfloat16)

            if quant_shape == [D,]:
                gamma = torch.randn(D, dtype=torch.bfloat16)
                beta  = torch.randn(D, dtype=torch.bfloat16)
            elif quant_shape == [1, D]:
                gamma = torch.randn(1, D, dtype=torch.bfloat16)
                beta  = torch.randn(1, D, dtype=torch.bfloat16)

            scale = (torch.rand(1, dtype=torch.bfloat16) * 0.8 + 0.2) 

            offset = torch.randint(-5, 6, (1,), dtype=torch.int8)

            x_npu       = x.npu()     
            gamma_npu   = gamma.npu() 
            beta_npu    = beta.npu() 
            scale_npu   = scale.npu() 
            offset_npu  = offset.npu() 

            y_ref = self.npu_rms_norm_quant_golden(x, gamma, beta, scale, offset, epsilon=1e-6)

            y_npu = torch_npu.npu_rms_norm_quant(x_npu, gamma_npu, beta_npu, scale_npu, offset_npu, epsilon=1e-6)

            y_ref_flat = y_ref.reshape(1, y_ref.numel())[0].cpu()
            y_npu_flat = y_npu.reshape(1, y_npu.numel())[0].cpu()
            self.assertTrue(self.compare(y_ref_flat, y_npu_flat, benchmark_int8))

if __name__ == "__main__":
    run_tests() 