import torch
import torch.nn as nn
<<<<<<< HEAD

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class TestNpuAttnSoftMax(TestCase):
    def npu_attn_softmax(self, attention_logits):
        torch_npu.npu_attn_softmax_(attention_logits)

    def npu_attn_softmax_backward(self, attention_logits, grad_output, v):
        torch_npu.npu_attn_softmax_backward_(
            attention_logits,
            grad_output,
            v
        )

    def golden_calc(self, attention_logits, grad_output, values):
        attention_logits_golden = attention_logits.detach().clone()
        attention_logits_golden.requires_grad = True
        softmax = nn.Softmax(dim=-1)
        softmax_output_golden = softmax(attention_logits_golden)
        output = torch.matmul(softmax_output_golden, values)
        output.backward(grad_output)
        grad_x_golden = attention_logits_golden.grad
        return softmax_output_golden, grad_x_golden

    def test_npu_attn_softmax(self):
        B = 10
        q_s = 4096
        kv_s = 4096
        H = 128
        attention_logits = torch.randn(B, q_s, kv_s).npu()
        grad_output = torch.randn(B, q_s, H).npu()
        values = torch.randn(B, kv_s, H).npu()
        
        # attention_logits = torch.randn(1, 1024, 8, 256, 256).npu()
        # grad_output = torch.randn(1, 1024, 8, 256, 8).npu()
        # values = torch.randn(1, 1024, 8, 256, 8).npu()

        # 标杆
        softmax_output_golden, grad_x_golden = self.golden_calc(attention_logits, grad_output, values)

        self.npu_attn_softmax(attention_logits)
        torch.testing.assert_close(softmax_output_golden, attention_logits, rtol=2**-11, atol=2**-11)

        self.npu_attn_softmax_backward(attention_logits, grad_output, values)
        torch.testing.assert_close(grad_x_golden, attention_logits, rtol=2**-11, atol=2**-11)

if __name__ == "__main__":
    run_tests()
=======
import torch_npu


def npu_attn_softmax(attention_logits, grad_output, v):
    torch_npu.npu_attn_softmax_(attention_logits)
    output = torch_npu.npu_attn_softmax_backward(
        attention_logits,
        grad_output,
        v
    )
    return attention_logits, output

def golden_calc(attention_logits, grad_output, values):
    attention_logits_golden = attention_logits.detach().clone()
    attention_logits_golden.requires_grad = True
    softmax = nn.Softmax(dim=-1)
    softmax_output_golden = softmax(attention_logits_golden)
    output = torch.matmul(softmax_output_golden, values)
    output.backward(grad_output)
    grad_x_golden = attention_logits_golden.grad
    return softmax_output_golden, grad_x_golden

def main(): 
    B = 10
    q_s = 4096
    kv_s = 4096
    H = 128
    attention_logits = torch.randn(B, q_s, kv_s).npu()
    grad_output = torch.randn(B, q_s, H).npu()
    values = torch.randn(B, kv_s, H).npu()

    softmax_output_golden, grad_x_golden = golden_calc(attention_logits, grad_output, values)
    attention_logits, out_backward = npu_attn_softmax(attention_logits, grad_output, values)

    print("-------------forward-----------------")
    torch.testing.assert_close(softmax_output_golden, attention_logits, rtol=2**-11, atol=2**-11)

    print("-------------backward-----------------")
    torch.testing.assert_close(grad_x_golden, out_backward, rtol=2**-11, atol=2**-11)

if __name__ == '__main__':
    main()
>>>>>>> c678007a3d398ec7bb2719cd555af6319cfb3707
