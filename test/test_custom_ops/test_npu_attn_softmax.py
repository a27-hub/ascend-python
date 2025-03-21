import torch
import torch.nn as nn
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
