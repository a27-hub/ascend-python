import torch
import torch_npu

x = torch.tensor([1,2,3,4]).reshape(2,1,2).to(torch.complex64)
x_npu = x.npu()

print(f"x = {x}")
print(f"x_npu = {x_npu}")

cpu = torch.fft.fft(x, dim=1, n=1)
npu = torch.fft.fft(x_npu, dim=1, n=1)

print(f"cpu = {cpu}")
print(f"npu = {npu}")