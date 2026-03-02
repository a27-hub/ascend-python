import torch
from torch.library import Library, impl

'''
Registering Meta implementations for over-adapted ops
'''

#meta register implementation
m_aten = Library("aten", "IMPL", "Meta")

@impl(m_aten, "matmul_backward")
def matmul_backward_meta(grad, self, other, mask):
    self_len = len(self.size())
    other_len = len(other.size())
    if self_len == 1 and (other_len == 1 or other_len == 2):
        self_shape = (1,) + self.size()
        return (torch.empty(self_shape, dtype=self.dtype, device=self.device), torch.empty_like(other))
    elif self_len != other_len and self_len > other_len > 2:
        other_shape = self.size()[:-2] + (self.size()[-1], grad.size()[-1])
        return (torch.empty_like(self), torch.empty(other_shape, dtype=other.dtype, device=other.device))
    elif self_len != other_len and other_len > self_len > 2:
        self_shape = grad.size()[:-1] + (other.size()[-2],)
        return (torch.empty(self_shape, dtype=self.dtype, device=self.device), torch.empty_like(other))

    return (torch.empty_like(self), torch.empty_like(other))