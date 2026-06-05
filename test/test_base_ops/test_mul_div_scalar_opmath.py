# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import torch
import torch_npu


def _check_equal_special(cpu, npu):
    npu_cpu = npu.cpu()
    assert torch.equal(torch.isnan(cpu), torch.isnan(npu_cpu)), (cpu, npu_cpu)
    assert torch.equal(torch.isinf(cpu), torch.isinf(npu_cpu)), (cpu, npu_cpu)
    assert torch.equal(torch.signbit(cpu), torch.signbit(npu_cpu)), (cpu, npu_cpu)


def test_mul_div_scalar_opmath():
    large = -1.8210423542253347e38
    small = 2.0 / 3159917657685606408

    zeros_cpu = torch.zeros(16, dtype=torch.float16)
    zeros_npu = zeros_cpu.npu()
    _check_equal_special(zeros_cpu * large, zeros_npu * large)

    inf_cpu = torch.full((16,), float("inf"), dtype=torch.float16)
    inf_npu = inf_cpu.npu()
    _check_equal_special(small * inf_cpu, small * inf_npu)

    ones_cpu = torch.ones(16, dtype=torch.float16)
    ones_npu = ones_cpu.npu()
    _check_equal_special(ones_cpu / large, ones_npu / large)

    _check_equal_special(small / inf_cpu, small / inf_npu)


if __name__ == "__main__":
    test_mul_div_scalar_opmath()
    print("mul/div scalar opmath alignment passed")
