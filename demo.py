#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See LICENSE in the root of the software repository for the full text of the License.

"""W4A8 demo for torch_npu.npu_grouped_matmul_finalize_routing on Atlas A2/A3."""

import numpy as np
import torch
import torch_npu


def encode_w4a8_scale(scale_fp32: np.ndarray) -> torch.Tensor:
    """Encode float scale values into the int64 layout required by the W4A8 API."""
    scale_fp32 = scale_fp32.astype(np.float16).astype(np.float32)
    scale_bits = scale_fp32.view(np.uint32)
    scale_u32 = np.zeros((*scale_bits.shape[:-1], scale_bits.shape[-1] * 2), dtype=np.uint32)
    scale_u32[..., ::2] = scale_bits
    return torch.from_numpy(scale_u32.view(np.int64))


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # A2/A3 W4A8 currently requires K=2048 and N=7168.
    expert_num, token_num, k, n = 1, 8, 2048, 7168

    # A8 activation.
    x = torch.randint(-5, 5, (token_num, k), dtype=torch.int8)

    # Quantize the ND weight from fp32 to packed int4 (quint4x2).
    weight_fp32 = torch.randint(-5, 5, (expert_num, k, n), dtype=torch.int32).float().npu()
    weight_scale = torch.ones(1, dtype=torch.float32, device="npu")
    weight_w4 = torch_npu.npu_quantize(
        weight_fp32,
        weight_scale,
        None,
        torch.quint4x2,
        -1,
        False,
    )

    # count-mode group_list: all tokens are assigned to the only expert.
    group_list = torch.tensor([token_num], dtype=torch.int64)

    # W4A8 scale uses int64 storage with shape (E, 1, N).
    scale_fp32 = np.random.normal(0.0, 0.01, (expert_num, 1, n)).astype(np.float32)
    scale = encode_w4a8_scale(scale_fp32)

    # bias is mandatory on Atlas A2/A3 and has shape (E, N).
    bias = torch.randn((expert_num, n), dtype=torch.float32) * 0.01
    pertoken_scale = torch.ones(token_num, dtype=torch.float32)

    # Identity routing: every input row is scattered to the same output row.
    logit = torch.ones(token_num, dtype=torch.float32)
    row_index = torch.arange(token_num, dtype=torch.int64)

    output = torch_npu.npu_grouped_matmul_finalize_routing(
        x.npu(),
        weight_w4,
        group_list.npu(),
        scale=scale.npu(),
        bias=bias.npu(),
        pertoken_scale=pertoken_scale.npu(),
        logit=logit.npu(),
        row_index=row_index.npu(),
        output_bs=token_num,
        group_list_type=1,
    )
    torch.npu.synchronize()

    print(f"output shape: {tuple(output.shape)}")
    print(f"output dtype: {output.dtype}")
    print(f"output[0, :8]: {output[0, :8].cpu()}")


if __name__ == "__main__":
    main()