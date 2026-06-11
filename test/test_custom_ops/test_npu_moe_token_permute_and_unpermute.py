# Copyright (c) 2025 Huawei Technologies Co., Ltd
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import itertools
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize, instantiate_parametrized_tests

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices


TOL_MAPPING = {
    torch.float: dict(atol=1e-4, rtol=1e-4),
    torch.float16: dict(atol=1e-3, rtol=1e-3),
    torch.bfloat16: dict(atol=5e-3, rtol=5e-3),
}


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode.
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def unpermute_with_padded_tokens(
    permuted_tokens: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their corresponding probabilities.

    This function takes a tensor of permuted tokens and reorders them according to the provided indices. It also combines the tokens with their associated probabilities.

    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    assert permuted_tokens.dim() == 2, f"Got {permuted_tokens.dim()}D."

    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    assert (
        permuted_tokens.shape == indices.shape
    ), "Shape mismatch between permuted_tokens and indices."

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape,
        dtype=combined_output.dtype,
        device=combined_output.device,
    )

    # Scatter the combined tokens back to their original positions
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    """Permute the tokens based on the indices. Token with the same index will be grouped together.
       The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or [num_tokens, topk].
        num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should equal the number of tokens not dropped. By default, set to None, meaning no tokens are dropped.
        padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens with their corresponding probabilities.

    Args:
        permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
        sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
        probs (torch.Tensor, optional): The tensor of probabilities corresponding to the permuted tokens. If provided, the unpermuted tokens will be merged with their respective probabilities.
        padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity] to denote selected tokens per expert. Defaults to False.
        restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode. Defaults to None.

    Returns:
        torch.Tensor: The unpermuted tokens, optionally merged with probabilities.
    """
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


class TestPermute(TestCase):
    @SupportedDevices(['Ascend950'])
    def test_permute_mx_quant(self):
        tokens = torch.randn(8, 128).npu().to(torch.bfloat16)
        indices = torch.randint(0, 8, (8, 4), dtype=torch.int32).npu()

        output_e5m2, sorted_indices, scale_e5m2 = torch_npu.npu_moe_token_permute(
            tokens, indices, num_out_tokens=20, quant_mode=2)
        self.assertEqual(output_e5m2.shape, (20, 128))
        self.assertEqual(output_e5m2.dtype, torch.float8_e5m2)
        self.assertEqual(sorted_indices.shape, (32,))
        self.assertEqual(scale_e5m2.shape, (20, 4))
        self.assertEqual(scale_e5m2.dtype, torch.float8_e8m0fnu)

        output_e4m3, _, scale_e4m3 = torch_npu.npu_moe_token_permute(
            tokens, indices, num_out_tokens=20, quant_mode=3)
        self.assertEqual(output_e4m3.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_e4m3.shape, (20, 4))

        output_fp4, _, scale_fp4 = torch_npu.npu_moe_token_permute(
            tokens, indices, num_out_tokens=20, quant_mode=9)
        self.assertEqual(output_fp4.shape, (20, 64))
        self.assertEqual(output_fp4.dtype, torch.uint8)
        self.assertEqual(scale_fp4.shape, (20, 2, 2))

    @SupportedDevices(['Ascend910B'])
    def test_permute_quant_mode_is_ignored_on_non_950(self):
        tokens = torch.randn(16, 64).npu().to(torch.bfloat16)
        indices = torch.randint(0, 8, (16, 2)).npu()

        expected_tokens, expected_indices, _ = torch_npu.npu_moe_token_permute(tokens, indices)
        actual_tokens, actual_indices, expanded_scale = torch_npu.npu_moe_token_permute(
            tokens, indices, quant_mode=2)

        self.assertEqual(actual_tokens, expected_tokens)
        self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(actual_tokens.dtype, tokens.dtype)
        self.assertEqual(expanded_scale.shape, (0,))
        self.assertEqual(expanded_scale.dtype, torch.float32)

    @SupportedDevices(['Ascend910B'])
    @parametrize('num_tokens', [1024, 2048])
    @parametrize('hidden_size', [6144, 8192])
    @parametrize('topk', [1, 4])
    @parametrize('num_experts', [4, 128])
    @parametrize('dtype', [torch.bfloat16])
    def test_permute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        token_ori = torch.randn(num_tokens, hidden_size).npu().to(dtype).requires_grad_(True)
        indices_ori = torch.randint(0, num_experts, (num_tokens, topk)).npu()

        token_ori = token_ori.requires_grad_(True)
        token_fused = token_ori.clone().detach().requires_grad_(True)
        indices_fused = indices_ori.clone().detach()

        permuted_tokens_ori, sorted_indices_ori = permute(token_ori, indices_ori)
        permuted_tokens_fused, sorted_indices_fused, expanded_scale = torch_npu.npu_moe_token_permute(
            token_fused, indices_fused)

        self.assertEqual(permuted_tokens_ori, permuted_tokens_fused, **tols)
        self.assertEqual(expanded_scale.numel(), 0)
        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True).to(sorted_indices_fused.dtype)
        self.assertEqual(sorted_indices_ori, sorted_indices_fused)

    @SupportedDevices(['Ascend910B'])
    @parametrize('num_tokens', [1024, 2048])
    @parametrize('hidden_size', [6144, 8192])
    @parametrize('topk', [1, 4])
    @parametrize('num_experts', [4, 128])
    @parametrize('dtype', [torch.bfloat16])
    def test_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        permuted_tokens_ori = torch.randn(num_tokens * topk, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        sorted_indices_ori = torch.argsort(indices.view(-1), stable=True).npu().to(dtype=torch.int32)
        probs_ori = None
        probs_fused = None
        if topk > 1:
            probs_ori = (torch.ones(num_tokens, topk) / topk).npu().to(dtype).requires_grad_(True)
            probs_fused = probs_ori.clone().detach().requires_grad_(True)
        permuted_tokens_fused = permuted_tokens_ori.clone().detach().requires_grad_(True)
        sorted_indices_fused = sorted_indices_ori.clone().detach()

        # The fusion operator will perform two torch.argsort operations internally
        sorted_indices_ori = torch.argsort(sorted_indices_ori, stable=True)
        unpermuted_tokens_ori = unpermute(
            permuted_tokens_ori, sorted_indices_ori, probs=probs_ori)

        unpermuted_tokens_fused = torch_npu.npu_moe_token_unpermute(
            permuted_tokens_fused, sorted_indices_fused, probs=probs_fused)

        self.assertEqual(unpermuted_tokens_ori, unpermuted_tokens_fused, **tols)

    @SupportedDevices(['Ascend910B'])
    @parametrize('num_tokens', [1024, 2048])
    @parametrize('hidden_size', [6144, 8192])
    @parametrize('topk', [1, 4])
    @parametrize('num_experts', [4, 128])
    @parametrize('dtype', [torch.bfloat16])
    def test_ori_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)

        permuted_tokens, sorted_indices = permute(tokens, indices)
        unpermuted_tokens = unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            self.assertEqual(unpermuted_tokens, tokens)
        else:
            self.assertEqual(unpermuted_tokens, tokens, **tols)

    @SupportedDevices(['Ascend910B'])
    @parametrize('num_tokens', [1024, 2048])
    @parametrize('hidden_size', [6144, 8192])
    @parametrize('topk', [1, 4])
    @parametrize('num_experts', [4, 128])
    @parametrize('dtype', [torch.bfloat16])
    def test_npu_npu_moe_token_permute_unpermute(self, num_tokens, hidden_size, topk, num_experts, dtype):
        tols = TOL_MAPPING.get(dtype)
        tokens = torch.randn(num_tokens, hidden_size).npu().to(dtype)
        indices = torch.randint(0, num_experts, (num_tokens, topk)).npu()
        probs = None
        if topk > 1:
            probs = (torch.ones_like(indices) / topk).npu().to(dtype)

        permuted_tokens, sorted_indices, expanded_scale = torch_npu.npu_moe_token_permute(tokens, indices)
        self.assertEqual(expanded_scale.numel(), 0)
        unpermuted_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs=probs)

        if topk == 1:
            self.assertEqual(unpermuted_tokens, tokens)
        else:
            self.assertEqual(unpermuted_tokens, tokens, **tols)


instantiate_parametrized_tests(TestPermute)
if __name__ == "__main__":
    run_tests()
