import unittest
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuCausalConv1dFnAdd(TestCase):
    def get_golden(self, x, weight, conv_states, query_start_loc, cache_indices,
                   initial_state_mode, residual_connection=0, pad_slot_id=-1):
        x_fp32 = x.float()
        weight_fp32 = weight.float()
        conv_states_fp32 = conv_states.float()
        dtype = x.dtype

        cu_seq_len, dim = x_fp32.shape
        batch_size = query_start_loc.shape[0] - 1
        kernel_width = weight_fp32.size(0)
        state_len = kernel_width - 1

        out = torch.zeros(cu_seq_len, dim, dtype=torch.float32)

        for batch_idx in range(batch_size):
            slot_idx = cache_indices[batch_idx].item()
            if slot_idx == pad_slot_id:
                continue

            start_idx = query_start_loc[batch_idx].item()
            end_idx = query_start_loc[batch_idx + 1].item()
            seq_x = x_fp32[start_idx:end_idx]

            has_state = initial_state_mode[batch_idx].item()

            if has_state == 1:
                cached_state = conv_states_fp32[slot_idx].clone()
            else:
                cached_state = torch.zeros(state_len, dim, dtype=torch.float32)

            padded_input = torch.cat([cached_state, seq_x], dim=0)

            result = F.conv1d(
                padded_input.transpose(0, 1).unsqueeze(0),
                weight_fp32.transpose(0, 1).unsqueeze(1),
                bias=None, stride=1, padding=0, groups=dim,
            ).squeeze(0).transpose(0, 1)

            if has_state == 2:
                result[:state_len] = 0

            if residual_connection == 1:
                out[start_idx:end_idx] = result + x_fp32[start_idx:end_idx]
            else:
                out[start_idx:end_idx] = result

            conv_states_fp32[slot_idx] = padded_input[-state_len:]

        return out.to(dtype), conv_states_fp32.to(dtype)

    @unittest.skip("Skip test_npu_fused_causal_conv1d_fn now")
    @SupportedDevices(['Ascend950'])
    def test_npu_fused_causal_conv1d_fn_float16(self):
        batch = 4
        dim = 128
        kernel_width = 3
        state_len = kernel_width - 1
        dtype = torch.float16

        seq_lens = [5, 3, 7, 4]
        cu_seq_len = sum(seq_lens)
        num_slots = batch * 2

        x = torch.randn(cu_seq_len, dim, dtype=dtype)
        weight = torch.randn(kernel_width, dim, dtype=dtype)
        conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype)

        starts = [0]
        for sl in seq_lens:
            starts.append(starts[-1] + sl)
        query_start_loc = torch.tensor(starts, dtype=torch.int32)
        cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32)
        initial_state_mode = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
        residual_connection = 0

        golden_out, golden_states = self.get_golden(
            x, weight, conv_states.clone(), query_start_loc, cache_indices,
            initial_state_mode, residual_connection=residual_connection,
        )

        conv_states_npu = conv_states.clone().npu()
        out_npu = torch_npu.npu_fused_causal_conv1d_fn(
            x.npu(), weight.npu(), conv_states_npu,
            query_start_loc=query_start_loc.npu(),
            cache_indices=cache_indices.npu(),
            initial_state_mode=initial_state_mode.npu(),
            activation="None",
            residual_connection=residual_connection,
            pad_slot_id=-1,
        )
        torch.npu.synchronize()

        self.assertRtolEqual(out_npu.cpu(), golden_out)
        self.assertRtolEqual(conv_states_npu.cpu(), golden_states)


if __name__ == "__main__":
    run_tests()
