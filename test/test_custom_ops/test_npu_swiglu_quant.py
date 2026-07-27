# Copyright (c) 2025 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

# dst_type follows the CANN ge::DataType enum used by aclnnSwiGluQuantV2:
#   2  -> INT8 output, 29 -> INT4 output (NOT aten ScalarType ordinals).
DST_TYPE_INT8 = 2
DST_TYPE_INT4 = 29


class TestNpuSwigluQuant(TestCase):

    def tearDown(self):
        # Drain any residual async device state between tests so a fault in one
        # test (e.g. a negative case that launches a failing kernel) cannot
        # cascade onto the next on this shared, heavily-loaded host.
        if torch.npu.is_available():
            try:
                torch.npu.synchronize()
            except RuntimeError:
                pass

    def _swiglu(self, x_np, activate_left):
        """SwiGlu activation: split last dim in half.

        Matches the CANN aclnnSwiGluQuantV2 convention (verified against the
        kernel): activate_left=False applies silu to the RIGHT half -> a*silu(b);
        activate_left=True applies silu to the LEFT half -> silu(a)*b.
        """
        half = x_np.shape[-1] // 2
        a = x_np[..., :half]
        b = x_np[..., half:]
        silu = lambda t: t / (1.0 + np.exp(-t))
        if activate_left:
            return silu(a) * b
        return a * silu(b)

    def _row_group_ids(self, m, group_index, group_list_type):
        """Map each of the m flattened rows to its group id (MoE), or all-zero (non-MoE)."""
        if group_index is None:
            return np.zeros((m,), dtype=np.int64)
        gi = group_index.astype(np.int64)
        # cumsum(0): gi are cumulative right boundaries; count(1): gi are per-group counts.
        boundaries = gi if group_list_type == 0 else np.cumsum(gi)
        ids = np.zeros((m,), dtype=np.int64)
        prev = 0
        for g, bound in enumerate(boundaries):
            ids[prev:bound] = g
            prev = bound
        return ids

    def _default_smooth_scales(self, x, group_index):
        """Synthesize an all-ones smooth_scales (numeric identity).

        The deployed CANN aclnnSwiGluQuantV2 kernel requires a valid
        smooth_scales buffer even though _def.cpp marks it OPTIONAL: with an
        absent smooth_scales it faults with an AICore MTE out-of-range access
        (surfaces as error 507057). `.md` also states smooth_scales "不支持空
        Tensor". An all-ones tensor is a numeric no-op, so tests that target
        other parameters still compare correct goldens while satisfying the
        kernel's real support surface.
        """
        half = x.shape[-1] // 2
        groups = group_index.numel() if group_index is not None else 1
        return torch.ones((groups, half), dtype=torch.float32).npu()

    def supported_op_exec(self, x, *, smooth_scales=None, offsets=None, group_index=None,
                          activate_left=False, quant_mode="dynamic", group_list_type=0,
                          dst_type=DST_TYPE_INT8):
        """NumPy CPU reference implementation for SwiGluQuant."""
        if smooth_scales is None:
            smooth_scales = self._default_smooth_scales(x, group_index)
        x_np = x.cpu().to(torch.float32).numpy()
        lead = x_np.shape[:-1]
        half = x_np.shape[-1] // 2

        act = self._swiglu(x_np, activate_left).reshape(-1, half)  # (M, N)
        m = act.shape[0]
        row_ids = self._row_group_ids(m, None if group_index is None else group_index.cpu().numpy(),
                                      group_list_type)

        y_tmp = act.copy()
        if smooth_scales is not None:
            ss = smooth_scales.cpu().to(torch.float32).numpy()
            if ss.ndim == 1:
                ss = ss[None, :]
            y_tmp = y_tmp * ss[row_ids]

        qmax = 127.0 if dst_type == DST_TYPE_INT8 else 7.0
        qlo, qhi = (-128, 127) if dst_type == DST_TYPE_INT8 else (-8, 7)

        if quant_mode == "static":
            if offsets is not None:
                off = offsets.cpu().to(torch.float32).numpy()
                if off.ndim == 1:
                    off = off[None, :]
                y_tmp = y_tmp + off[row_ids]
            y = np.clip(np.round(y_tmp), qlo, qhi).astype(np.int8)
            # Static quant does not produce a per-row dynamic scale; the kernel
            # returns an all-zero scale tensor.
            scale = np.zeros((m,), dtype=np.float32)
        else:  # dynamic
            row_max = np.maximum(np.max(np.abs(y_tmp), axis=-1), 1e-12)
            scale = (qmax / row_max).astype(np.float32)
            y = np.clip(np.round(y_tmp * scale[:, None]), qlo, qhi).astype(np.int8)

        y = y.reshape(*lead, half)
        scale = scale.reshape(*lead)
        return y, scale

    def custom_op_exec(self, x, *, smooth_scales=None, offsets=None, group_index=None,
                       activate_left=False, quant_mode="dynamic", group_list_type=0,
                       dst_type=DST_TYPE_INT8):
        """NPU operator call."""
        if smooth_scales is None:
            smooth_scales = self._default_smooth_scales(x, group_index)
        y, scale = torch_npu.npu_swiglu_quant(
            x, smooth_scales=smooth_scales, offsets=offsets, group_index=group_index,
            activate_left=activate_left, quant_mode=quant_mode,
            group_list_type=group_list_type, dst_type=dst_type)
        # Force any async device fault to surface inside this test rather than
        # cascading onto later tests (the shared host is heavily loaded).
        torch.npu.synchronize()
        return y.cpu().to(torch.int8).numpy(), scale.cpu().to(torch.float32).numpy()

    # ---- Basic functionality (dtype coverage) ----
    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self, device="npu"):
        """Basic dynamic quant, bfloat16 input."""
        torch.manual_seed(0)
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x)
        npu_y, npu_s = self.custom_op_exec(x.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self, device="npu"):
        """Basic dynamic quant, float16 input."""
        torch.manual_seed(0)
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x)
        npu_y, npu_s = self.custom_op_exec(x.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self, device="npu"):
        """Basic dynamic quant, float32 input."""
        torch.manual_seed(0)
        x = torch.randn((32, 256), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x)
        npu_y, npu_s = self.custom_op_exec(x.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.0001)

    # ---- Optional param combinations ----
    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_bf16(self, device="npu"):
        """smooth_scales + offsets + group_index (static MoE), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        smooth_scales = torch.ones((2, 128), dtype=torch.float32).npu()
        offsets = torch.zeros((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, offsets=offsets, group_index=group_index,
            quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), offsets=offsets.clone(),
            group_index=group_index.clone(), quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_fp16(self, device="npu"):
        """smooth_scales + offsets + group_index (static MoE), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((2, 128), dtype=torch.float32).npu()
        offsets = torch.zeros((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, offsets=offsets, group_index=group_index,
            quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), offsets=offsets.clone(),
            group_index=group_index.clone(), quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_bf16(self, device="npu"):
        """smooth_scales + offsets (static, non-MoE), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        smooth_scales = torch.ones((1, 128), dtype=torch.float32).npu()
        offsets = torch.zeros((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, offsets=offsets, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), offsets=offsets.clone(),
            quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_fp16(self, device="npu"):
        """smooth_scales + offsets (static, non-MoE), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((1, 128), dtype=torch.float32).npu()
        offsets = torch.zeros((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, offsets=offsets, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), offsets=offsets.clone(),
            quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_bf16(self, device="npu"):
        """smooth_scales + group_index (dynamic MoE), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        smooth_scales = torch.ones((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, group_index=group_index)
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), group_index=group_index.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_fp16(self, device="npu"):
        """smooth_scales + group_index (dynamic MoE), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, smooth_scales=smooth_scales, group_index=group_index)
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), smooth_scales=smooth_scales.clone(), group_index=group_index.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_bf16(self, device="npu"):
        """smooth_scales only (dynamic non-MoE), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        smooth_scales = torch.ones((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, smooth_scales=smooth_scales)
        npu_y, npu_s = self.custom_op_exec(x.clone(), smooth_scales=smooth_scales.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_fp16(self, device="npu"):
        """smooth_scales only (dynamic non-MoE), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, smooth_scales=smooth_scales)
        npu_y, npu_s = self.custom_op_exec(x.clone(), smooth_scales=smooth_scales.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_bf16(self, device="npu"):
        """offsets + group_index (static MoE, no smooth_scales), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        offsets = torch.zeros((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, offsets=offsets, group_index=group_index, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), offsets=offsets.clone(), group_index=group_index.clone(),
            quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_fp16(self, device="npu"):
        """offsets + group_index (static MoE, no smooth_scales), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        offsets = torch.zeros((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(
            x, offsets=offsets, group_index=group_index, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(
            x.clone(), offsets=offsets.clone(), group_index=group_index.clone(),
            quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_bf16(self, device="npu"):
        """offsets only (static non-MoE, no smooth_scales), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        offsets = torch.zeros((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, offsets=offsets, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(x.clone(), offsets=offsets.clone(), quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_fp16(self, device="npu"):
        """offsets only (static non-MoE, no smooth_scales), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        offsets = torch.zeros((1, 128), dtype=torch.float32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, offsets=offsets, quant_mode="static")
        npu_y, npu_s = self.custom_op_exec(x.clone(), offsets=offsets.clone(), quant_mode="static")
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_group_index_bf16(self, device="npu"):
        """group_index only (dynamic MoE, no smooth_scales), bf16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.bfloat16).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, group_index=group_index)
        npu_y, npu_s = self.custom_op_exec(x.clone(), group_index=group_index.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_group_index_fp16(self, device="npu"):
        """group_index only (dynamic MoE, no smooth_scales), fp16."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, group_index=group_index)
        npu_y, npu_s = self.custom_op_exec(x.clone(), group_index=group_index.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    # ---- dst_type coverage ----
    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int8(self, device="npu"):
        """dst_type=2 (INT8 output)."""
        torch.manual_seed(0)
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, dst_type=DST_TYPE_INT8)
        npu_y, npu_s = self.custom_op_exec(x.clone(), dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int4(self, device="npu"):
        """dst_type=29 (INT4 output); last dim (256) is a multiple of 4."""
        torch.manual_seed(0)
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, dst_type=DST_TYPE_INT4)
        npu_y, npu_s = self.custom_op_exec(x.clone(), dst_type=DST_TYPE_INT4)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_int4_output_shape(self, device="npu"):
        """INT4 output shape: last dim = x last dim / 2, other dims unchanged."""
        torch.manual_seed(0)
        x = torch.randn((4, 64), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x, dst_type=DST_TYPE_INT4)
        npu_y, npu_s = self.custom_op_exec(x.clone(), dst_type=DST_TYPE_INT4)
        self.assertEqual(list(npu_y.shape), [4, 32])
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    # ---- Numerical / large data ----
    @SupportedDevices(['Ascend910B'])
    def test_scale_value(self, device="npu"):
        """Verify dynamic scale golden (not pure shape smoke)."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x)
        npu_y, npu_s = self.custom_op_exec(x.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_large_input(self, device="npu"):
        """Large data path; compare y and scale golden."""
        torch.manual_seed(0)
        x = torch.randn((3072, 7168), dtype=torch.float16).npu()
        cpu_y, cpu_s = self.supported_op_exec(x)
        npu_y, npu_s = self.custom_op_exec(x.clone())
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    # ---- Graph mode (torch.compile) ----
    @SupportedDevices(['Ascend910B'])
    def test_graph_mode(self, device="npu"):
        """Graph mode (npugraph_ex) output must match eager."""
        torch.manual_seed(0)
        x = torch.randn((2, 32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((2, 128), dtype=torch.float32).npu()
        group_index = torch.tensor([32, 64], dtype=torch.int32).npu()

        def fn(inp, ss, gi):
            return torch_npu.npu_swiglu_quant(inp, smooth_scales=ss, group_index=gi)

        eager_y, eager_s = fn(x, smooth_scales, group_index)
        compiled = torch.compile(fn, backend="npugraph_ex", dynamic=False)
        graph_y, graph_s = compiled(x.clone(), smooth_scales.clone(), group_index.clone())
        self.assertRtolEqual(eager_y.cpu().to(torch.int8).numpy(),
                             graph_y.cpu().to(torch.int8).numpy(), 1)
        self.assertRtolEqual(eager_s.cpu().to(torch.float32).numpy(),
                             graph_s.cpu().to(torch.float32).numpy(), 0.001)

    # ---- Negative cases ----
    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_x_invalid_dtype(self, device="npu"):
        """x with unsupported dtype (int64) should raise."""
        x = torch.randint(0, 10, (32, 256), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swiglu_quant(x)

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_smooth_scales_invalid_dtype(self, device="npu"):
        """smooth_scales with unsupported dtype (int64) should raise."""
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        smooth_scales = torch.ones((1, 128), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swiglu_quant(x, smooth_scales=smooth_scales)

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_offsets_invalid_dtype(self, device="npu"):
        """offsets with unsupported dtype (int64) should raise."""
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        offsets = torch.ones((1, 128), dtype=torch.int64).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swiglu_quant(x, offsets=offsets, quant_mode="static")

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_group_index_invalid_dtype(self, device="npu"):
        """group_index with unsupported dtype (float32) should raise."""
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        group_index = torch.tensor([32.0, 64.0], dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swiglu_quant(x, group_index=group_index)

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_dst_type_invalid(self, device="npu"):
        """dst_type not in {2, 29} should raise PARAM_INVALID."""
        x = torch.randn((32, 256), dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, r"dst_type"):
            torch_npu.npu_swiglu_quant(x, dst_type=3)

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_x_odd_last_dim(self, device="npu"):
        """x last dim not even should raise PARAM_INVALID."""
        x = torch.randn((32, 255), dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, r"even"):
            torch_npu.npu_swiglu_quant(x)

    @unittest.skipIf(not torch.npu.is_available(), "NPU required for negative validation")
    def test_neg_int4_last_dim_not_align(self, device="npu"):
        """dst_type=29 (INT4) with x last dim not a multiple of 4 should raise."""
        x = torch.randn((32, 258), dtype=torch.float16).npu()
        with self.assertRaisesRegex(RuntimeError, r"multiple of 4"):
            torch_npu.npu_swiglu_quant(x, dst_type=DST_TYPE_INT4)


if __name__ == "__main__":
    run_tests()
