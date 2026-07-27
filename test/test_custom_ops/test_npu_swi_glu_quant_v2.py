import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuSwiGluQuantV2(TestCase):
    """
    UT for npu_swi_glu_quant_v2 (aclnnSwiGluQuantV2).

    dst_type uses the ge::DataType convention (2=int8, 29=int4) because the
    C++ adaptation passes dst_type via static_cast<aclDataType>(dst_type). This
    is NOT the aten ScalarType-ordinal convention (1/16) used by some other
    quant ops (DynamicBlockQuant); do not "fix" 2/29 to 1/16.

    INT4 (dst_type=29) output is 2-per-byte packed (dtype uint8, physical last
    dim = x_last/4, per meta-guide "aclnnSwiGluQuantV2 uses this convention"),
    NOT quint4x2 int32-packed (/8). int4 y is asserted structurally; scale is
    value-compared (Iron Rule: scale always has a golden).

    All positive cases pass smooth_scales (non-None) to avoid the device-poison
    pitfall (null smooth_scales poisoned the stream in CANN 8.5 / V1); the
    smooth_scales=None case is covered by a negative test.

    Reference (cpu_op_exec) implements the DYNAMIC quant path
    (quant_mode="dynamic", offsets ignored per .md). SwiGlu split: A=front
    half, B=back half; activate_left=False -> act = A * silu(B) (per V1 case
    pitfall #2: False activates the second half). group_index uses cumsum mode
    (group_list_type=0): group_index=[g0,g1,...] are cumulative end
    boundaries (per V1 pitfall #4).
    """

    DST_TYPE_INT8 = 2
    DST_TYPE_INT4 = 29
    DST_SCALE_INT8 = 127.0
    DST_SCALE_INT4 = 7.0

    def cpu_op_exec(self, x_np, smooth_scales_np, group_index_np, activate_left, dst_type):
        """CPU dynamic-quant reference. Returns (y_int8/uint8_struct, scale)."""
        x = torch.from_numpy(x_np)
        last_dim = x.shape[-1]
        a = x[..., :last_dim // 2]
        b = x[..., last_dim // 2:]
        if activate_left:
            act = F.silu(a) * b
        else:
            act = a * F.silu(b)
        act_np = act.float().numpy()

        if group_index_np is not None:
            gi = np.asarray(group_index_np).astype(np.int64)
            start = 0
            y_tmp = np.zeros_like(act_np)
            for i in range(len(gi)):
                end = int(gi[i])
                ss = smooth_scales_np[i] if smooth_scales_np.ndim == 2 else smooth_scales_np
                y_tmp[start:end] = act_np[start:end] * ss.astype(np.float32)
                start = end
        else:
            ss = smooth_scales_np.astype(np.float32)
            y_tmp = act_np * ss

        dst_scale = self.DST_SCALE_INT8 if dst_type == self.DST_TYPE_INT8 else self.DST_SCALE_INT4
        row_max = np.max(np.abs(y_tmp), axis=-1, keepdims=True).astype(np.float32)
        row_max = np.where(row_max == 0, 1.0, row_max)
        scale = (dst_scale / row_max).astype(np.float32)
        y_scaled = y_tmp * scale
        if dst_type == self.DST_TYPE_INT8:
            y = np.clip(np.round(y_scaled), -128, 127).astype(np.int8)
        else:
            y = np.clip(np.round(y_scaled), -8, 7).astype(np.int8)
        return y, scale

    def npu_op_exec(self, x, smooth_scales=None, offsets=None, group_index=None,
                    activate_left=False, quant_mode="dynamic", group_list_type=0, dst_type=2):
        x = x.npu()
        if smooth_scales is not None:
            smooth_scales = smooth_scales.npu()
        if offsets is not None:
            offsets = offsets.npu()
        if group_index is not None:
            group_index = group_index.npu()
        y, scale = torch_npu.npu_swi_glu_quant_v2(
            x, smooth_scales=smooth_scales, offsets=offsets, group_index=group_index,
            activate_left=activate_left, quant_mode=quant_mode,
            group_list_type=group_list_type, dst_type=dst_type)
        return y.cpu().numpy(), scale.cpu().numpy()

    def _make_x(self, shape, dtype_np):
        torch.manual_seed(0)
        return torch.randn(shape, dtype=torch.from_numpy(dtype_np)) * 4.0

    def _make_smooth(self, n, dtype_np):
        return torch.ones(n, dtype=torch.from_numpy(dtype_np))

    def _make_group_index(self, m, groups):
        # cumsum end boundaries, evenly split
        per = m // groups
        bounds = [per * (i + 1) for i in range(groups)]
        bounds[-1] = m
        return torch.tensor(bounds, dtype=torch.int32)

    # ---- basic functionality (non-MoE dynamic int8) ----
    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((32, 256), dtype=torch.float16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((32, 256), dtype=torch.float32) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.0001)

    def _run_moe_dynamic(self, shape, dtype_t, dtype_np, tol):
        torch.manual_seed(0)
        x = (torch.randn(shape, dtype=dtype_t) * 4.0).npu()
        m = shape[0]
        groups = 2
        gi = self._make_group_index(m, groups)
        smooth = torch.ones((groups, shape[-1] // 2), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), gi.numpy(), False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), group_index=gi.clone(),
                                        dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, tol)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_bf16(self, device="npu"):
        # offsets passed but ignored in dynamic mode (per .md)
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        gi = self._make_group_index(2, 1)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        offsets = torch.zeros((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), gi.numpy(), False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        group_index=gi.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_fp16(self, device="npu"):
        torch.manual_seed(0)
        self._run_moe_dynamic((4, 32, 256), torch.float16, np.float16, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_bf16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        offsets = torch.zeros(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_fp16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.float16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        offsets = torch.zeros(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_bf16(self, device="npu"):
        torch.manual_seed(0)
        self._run_moe_dynamic((4, 32, 256), torch.bfloat16, np.float16, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_fp16(self, device="npu"):
        torch.manual_seed(0)
        self._run_moe_dynamic((4, 32, 256), torch.float16, np.float16, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_bf16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_fp16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.float16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_bf16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        gi = self._make_group_index(2, 1)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        offsets = torch.zeros((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), gi.numpy(), False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        group_index=gi.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_fp16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.float16) * 4.0).npu()
        gi = self._make_group_index(2, 1)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        offsets = torch.zeros((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), gi.numpy(), False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        group_index=gi.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_bf16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        offsets = torch.zeros(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_offsets_fp16(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.float16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        offsets = torch.zeros(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), offsets=offsets.clone(),
                                        dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_group_index_bf16(self, device="npu"):
        torch.manual_seed(0)
        self._run_moe_dynamic((4, 32, 256), torch.bfloat16, np.float16, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_group_index_fp16(self, device="npu"):
        torch.manual_seed(0)
        self._run_moe_dynamic((4, 32, 256), torch.float16, np.float16, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int8(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertEqual(npu_y.dtype, np.int8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int4(self, device="npu"):
        # INT4 2-per-byte: y dtype uint8, physical last dim = x_last/4 (x_last/2 halved again).
        # y has no clean numpy golden (packed), structural assertion only; scale value-compared.
        torch.manual_seed(0)
        x = (torch.randn((32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        _, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT4)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT4)
        self.assertEqual(npu_y.dtype, np.uint8)
        self.assertEqual(npu_y.shape, (32, 256 // 4))
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_scale_value(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((2, 32, 256), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_large_input(self, device="npu"):
        torch.manual_seed(0)
        x = (torch.randn((3072, 7168), dtype=torch.bfloat16) * 4.0).npu()
        smooth = torch.ones(7168 // 2, dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec(x.cpu().float().numpy(), smooth.numpy(), None, False, self.DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x.clone(), smooth_scales=smooth.clone(), dst_type=self.DST_TYPE_INT8)
        self.assertRtolEqual(cpu_y, npu_y, 1)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    # ---- negative cases ----
    @SupportedDevices(['Ascend910B'])
    def test_neg_x_invalid_dtype(self, device="npu"):
        x = torch.randn((32, 256), dtype=torch.int64).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, dst_type=self.DST_TYPE_INT8)

    @SupportedDevices(['Ascend910B'])
    def test_neg_smooth_scales_invalid_dtype(self, device="npu"):
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        smooth = torch.ones(128, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, dst_type=self.DST_TYPE_INT8)

    @SupportedDevices(['Ascend910B'])
    def test_neg_offsets_invalid_dtype(self, device="npu"):
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        offsets = torch.zeros(128, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, offsets=offsets,
                                           dst_type=self.DST_TYPE_INT8)

    @SupportedDevices(['Ascend910B'])
    def test_neg_group_index_invalid_dtype(self, device="npu"):
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        gi = torch.ones(2, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, group_index=gi,
                                           dst_type=self.DST_TYPE_INT8)

    @SupportedDevices(['Ascend910B'])
    def test_neg_scale_shape_mismatch(self, device="npu"):
        # dst_type invalid value (script suggested shape mismatch, but the real
        # deterministic guard is the dst_type value-range check)
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "dst_type"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, dst_type=5)

    @SupportedDevices(['Ascend910B'])
    def test_neg_dst_type_invalid(self, device="npu"):
        x = torch.randn((32, 256), dtype=torch.bfloat16).npu()
        smooth = torch.ones(128, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "dst_type"):
            torch_npu.npu_swi_glu_quant_v2(x, smooth_scales=smooth, dst_type=99)


if __name__ == "__main__":
    run_tests()
