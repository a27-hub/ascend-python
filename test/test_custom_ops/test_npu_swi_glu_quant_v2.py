import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

MAX_VALUE_WITH_INT8 = 127
INT4_MAX = 7
# dst_type follows the ge::DataType convention: 2 -> INT8, 29 -> INT4
DST_TYPE_INT8 = 2
DST_TYPE_INT4 = 29


def swiglu_act(x, activate_left):
    """SwiGlu activation: Swish(A) * B. A/B are the two halves of the last dim.
    activate_left=False -> swish on the SECOND half (matches CANN semantics)."""
    n = x.shape[-1] // 2
    if activate_left:
        a = x[..., :n]
        b = x[..., n:]
    else:
        a = x[..., n:]
        b = x[..., :n]
    return torch.nn.functional.silu(a.float()) * b.float()


class TestNPUSwiGluQuantV2(TestCase):
    def cpu_op_exec_dynamic(self, x, smooth_scales, group_index, activate_left, dst_type):
        """CPU golden for the dynamic per-row quantization path (non-MoE or MoE cumsum).

        scale = dst_scale / row_max(abs(Y_tmp));  Y = round(Y_tmp * scale)
        dst_scale = 127 (int8) or 7 (int4). NPU returns the quant scale (NOT dequant).
        group_index (cumsum mode): cumulative row boundaries [g0, g1, ...].
        Returns (y_int8_np, scale_np).
        """
        x2d = x.reshape(-1, x.shape[-1]).float()
        act = swiglu_act(x2d, activate_left)  # [rows, n]
        rows = act.shape[0]
        # Op output shape convention: y = x.shape[:-1] + (n,); scale = x.shape[:-1].
        out_y_shape = list(x.shape[:-1]) + [act.shape[1]]
        out_scale_shape = list(x.shape[:-1])
        if smooth_scales is None:
            smooth = torch.ones((1, act.shape[1]), dtype=torch.float32)
        else:
            smooth = smooth_scales.float()
            if smooth.dim() == 1 and smooth.shape[0] == act.shape[1]:
                smooth = smooth.unsqueeze(0)
        dst_scale = MAX_VALUE_WITH_INT8 if dst_type == DST_TYPE_INT8 else INT4_MAX

        if group_index is None:
            # non-MoE: smooth broadcast over all rows
            if smooth.dim() == 2 and smooth.shape[0] in (1, rows):
                y_tmp = act * smooth
            else:
                y_tmp = act * smooth[:1]
            row_max = torch.amax(torch.abs(y_tmp), dim=-1)
            scale = dst_scale / row_max
            y = torch.clamp(torch.round(y_tmp * scale.unsqueeze(1)), min=-dst_scale, max=dst_scale)
            return y.to(torch.int8).reshape(out_y_shape).numpy(), scale.reshape(out_scale_shape).numpy()

        # MoE cumsum path: group_index = cumulative boundaries
        boundaries = group_index if isinstance(group_index, torch.Tensor) else torch.tensor(group_index)
        y_out = torch.zeros_like(act)
        scale_out = torch.zeros(rows, dtype=torch.float32)
        offset = 0
        for g in range(boundaries.shape[0]):
            end = boundaries[g].item()
            seg = act[offset:end]
            seg_smooth = smooth[g:g + 1] if (smooth.dim() == 2 and smooth.shape[0] == boundaries.shape[0]) else smooth[:1]
            y_seg = seg * seg_smooth
            seg_scale = dst_scale / torch.amax(torch.abs(y_seg), dim=-1)
            y_out[offset:end] = y_seg * seg_scale.unsqueeze(1)
            scale_out[offset:end] = seg_scale
            offset = end
        y_out = torch.clamp(torch.round(y_out), min=-dst_scale, max=dst_scale)
        return y_out.to(torch.int8).reshape(out_y_shape).numpy(), scale_out.reshape(out_scale_shape).numpy()

    def npu_op_exec(self, x, smooth_scales=None, offsets=None, group_index=None,
                    activate_left=False, quant_mode="dynamic", group_list_type=0, dst_type=DST_TYPE_INT8):
        x = x.npu()
        if smooth_scales is not None:
            smooth_scales = smooth_scales.npu()
        if offsets is not None:
            offsets = offsets.npu()
        if group_index is not None:
            group_index = group_index.npu()
        y, scale = torch_npu.npu_swi_glu_quant_v2(
            x, smooth_scales=smooth_scales, offsets=offsets, group_index=group_index,
            activate_left=activate_left, quant_mode=quant_mode, group_list_type=group_list_type, dst_type=dst_type)
        return y.cpu().numpy(), scale.cpu().numpy()

    def gen_x(self, shape, dtype):
        return torch.randn(shape, dtype=dtype)

    # ---- Basic functionality (dynamic, int8, non-MoE) ----
    @SupportedDevices(['Ascend910B'])
    def test_basic_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)
        self.assertRtolEqual(cpu_y, npu_y, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.float16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s, 0.001)
        self.assertRtolEqual(cpu_y, npu_y, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_basic_fp32(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.float32)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s)
        self.assertRtolEqual(cpu_y, npu_y)

    # ---- Optional parameter combinations (dynamic, int8) — shape/dtype checks ----
    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, offsets=None, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))
        self.assertEqual(npu_s.shape, (2, 32))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_group_index_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, quant_mode="static",
                                        offsets=torch.zeros((1, 128), dtype=torch.float32), dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_offsets_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, quant_mode="static",
                                        offsets=torch.zeros((1, 128), dtype=torch.float32), dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_group_index_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_smooth_scales_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_offsets_group_index_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_offsets_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, quant_mode="static",
                                        offsets=torch.zeros((1, 128), dtype=torch.float32), dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_offsets_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, quant_mode="static",
                                        offsets=torch.zeros((1, 128), dtype=torch.float32), dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_group_index_bf16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    @SupportedDevices(['Ascend910B'])
    def test_group_index_fp16(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float16)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 64], dtype=torch.int32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))

    # ---- dst_type coverage ----
    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int8(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.dtype, np.int8)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_dst_type_int4(self):
        # INT4 output: 2-per-byte packed as uint8. Structure/dtype/shape check (no int4 golden).
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT4)
        self.assertEqual(npu_y.dtype, np.uint8)
        self.assertEqual(npu_y.shape, (32, 64))  # x last dim 256 / 4 (2-per-byte)
        self.assertEqual(npu_s.shape, (32,))

    @SupportedDevices(['Ascend910B'])
    def test_int4_output_shape(self):
        # INT4 2-per-byte: y last dim = x last dim / 4. x last dim must be %4==0.
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((4, 64), torch.bfloat16)
        smooth = torch.ones((1, 32), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT4)
        self.assertEqual(npu_y.shape, (4, 16))  # 64 / 4
        self.assertEqual(npu_y.dtype, np.uint8)

    @SupportedDevices(['Ascend910B'])
    def test_scale_value(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    @SupportedDevices(['Ascend910B'])
    def test_large_input(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((8, 8192), torch.bfloat16)  # last dim 8192 (max), even
        smooth = torch.ones((1, 4096), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, False, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s, 0.004)

    # ---- attr value coverage ----
    @SupportedDevices(['Ascend910B'])
    def test_activate_left_true(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.float32)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        cpu_y, cpu_s = self.cpu_op_exec_dynamic(x, smooth, None, True, DST_TYPE_INT8)
        npu_y, npu_s = self.npu_op_exec(x, smooth, activate_left=True, dst_type=DST_TYPE_INT8)
        self.assertRtolEqual(cpu_s, npu_s)
        self.assertRtolEqual(cpu_y, npu_y)

    @SupportedDevices(['Ascend910B'])
    def test_quant_mode_static(self):
        # static quantization: Y = Cast(Y_tmp); offsets required. Shape check.
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.float32)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        offsets = torch.zeros((1, 128), dtype=torch.float32)
        npu_y, npu_s = self.npu_op_exec(x, smooth, offsets=offsets, quant_mode="static", dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (32, 128))
        self.assertEqual(npu_y.dtype, np.int8)

    @SupportedDevices(['Ascend910B'])
    def test_group_list_type_count(self):
        # group_list_type=1 (count mode): group_index elements are per-group lengths.
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((2, 32, 256), torch.float32)
        smooth = torch.ones((2, 128), dtype=torch.float32)
        gidx = torch.tensor([32, 32], dtype=torch.int32)  # count mode: 2 groups of 32 rows
        npu_y, npu_s = self.npu_op_exec(x, smooth, group_index=gidx, group_list_type=1, dst_type=DST_TYPE_INT8)
        self.assertEqual(npu_y.shape, (2, 32, 128))
        self.assertEqual(npu_s.shape, (2, 32))

    # ---- Graph mode (torch.compile) ----
    @SupportedDevices(['Ascend910B'])
    def test_graph_dynamic_int8(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((32, 256), torch.float32)
        smooth = torch.ones((1, 128), dtype=torch.float32)

        def fn(xx, ss):
            return torch_npu.npu_swi_glu_quant_v2(xx, smooth_scales=ss, dst_type=DST_TYPE_INT8)
        compiled = torch.compile(fn, backend="npugraph_ex")
        y, scale = compiled(x.npu(), smooth.npu())
        y = y.cpu().numpy()
        scale = scale.cpu().numpy()
        self.assertEqual(y.shape, (32, 128))
        self.assertEqual(y.dtype, np.int8)
        self.assertEqual(scale.shape, (32,))

    @SupportedDevices(['Ascend910B'])
    def test_graph_static(self):
        torch.manual_seed(0); np.random.seed(0)
        x = self.gen_x((16, 128), torch.float32)
        smooth = torch.ones((1, 64), dtype=torch.float32)
        offsets = torch.zeros((1, 64), dtype=torch.float32)  # static mode requires offsets

        def fn(xx, ss, oo):
            return torch_npu.npu_swi_glu_quant_v2(xx, smooth_scales=ss, offsets=oo,
                                                   quant_mode="static", dst_type=DST_TYPE_INT8)
        compiled = torch.compile(fn, backend="npugraph_ex")
        y, scale = compiled(x.npu(), smooth.npu(), offsets.npu())
        self.assertEqual(y.cpu().numpy().shape, (16, 64))

    # ---- Negative cases ----
    @SupportedDevices(['Ascend910B'])
    def test_neg_x_invalid_dtype(self):
        x = torch.randint(0, 10, (32, 256), dtype=torch.int64)  # int not supported
        smooth = torch.ones((1, 128), dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swi_glu_quant_v2(x.npu(), smooth_scales=smooth.npu())

    @SupportedDevices(['Ascend910B'])
    def test_neg_smooth_scales_invalid_dtype(self):
        x = torch.randn((32, 256), dtype=torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.int32)  # must be FLOAT
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swi_glu_quant_v2(x.npu(), smooth_scales=smooth.npu())

    @SupportedDevices(['Ascend910B'])
    def test_neg_offsets_invalid_dtype(self):
        x = torch.randn((32, 256), dtype=torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        offsets = torch.ones((1, 128), dtype=torch.int32)  # must be FLOAT
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swi_glu_quant_v2(x.npu(), smooth_scales=smooth.npu(),
                                            offsets=offsets.npu(), quant_mode="static")

    @SupportedDevices(['Ascend910B'])
    def test_neg_group_index_invalid_dtype(self):
        x = torch.randn((32, 256), dtype=torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        gidx = torch.zeros((1,), dtype=torch.float32)  # must be INT32
        with self.assertRaises(RuntimeError):
            torch_npu.npu_swi_glu_quant_v2(x.npu(), smooth_scales=smooth.npu(), group_index=gidx.npu())

    @SupportedDevices(['Ascend910B'])
    def test_neg_dst_type_invalid(self):
        x = torch.randn((32, 256), dtype=torch.bfloat16)
        smooth = torch.ones((1, 128), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "dst_type"):
            torch_npu.npu_swi_glu_quant_v2(x.npu(), smooth_scales=smooth.npu(), dst_type=99)


if __name__ == "__main__":
    run_tests()
