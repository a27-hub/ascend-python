import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.config.allow_internal_format = False


class TestSwigluGroupQuant(TestCase):

    def golden_swiglu_group_quant(self, x, weight=None, group_index=None, scale=None,
                                  dst_type=296, quant_mode=0, block_size=0, round_scale=False,
                                  clamp_limit=-1.0, dst_type_max=0.0, output_origin=False):
        input_dtype = x.dtype
        original_shape = x.shape
        H = x.shape[-1] // 2

        x = x.float()
        x = x.reshape(-1, 2 * H)
        x0 = x[:, :H]
        x1 = x[:, H:]

        if clamp_limit != 0:
            x0 = torch.clamp(x0, max=clamp_limit)
            x1 = torch.clamp(x1, -clamp_limit, clamp_limit)

        y = torch.nn.functional.silu(x0) * x1

        output_shape = original_shape[:-1] + (H,)
        if weight is not None:
            weight = weight.reshape(-1, 1)
            y = y * weight

        y_origin = y.to(input_dtype).reshape(output_shape) if output_origin else torch.empty(0, device=x.device)

        eps = 1e-8
        if group_index is None:
            amax = torch.max(torch.abs(y))
            amax = torch.clamp(amax, min=eps)
            print(f"max {amax}")
            y_scale = amax / dst_type_max
            y_quant = torch_npu.npu_dtype_cast(y.npu() / y_scale.npu(), torch_npu.hifloat8)
            y_quant = y_quant.reshape(output_shape)
            return y_quant, y_scale.unsqueeze(0), y_origin
        else:
            num_groups = group_index.shape[0]
            total_tokens = y.shape[0]
            y_scale = torch.zeros(num_groups, dtype=torch.float32, device=x.device)
            y_quant = torch.zeros(total_tokens, H, dtype=torch.uint8, device=x.device)
            start = 0
            for g in range(num_groups):
                end = start + group_index[g].item()
                y_g = y[start:end, :]
                amax_g = torch.max(torch.abs(y_g))
                amax_g = torch.clamp(amax_g, min=eps)
                scale_g = amax_g / dst_type_max
                y_scale[g] = scale_g
                y_g_quant = torch_npu.npu_dtype_cast(y_g.npu() / scale_g.npu(), torch_npu.hifloat8)
                y_quant[start:end, :] = y_g_quant
                start = end
            y_quant = y_quant.reshape(output_shape)
            return y_quant, y_scale, y_origin


    def hifloat8_to_float(self, uint8_tensor):
        bits = uint8_tensor.to(torch.int32)
        sign = (bits >> 7) & 0x1
        sv = torch.where(sign.bool(), -1.0, 1.0)
        remaining = bits & 0x7F

        b6 = (remaining >> 6) & 0x1
        b5 = (remaining >> 5) & 0x1
        b4 = (remaining >> 4) & 0x1
        b3 = (remaining >> 3) & 0x1

        result = torch.zeros_like(bits.float())

        # D=0000: Subnormal, M = bits[2:0] (3 bits)
        mask_d0000 = (b6 == 0) & (b5 == 0) & (b4 == 0) & (b3 == 0)
        m_d0000 = (remaining & 0x7).float()
        result = torch.where(mask_d0000, sv * m_d0000 * (2.0 ** -3), result)

        # D=0001: Ev=0, M = bits[2:0] (3 bits)
        mask_d0001 = (b6 == 0) & (b5 == 0) & (b4 == 0) & (b3 == 1)
        m_d0001 = (remaining & 0x7).float()
        mv_d0001 = m_d0001 * (2.0 ** -3)
        result = torch.where(mask_d0001, sv * (1.0 + mv_d0001), result)

        # D=001: 1 E bit, 3 M bits; Ev = +-1
        mask_d001 = (b6 == 0) & (b5 == 0) & (b4 == 1)
        e_d001 = (remaining >> 3) & 0x1
        m_d001 = (remaining & 0x7).float()
        ev_d001 = torch.where(e_d001.bool(), -1.0, 1.0)
        mv_d001 = m_d001 * (2.0 ** -3)
        result = torch.where(mask_d001, sv * torch.pow(2.0, ev_d001) * (1.0 + mv_d001), result)

        # D=01: 2 E bits, 3 M bits; Ev = +-[2,3]
        mask_d01 = (b6 == 0) & (b5 == 1)
        e_d01 = (remaining >> 3) & 0x3
        m_d01 = (remaining & 0x7).float()
        es_d01 = (e_d01 >> 1) & 0x1
        em_d01 = (e_d01 & 0x1).float()
        ev_d01 = torch.where(es_d01.bool(), -(2.0 + em_d01), 2.0 + em_d01)
        mv_d01 = m_d01 * (2.0 ** -3)
        result = torch.where(mask_d01, sv * torch.pow(2.0, ev_d01) * (1.0 + mv_d01), result)

        # D=10: 3 E bits, 2 M bits; Ev = +-[4,7]
        mask_d10 = (b6 == 1) & (b5 == 0)
        e_d10 = (remaining >> 2) & 0x7
        m_d10 = (remaining & 0x3).float()
        es_d10 = (e_d10 >> 2) & 0x1
        em_d10 = (e_d10 & 0x3).float()
        ev_d10 = torch.where(es_d10.bool(), -(4.0 + em_d10), 4.0 + em_d10)
        mv_d10 = m_d10 * (2.0 ** -2)
        result = torch.where(mask_d10, sv * torch.pow(2.0, ev_d10) * (1.0 + mv_d10), result)

        # D=11: 4 E bits, 1 M bit; Ev = +-[8,15]
        mask_d11 = (b6 == 1) & (b5 == 1)
        e_d11 = (remaining >> 1) & 0xF
        m_d11 = (remaining & 0x1).float()
        es_d11 = (e_d11 >> 3) & 0x1
        em_d11 = (e_d11 & 0x7).float()
        ev_d11 = torch.where(es_d11.bool(), -(8.0 + em_d11), 8.0 + em_d11)
        mv_d11 = m_d11 * (2.0 ** -1)
        result = torch.where(mask_d11, sv * torch.pow(2.0, ev_d11) * (1.0 + mv_d11), result)

        return result


    @SupportedDevices(['Ascend950'])
    def test_swiglu_group_quant(self):
        x = torch.randn([8, 16], dtype=torch.float32)
        weight = torch.randn([8, 1], dtype=torch.float32)
        group_index = torch.randint(1, 10, (2,), dtype=torch.int64)
        scale = torch.randn([2], dtype=torch.float32)

        quant_mode = 3
        round_scale = False
        clamp_limit = 0.01
        dst_type_max = 15.0
        output_origin = True

        y_golden, y_scale_golden, y_origin_golden = self.golden_swiglu_group_quant(x,
                                                                                   weight=weight,
                                                                                   group_index=group_index,
                                                                                   scale=scale,
                                                                                   quant_mode=quant_mode,
                                                                                   round_scale=round_scale,
                                                                                   clamp_limit=clamp_limit,
                                                                                   dst_type_max=dst_type_max,
                                                                                   output_origin=output_origin)

        x_npu = x.npu()
        weight_npu = weight.npu()
        group_index_npu = group_index.npu()
        scale_npu = scale.npu()

        y_npu, y_scale_npu, y_origin_npu = torch_npu.npu_swiglu_group_quant(x_npu,
                                                                            weight=weight_npu,
                                                                            group_index=group_index_npu,
                                                                            scale=scale_npu,
                                                                            quant_mode=quant_mode,
                                                                            round_scale=round_scale,
                                                                            clamp_limit=clamp_limit,
                                                                            dst_type_max=dst_type_max,
                                                                            output_origin=output_origin)
        y_cpu = y_npu.cpu()
        y_scale_cpu = y_scale_npu.cpu()
        y_origin_cpu = y_origin_npu.cpu()

        self.assertRtolEqual(y_golden.type(torch.float32), self.hifloat8_to_float(y_cpu))
        self.assertRtolEqual(y_scale_golden.type(torch.float32), y_scale_cpu.type(torch.float32))
        self.assertRtolEqual(y_origin_golden.type(torch.float32), y_origin_cpu.type(torch.float32))


if __name__ == "__main__":
    run_tests()
