import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestFFN(TestCase):

    def deq_scale_generate(self, deq_scale_shape):
        fp32_deq_scale = np.random.uniform(low=0.01, high=0.05, size=deq_scale_shape).astype(np.float32)
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        # Perform AND operation with the upper 19 bits to simulate hardware
        uint32_deq_scale &= 0XFFFFE000
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        return fp32_deq_scale, uint64_deq_scale

    def calc_ffn(self, x, weight1, weight2, activation, *, antiquant_scale1=None, antiquant_scale2=None,
                 antiquant_offset1=None, antiquant_offset2=None, scale=None, offset=None, deq_scale1=None,
                 deq_scale2=None):
        if antiquant_scale1 is not None:
            x = x.to(torch.float32)
            weight1 = weight1.to(torch.float16)
            antiquant_offset1 = torch.broadcast_to(antiquant_offset1, weight1.size())
            antiquant_scale1 = torch.broadcast_to(antiquant_scale1, weight1.size())
            weight1 = (weight1 + antiquant_offset1) * antiquant_scale1
            weight1 = weight1.to(torch.float32)

            weight2 = weight2.to(torch.float16)
            antiquant_offset2 = torch.broadcast_to(antiquant_offset2, weight2.size())
            antiquant_scale2 = torch.broadcast_to(antiquant_scale2, weight2.size())
            weight2 = (weight2 + antiquant_offset2) * antiquant_scale2
            weight2 = weight2.to(torch.float32)
        elif scale is not None:
            x = x.to(torch.int32)
            weight1 = weight1.to(torch.int32)
            weight2 = weight2.to(torch.int32)
            
        # mm1
        mm1_res = (x @ weight1).to(torch.float16)
        # activation
        if activation == "relu":
            activation_res = torch.nn.functional.relu(mm1_res)
        elif activation == "gelu":
            activation_res = torch.nn.functional.gelu(mm1_res)
        elif activation == "silu":
            activation_res = torch.nn.functional.silu(mm1_res)
        if antiquant_scale1 is not None:
            activation_res = activation_res.to(torch.float32)
        elif scale is not None:
            deq_scale1 = deq_scale1.reshape(1, -1)[:, :mm1_res.shape[-1]]
            deq_scale1 = torch.from_numpy(deq_scale1)
            mm1_res = (mm1_res * deq_scale1).to(torch.float16)
            scale = scale.to(torch.float16)
            offset = offset.to(torch.float16)
            activation_res = activation_res * scale + offset
            activation_res = torch.round(activation_res, decimals=0)
            activation_res = activation_res.clamp(-5, 5).to(torch.int32)
        # mm2
        mm2_res = (activation_res @ weight2).to(torch.float16)
        if scale is not None:
            deq_scale2 = deq_scale2.reshape(1, -1)[:, :mm2_res.shape[-1]]
            deq_scale2 = torch.from_numpy(deq_scale2)    
            mm2_res = (mm2_res * deq_scale2).to(torch.float16)
        return mm2_res


    def supported_op_exec(self, x, weight1, weight2, activation, *, expert_tokens=None, **kwargs):
        if expert_tokens:
            x = x.split(expert_tokens, dim=0)
            y = []
            for idx, x_i in enumerate(x):
                y_i = self.calc_ffn(x_i, weight1[idx], weight2[idx], activation, **kwargs)
                y.append(y_i)
            y = torch.cat(y)
        else:
            y = self.calc_ffn(x, weight1, weight2, activation, **kwargs)
        
        return y

    def custom_op_exec(self, x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None,
                       antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None,
                       scale=None, offset=None, deq_scale1=None, deq_scale2=None, output_dtype=None):
        if antiquant_scale1 is not None:
            antiquant_scale1 = antiquant_scale1.npu()
            antiquant_scale2 = antiquant_scale2.npu()
            antiquant_offset1 = antiquant_offset1.npu()
            antiquant_offset2 = antiquant_offset2.npu()
            return torch_npu.npu_ffn(x, weight1, weight2, activation, expert_tokens=expert_tokens,
                                     expert_tokens_index=expert_tokens_index, antiquant_scale1=antiquant_scale1,
                                     antiquant_scale2=antiquant_scale2, antiquant_offset1=antiquant_offset1,
                                     antiquant_offset2=antiquant_offset2, inner_precise=1, output_dtype=output_dtype)
        elif scale is not None:
            scale = scale.npu()
            offset = offset.npu()
            deq_scale1 = torch.from_numpy(deq_scale1.astype(np.int64)).npu()
            deq_scale2 = torch.from_numpy(deq_scale2.astype(np.int64)).npu()
            return torch_npu.npu_ffn(x, weight1, weight2, activation, expert_tokens=expert_tokens,
                                     expert_tokens_index=expert_tokens_index, scale=scale, offset=offset,
                                     deq_scale1=deq_scale1, deq_scale2=deq_scale2, inner_precise=1,
                                     output_dtype=output_dtype)
        else:
            return torch_npu.npu_ffn(x, weight1, weight2, activation, expert_tokens=expert_tokens,
                                     expert_tokens_index=expert_tokens_index, inner_precise=1)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn(self, device="npu"):
        torch.manual_seed(0)
        x = torch.normal(mean=0., std=0.01, size=(8192, 320), dtype=torch.float16).npu()
        weight1 = torch.normal(mean=0., std=0.01, size=(320, 2560), dtype=torch.float16).npu()
        weight2 = torch.normal(mean=0., std=0.01, size=(2560, 320), dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "silu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_expert_tokens(self, device="npu"):
        torch.manual_seed(0)
        BS, N, H, E = 340, 2560, 5120, 16
        x = torch.normal(mean=0., std=0.01, size=(BS, H), dtype=torch.float16).npu()
        weight1 = torch.normal(mean=0., std=0.01, size=(E, H, N), dtype=torch.float16).npu()
        weight2 = torch.normal(mean=0., std=0.01, size=(E, N, H), dtype=torch.float16).npu()
        expert_tokens = [50, 15, 4, 15, 20, 21, 36, 15, 25, 21, 15, 10, 19, 30, 24, 20]
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "gelu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, expert_tokens=expert_tokens)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation,
                                            expert_tokens=expert_tokens)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_expert_tokens_index(self, device="npu"):
        torch.manual_seed(0)
        BS, N, H, E = 340, 2560, 5120, 16
        x = torch.normal(mean=0., std=0.01, size=(BS, H), dtype=torch.float16).npu()
        weight1 = torch.normal(mean=0., std=0.01, size=(E, H, N), dtype=torch.float16).npu()
        weight2 = torch.normal(mean=0., std=0.01, size=(E, N, H), dtype=torch.float16).npu()
        expert_tokens = [50, 15, 4, 15, 20, 21, 36, 15, 25, 21, 15, 10, 19, 30, 24, 20]
        expert_tokens_index = np.cumsum(expert_tokens).tolist()
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "gelu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, expert_tokens=expert_tokens)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation,
                                            expert_tokens_index=expert_tokens_index)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_antiquant(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.normal(mean=0., std=0.1, size=(8192, 320), dtype=torch.float16).npu()
        weight1 = torch.from_numpy(np.random.randint(-5, 5, size=(320, 2560), dtype=np.int8)).npu()
        weight2 = torch.from_numpy(np.random.randint(-5, 5, size=(2560, 320), dtype=np.int8)).npu()
        antiquant_scale1 = torch.normal(mean=0., std=0.2, size=(2560,), dtype=torch.float16).npu()
        antiquant_scale2 = torch.normal(mean=0., std=0.2, size=(320,), dtype=torch.float16).npu()
        antiquant_offset1 = torch.normal(mean=0., std=0.2, size=(2560,), dtype=torch.float16).npu()
        antiquant_offset2 = torch.normal(mean=0., std=0.2, size=(320,), dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = weight1.npu()
        weight2_clone = weight2.npu()
        antiquant_scale1_clone = antiquant_scale1.clone()
        antiquant_scale2_clone = antiquant_scale2.clone()
        antiquant_offset1_clone = antiquant_offset1.clone()
        antiquant_offset2_clone = antiquant_offset2.clone()
        activation = "relu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation,
                                                  antiquant_scale1=antiquant_scale1,
                                                  antiquant_scale2=antiquant_scale2,
                                                  antiquant_offset1=antiquant_offset1,
                                                  antiquant_offset2=antiquant_offset2)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation,
                                            antiquant_scale1=antiquant_scale1_clone,
                                            antiquant_scale2=antiquant_scale2_clone,
                                            antiquant_offset1=antiquant_offset1_clone,
                                            antiquant_offset2=antiquant_offset2_clone)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_quant(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.from_numpy(np.random.randint(-5, 5, size=(8192, 320), dtype=np.int8))
        weight1 = torch.from_numpy(np.random.randint(-5, 5, size=(320, 2560), dtype=np.int8))
        weight2 = torch.from_numpy(np.random.randint(-5, 5, size=(2560, 320), dtype=np.int8))
        scale = torch.ones(1, dtype=torch.float32)
        offset = torch.zeros(1, dtype=torch.float32)
        deq_scale1_fp32, deq_scale1_uint64 = self.deq_scale_generate((2560))
        deq_scale2_fp32, deq_scale2_uint64 = self.deq_scale_generate((320))

        x_clone = x.clone().npu()
        weight1_clone = weight1.clone().npu()
        weight2_clone = weight2.clone().npu()
        scale_clone = scale.clone().npu()
        offset_clone = offset.clone().npu()
        activation = "gelu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, scale=scale, offset=offset,
                                                  deq_scale1=np.expand_dims(deq_scale1_fp32, axis=0),
                                                  deq_scale2=np.expand_dims(deq_scale2_fp32, axis=0))
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation, scale=scale_clone,
                                            offset=offset_clone, deq_scale1=deq_scale1_uint64,
                                            deq_scale2=deq_scale2_uint64)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

    @SupportedDevices(['Ascend910B'])
    def test_npu_ffn_quant_with_outputdtype(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.from_numpy(np.random.randint(-5, 5, size=(8192, 320), dtype=np.int8))
        weight1 = torch.from_numpy(np.random.randint(-5, 5, size=(320, 2560), dtype=np.int8))
        weight2 = torch.from_numpy(np.random.randint(-5, 5, size=(2560, 320), dtype=np.int8))
        scale = torch.ones(1, dtype=torch.float32)
        offset = torch.zeros(1, dtype=torch.float32)
        deq_scale1_fp32, deq_scale1_uint64 = self.deq_scale_generate((2560))
        deq_scale2_fp32, deq_scale2_uint64 = self.deq_scale_generate((320))

        x_clone = x.clone().npu()
        weight1_clone = weight1.clone().npu()
        weight2_clone = weight2.clone().npu()
        scale_clone = scale.clone().npu()
        offset_clone = offset.clone().npu()
        activation = "gelu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, scale=scale, offset=offset,
                                                  deq_scale1=np.expand_dims(deq_scale1_fp32, axis=0),
                                                  deq_scale2=np.expand_dims(deq_scale2_fp32, axis=0))
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation, scale=scale_clone,
                                            offset=offset_clone, deq_scale1=deq_scale1_uint64,
                                            deq_scale2=deq_scale2_uint64, output_dtype=torch.float16)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)

if __name__ == "__main__":
    run_tests()
