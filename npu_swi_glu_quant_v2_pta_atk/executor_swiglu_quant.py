# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

import torch.nn.functional as F
import torch_npu
import argparse
import numpy as np
from copy import deepcopy
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclTensorlistStruct, AclFormat

@register("function_npu_swiglu_quant")
class TorchSwiGluQuant(BaseApi):

    def __init__(self, task_result: TaskResult):
        super(TorchSwiGluQuant, self).__init__(task_result)
        if self.device == "npu":
            self.set_device = f"npu:{self.device_id}"
        elif self.device == "cpu":
            self.set_device = "cpu"
        else:
            self.set_device = self.device

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def generate_swiglu_input_data(self, input_data: InputDataset):
        data_x = input_data.kwargs["x"].to(torch.float).numpy().astype(np.float32)
        tiling_key = "0"
        if self.dtype == torch.float16:  # fp16
            tiling_key = "1"
        elif self.dtype == torch.bfloat16:  # bf16
            tiling_key = "2"
        elif self.dtype == torch.float32:  # fp32
            tiling_key = "3"
        a, b = np.split(data_x, 2, axis = -1)
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        return a, b, tiling_key

    def do_swiglu(self, a, b):
        if self.activate_left:
            sigmoid_mul_a = self.sigmoid(a) * a
            swiglu_y = sigmoid_mul_a * b
        else:
            sigmoid_mul_a = self.sigmoid(b) * b
            swiglu_y = sigmoid_mul_a * a
        return swiglu_y

    def generate_quant_input_data(self, tiling_key, input_data: InputDataset):
        scales, offsets, group_index = None, None, None

        offsets = input_data.kwargs["offsets_t"].numpy().astype(np.float32)
        scales = input_data.kwargs["smooth_scales"].numpy().astype(np.float32)
        group_index = input_data.kwargs["group_index_t"].numpy().astype(np.int32)

        if self.group_length >= 1:
            if self.quant_mode == "dynamic":  # 有group index场景, 动态量化
                tiling_key += "06"
            else:
                if self.static_mode == "per_tensor":  # 有group index场景, 静态per_tensor量化
                    tiling_key += "04"
                else:  # 有group index场景, 静态per_channel量化
                    tiling_key += "05"

        return scales, offsets, group_index, tiling_key

    def do_quant_06(self, swiglu_y, scales, group_index):
        ori_shape = swiglu_y.shape
        prefix_shape = swiglu_y.shape[:-1]
        prefix_dims = int(np.prod(prefix_shape))
        last_shape = swiglu_y.shape[-1]

        y_tmp = swiglu_y.copy()
        swiglu_y_tmp = swiglu_y.copy()
        y_tmp = y_tmp.reshape(prefix_dims, last_shape)
        swiglu_y_tmp = swiglu_y_tmp.reshape(prefix_dims, last_shape)
        y_tmp[0:group_index[0], :] = swiglu_y_tmp[0:group_index[0], :] * scales[0, :]
        for i in range(self.group_length - 1):
            y_tmp[group_index[i]:group_index[i + 1], :] = swiglu_y_tmp[group_index[i]:group_index[i + 1], :] * scales[
                                                                                                           i + 1, :]

        y_max = np.amax(np.abs(y_tmp), axis=-1, keepdims=True)  # 动态量化依赖于每一行的最大值来调整缩放因子
        int_scale = 127
        if self.dst_type == 29:
            int_scale = 7
        dynamic_scale = int_scale / y_max

        quant_y_tmp = np.round(y_tmp * dynamic_scale)  # 使用 numpy 进行乘法和四舍五入
        quant_y = quant_y_tmp.astype(np.int8)  # 转换为 int8 类型
        dynamic_scale = dynamic_scale[..., 0]
        dynamic_scale = dynamic_scale.reshape(prefix_shape)
        quant_y = quant_y.reshape(ori_shape)
        return quant_y, dynamic_scale

    def do_quant_no6(self, swiglu_y, scales, group_index , offsets):
        ori_shape = swiglu_y.shape
        prefix_shape = swiglu_y.shape[:-1]
        prefix_dims = int(np.prod(prefix_shape))
        last_shape = swiglu_y.shape[-1]
        swiglu_y_tmp = swiglu_y.copy()

        y_tmp = swiglu_y.copy()
        y_tmp = y_tmp.reshape(prefix_dims, last_shape)
        swiglu_y_tmp = swiglu_y_tmp.reshape(prefix_dims, last_shape)

        if self.static_mode == 'per_tensor':
            y_tmp[0: group_index[0], :] = swiglu_y_tmp[0: group_index[0], :] * scales[0]
            if self.group_length > 1:
                for i in range(self.group_length - 1):
                    y_tmp[group_index[i]: group_index[i + 1], :] = swiglu_y_tmp[group_index[i]: group_index[i + 1], :] * \
                                                                   scales[
                                                                       i + 1]
            y_tmp[0: group_index[0], :] = y_tmp[0: group_index[0], :] + offsets[0]
            if self.group_length > 1:
                for i in range(self.group_length - 1):
                    y_tmp[group_index[i]: group_index[i + 1], :] = y_tmp[group_index[i]: group_index[i + 1], :] + \
                                                                   offsets[
                                                                       i + 1]

        else:
            y_tmp[0:group_index[0], :] = swiglu_y_tmp[0:group_index[0], :] * scales[0, :] + offsets[0, :]
            for i in range(self.group_length - 1):
                y_tmp[group_index[i]:group_index[i + 1], :] = swiglu_y_tmp[group_index[i]:group_index[i + 1], :] * scales[
                                                                                                               i + 1,
                                                                                                               :] + offsets[
                                                                                                                    i + 1,
                                                                                                                    :]

        int_scale = 127
        if self.dst_type == 29:
            int_scale = 7
            y_tmp = np.round(y_tmp)
            y_tmp = np.clip(y_tmp, -8, 7)
        else:
            y_tmp = np.round(y_tmp)
            y_tmp = np.clip(y_tmp, -128, 127)

        dynamic_scale = int_scale / y_tmp
        quant_y = y_tmp.astype(np.int8)  # 转换为 int8 类型
        dynamic_scale = dynamic_scale[..., 0]
        dynamic_scale = dynamic_scale.reshape(prefix_shape)
        quant_y = quant_y.reshape(ori_shape)
        return quant_y, dynamic_scale

    def do_quant(self, swiglu_y, scales, offsets, group_index, tiling_key):
        quant_y = 0
        dynamic_scale = 0
        if tiling_key == "206" or tiling_key == "106" or tiling_key == "306":  # BF16, 有group index场景, 动态量化
            quant_y, dynamic_scale = self.do_quant_06(swiglu_y, scales, group_index)
        else:
            quant_y, dynamic_scale = self.do_quant_no6(swiglu_y, scales, group_index,offsets)
        return quant_y, dynamic_scale

    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["x"]
        if 0 not in input_data.kwargs["x"].shape:
            smooth_scales = input_data.kwargs["smooth_scales"]
            offsets_t = input_data.kwargs["offsets_t"]
            activate_left = input_data.kwargs["activate_left"]
            quantModeOptional = input_data.kwargs["quantModeOptional"]
            group_index_t = input_data.kwargs["group_index_t"]
            staticMode = input_data.kwargs["staticMode"]
            groupListType = input_data.kwargs["groupListType"]

            self.dtype = x.dtype
            self.activate_left = activate_left
            self.row_len = np.prod(x.shape[:-1])
            self.col_len = x.shape[-1]
            self.group_length = np.prod(group_index_t.shape)
            self.quant_mode = quantModeOptional
            self.static_mode = staticMode
            self.dst_type = input_data.kwargs['dstType']

            group_index = None
            np.random.seed(0)
            group_index = np.random.choice(range(1, self.row_len - 1), int(self.group_length - 1), replace=False).astype(np.int32)
            group_index = np.sort(group_index)
            group_index = np.append(group_index, self.row_len).astype(np.int32)

            if groupListType == 0:
                group_index = np.cumsum(group_index)
            elif self.device == "cpu":
                group_index = np.cumsum(group_index)

            if self.device == "cpu":
                input_data.kwargs['group_index_t'] = torch.from_numpy(group_index).to(torch.int32)
            if self.device == "pyaclnn" or self.device == "npu":
                input_data.kwargs['group_index_t'] = torch.from_numpy(group_index).to(torch.int32).npu()

            if self.static_mode == "per_tensor" :
                smooth_scales = smooth_scales.unsqueeze(-1)
                offsets_t = offsets_t.unsqueeze(-1)

    def pta_tensor_int8_to_int4s(self, tensor_int8):
        tensor_array = tensor_int8.reshape(1, tensor_int8.numel())[0]
        int8_array = tensor_array[0:tensor_int8.numel()].view(torch.int8).cpu().numpy()
        masks = np.array([0xF << (i * 4) for i in range(2)], dtype=np.uint8)
        shifted = (int8_array[:, None] & masks) >> np.arange(0, 8, 4)
        sign_extended = np.where(shifted & 0x8, shifted - 16, shifted)
        x_shape = tensor_int8.shape
        last_dim = x_shape[-1]
        new_shape = x_shape[:-1] + (last_dim * 2,)
        out = torch.tensor(sign_extended.astype(np.int32)).to(torch.int8).reshape(new_shape)
        return out

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # shape guard applies to both backends
        if 0 in input_data.kwargs["x"].shape:
            return []

        # --- CPU golden path (reference output for ATK accuracy compare) ---
        if self.device == "cpu":
            # 1. 首先计算swiglu
            input_a, input_b, tiling_key = self.generate_swiglu_input_data(input_data)
            swiglu_y = self.do_swiglu(input_a, input_b)
            # 2. 执行quant操作
            scales, offsets, group_index, tiling_key = self.generate_quant_input_data(tiling_key, input_data)
            quant_y, dynamic_scale = self.do_quant(swiglu_y, scales, offsets, group_index, tiling_key)
            sp = dynamic_scale.shape
            scale = np.zeros(sp, dtype=np.float32)
            if input_data.kwargs["quantModeOptional"] == "dynamic":
                return torch.from_numpy(quant_y), torch.from_numpy(dynamic_scale)
            else:
                return torch.from_numpy(quant_y), torch.from_numpy(scale)

        # --- NPU path: call torch_npu.npu_swi_glu_quant_v2 (V2 op) ---
        # NOTE: CANN ATK case was authored for V1 (npu_swiglu_quant + int quantMode +
        # torch-dtype dst_type). V2 (npu_swi_glu_quant_v2) takes str quant_mode and raw
        # int dst_type (2=int8, 29=int4). All tensor inputs: deepcopy + .to(npu).
        elif self.device == "npu":
            x_npu = deepcopy(input_data.kwargs["x"]).to(self.set_device)
            smooth_scales_npu = deepcopy(input_data.kwargs["smooth_scales"]).to(self.set_device)
            # offsets only meaningful in static mode; pass None for dynamic (matches golden
            # do_quant_06 which ignores offsets in dynamic quantization).
            offsets_t = input_data.kwargs.get("offsets_t")
            offsets_npu = deepcopy(offsets_t).to(self.set_device) if offsets_t is not None else None
            # group_index_t already moved to npu in init_by_input_data for npu backend;
            # deepcopy + .to(npu) is a safe idempotent guard.
            group_index_npu = deepcopy(input_data.kwargs["group_index_t"]).to(self.set_device)

            activate_left = input_data.kwargs["activate_left"]
            quant_mode = input_data.kwargs["quantModeOptional"]   # str: "dynamic" / "static"
            group_list_type = input_data.kwargs["groupListType"]   # int: 0 (cumsum) / 1 (count)
            dst_type = input_data.kwargs["dstType"]                # int: 2 (int8) / 29 (int4)

            y_out, scales = torch_npu.npu_swi_glu_quant_v2(
                x_npu,
                smooth_scales=smooth_scales_npu,
                offsets=offsets_npu,
                group_index=group_index_npu,
                activate_left=activate_left,
                quant_mode=quant_mode,
                group_list_type=group_list_type,
                dst_type=dst_type,
            )

            # int4 (dst_type=29): npu returns 2-per-byte packed uint8; unpack to
            # one-int4-per-int8 for golden comparison (golden stores int4 in int8).
            if dst_type == 29:
                y_out = self.pta_tensor_int8_to_int4s(y_out)

            if quant_mode == "dynamic":
                return (y_out, scales)
            else:
                return (y_out, torch.zeros_like(scales))