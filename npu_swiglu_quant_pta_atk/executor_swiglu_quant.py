# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import torch
from copy import deepcopy
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
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclTensorlistStruct, AclFormat

@register("function_npu_swiglu_quant")
class TorchSwiGluQuant(BaseApi):

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

    def npu_int4_unpack_to_golden(self, y_uint8, golden_shape):
        """NPU int4(dst_type=29) 输出为 uint8 打包格式：每字节按 (低4位, 高4位) 顺序
        承载两个有符号 int4，整张 golden 的 int4 值按行优先展平后打包进 uint8 序列，
        尾部为 padding。解包 = 交错拆低/高 nibble + 符号扩展 + 取前 prod(golden_shape) 个
        + reshape 成 golden 形状（末维 = x末维/2）。已用 case13 实测与 golden 逐值一致。
        """
        flat = y_uint8.reshape(-1).to(torch.int32).cpu().numpy().astype(np.int32)
        low = flat & 0xF
        high = (flat >> 4) & 0xF
        low = np.where((low & 0x8) != 0, low - 16, low)
        high = np.where((high & 0x8) != 0, high - 16, high)
        inter = np.empty(flat.size * 2, dtype=np.int32)
        inter[0::2] = low
        inter[1::2] = high
        n_valid = int(np.prod(golden_shape))
        out = inter[:n_valid].astype(np.int8)
        return torch.from_numpy(out).reshape(golden_shape)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if 0 in input_data.kwargs["x"].shape:
            return []

        set_device = f"npu:{getattr(self, 'device_id', 0)}" if self.device == "npu" else "cpu"

        if self.device != "npu":
            # ---- CPU golden (original CANN reference logic, unchanged) ----
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

        # ---- NPU: call torch_npu.npu_swiglu_quant per prototype ----
        # 接口原型（_op_plugin_docs.py / 已安装 torch_npu 一致）：
        #   torch_npu.npu_swiglu_quant(Tensor x, *, Tensor? smooth_scales=None,
        #     Tensor? offsets=None, Tensor? group_index=None, bool activate_left=False,
        #     str quant_mode="dynamic", int group_list_type=0, int dst_type=2) -> (Tensor, Tensor)
        quant_mode = input_data.kwargs["quantModeOptional"]          # str: "dynamic"/"static"
        dst_type = int(input_data.kwargs["dstType"])                 # int: 2=int8 / 29=int4
        activate_left = bool(input_data.kwargs["activate_left"])
        group_list_type = int(input_data.kwargs["groupListType"])

        x_npu = deepcopy(input_data.kwargs["x"]).to(set_device)
        # smooth_scales 必传：缺失时 kernel 报 507057 / AICore MTE 越界
        smooth_scales_npu = deepcopy(input_data.kwargs["smooth_scales"]).to(set_device)
        group_index_npu = deepcopy(input_data.kwargs["group_index_t"]).to(set_device)
        # offsets 仅静态量化生效；动态量化传 None
        if quant_mode == "dynamic":
            offsets_npu = None
        else:
            offsets_npu = deepcopy(input_data.kwargs["offsets_t"]).to(set_device)

        y_out, scale_out = torch_npu.npu_swiglu_quant(
            x_npu,
            smooth_scales=smooth_scales_npu,
            offsets=offsets_npu,
            group_index=group_index_npu,
            activate_left=activate_left,
            quant_mode=quant_mode,
            group_list_type=group_list_type,
            dst_type=dst_type,
        )

        # kernel 对 int4(dst_type=29) 返回 uint8 打包格式（低/高 nibble 各一个有符号 int4），
        # 解包到 golden 布局：y 末维 = x末维/2，其余维与 x 一致。
        if dst_type == 29:
            x_shape = tuple(input_data.kwargs["x"].shape)
            golden_shape = x_shape[:-1] + (x_shape[-1] // 2,)
            y_out = self.npu_int4_unpack_to_golden(y_out, golden_shape)
            return y_out, scale_out.cpu()

        return y_out.cpu(), scale_out.cpu()