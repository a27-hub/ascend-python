# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import torch
import torch_npu
from torch import Tensor
import numpy as np
import ctypes
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from copy import deepcopy

def generate_group_index(G, max_tokens):
    np.random.seed(G + max_tokens)
    tokens_list = []
    remaining = max_tokens
    for i in range(G - 1):
        max_val = max(1, remaining // (G - i))
        val = np.random.randint(1, max_val + 1)
        tokens_list.append(val)
        remaining -= val
    tokens_list.append(max(1, remaining))
    # tokens_list.append(np.random.randint(1, remaining))
    total = sum(tokens_list)
    if total > max_tokens:
        diff = total - max_tokens
        tokens_list[-1] -= diff
    return tokens_list


def golden_swiglu_group_quant_backward(
    grad_y, x, weight=None, y_origin=None, group_index=None, clamp_limit=0.0
):
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if grad_y.dtype not in supported_dtypes:
        raise ValueError(
            f"grad_y dtype must be float16/bfloat16/float32, got {grad_y.dtype}"
        )
    if x.dtype not in supported_dtypes:
        raise ValueError(f"x dtype must be float16/bfloat16/float32, got {x.dtype}")
    if weight is not None and weight.dtype != torch.float32:
        raise ValueError(f"weight dtype must be float32, got {weight.dtype}")
    if y_origin is not None and y_origin.dtype not in supported_dtypes:
        raise ValueError(
            f"y_origin dtype must be float16/bfloat16/float32, got {y_origin.dtype}"
        )

    input_dtype = x.dtype
    grad_y = grad_y.float()
    x = x.float()
    if y_origin is not None:
        y_origin = y_origin.float()

    grad_weight = None
    trunc = 0
    if group_index is not None:
        for group in group_index:
            trunc += group

    if weight is not None:
        grad_weight = torch.sum(grad_y * y_origin, dim=-1, keepdim=True)
        if group_index is not None:
            original_gw_shape = grad_weight.shape
            grad_weight = grad_weight.reshape([-1, 1])
            num_rows = grad_weight.shape[0]
            row = torch.arange(num_rows)
            mask = (row < trunc).unsqueeze(-1).float()
            grad_weight = grad_weight * mask
            grad_weight = grad_weight.reshape(original_gw_shape)
        grad_y0 = grad_y * weight.float()
    else:
        grad_y0 = grad_y

    original_shape = x.shape
    x = x.reshape([-1, x.shape[-1]])
    H = x.shape[-1] // 2
    x0 = x[:, :H]
    x1 = x[:, H:]

    x0_truncated = x0
    x1_truncated = x1

    if clamp_limit != 0:
        x0 = torch.clamp(x0, max=clamp_limit)
        x1 = torch.clamp(x1, -clamp_limit, clamp_limit)

    sigmoid_x0 = torch.sigmoid(x0)
    silu_x0 = x0 * sigmoid_x0
    silu_grad_x0 = sigmoid_x0 * (1 + x0 * (1 - sigmoid_x0))

    grad_y0_flat = grad_y0.reshape([-1, H])
    grad_x0 = grad_y0_flat * x1 * silu_grad_x0
    grad_x1 = grad_y0_flat * silu_x0

    if clamp_limit != 0:
        mask_x0 = (x0_truncated < clamp_limit).float()
        mask_x1 = ((-clamp_limit < x1_truncated) & (x1_truncated < clamp_limit)).float()
        grad_x0 = grad_x0 * mask_x0
        grad_x1 = grad_x1 * mask_x1

    if group_index is not None:
        num_rows = grad_x0.shape[0]
        row = torch.arange(num_rows)
        mask = (row < trunc).unsqueeze(-1).float()
        grad_x0 = grad_x0 * mask
        grad_x1 = grad_x1 * mask

    grad_x = torch.cat([grad_x0, grad_x1], dim=-1)
    grad_x = grad_x.reshape(original_shape)
    grad_x = grad_x.to(input_dtype)

    return grad_x, grad_weight


@register("function_swiglu_group_quant_backward")
class FunctionSwigluGroupQuantBackward(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionSwigluGroupQuantBackward, self).__init__(task_result)
        if self.device == "npu":
            self.set_device = f"npu:{self.device_id}"
        elif self.device == "cpu":
            self.set_device = f"cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # print(f"pc: input_data = { input_data }")
        grad_y = input_data.kwargs['grad_y']
        x = input_data.kwargs['x']

        weight  = input_data.kwargs.get('weight', None)
        y_origin  = input_data.kwargs.get('y_origin', None)
        group_index  = input_data.kwargs.get('group_index', None)
        clamp_limit = input_data.kwargs.get('clamp_limit', 0.0)


        # print("输入数据打印==========================================")
        # print(f" grad_y: { grad_y}")
        # print(f" x:  { x }")
        # print(f" grad_y: {weight}")
        # print(f" x:  { y_origin }")
        # print(f" grad_y: {weight}")
        # print(f" group_index:  { group_index }")
        # print(f" group_index:  { clamp_limit }")

      

        grad_x, grad_weight = self.npu_swiglu_group_quant_backward_golden(grad_y, x, weight, y_origin, group_index, clamp_limit)

        # print(f" grad_weight: {  grad_weight.size() }")
        # print(f" grad_x:  {  grad_x.size() }")
        # print(f" grad_weight: {  grad_weight }")
        # print(f" grad_x:  {  grad_x }")

        return grad_x, grad_weight

    def npu_swiglu_group_quant_backward_golden(self, grad_y, x, weight, y_origin, group_index, clamp_limit):
        idx = self.task_result.case_config.id
        torch.random.manual_seed(idx)
        out = []
        print(f"weight = {weight.shape}")
        # cpu 
        if self.device == "cpu":
            print("pc: cpu start")
            # Prepare input device
            grad_y_cpu = deepcopy(grad_y).cpu()
            x_cpu = deepcopy(x).cpu()
            weight_cpu = deepcopy(weight).cpu() if weight is not None else None
            y_origin_cpu = deepcopy(y_origin).cpu()  if y_origin is not None else None

            grad_x_cpu, grad_weight_cpu = golden_swiglu_group_quant_backward(grad_y_cpu, x_cpu, weight_cpu, y_origin_cpu, group_index, clamp_limit)

            out = grad_x_cpu, grad_weight_cpu
            # print("pc: cpu end")
            # print("pc: out= ", out)

        # npu
        elif self.device == "npu":
            # Prepare input device
            print("pc: npu start")
            grad_y_npu = deepcopy(grad_y)
            grad_y_npu = grad_y_npu.to(self.set_device)

            x_npu = deepcopy(x)
            x_npu = x_npu.to(self.set_device)

            weight_npu = deepcopy(weight) if weight is not None else None
            weight_npu = weight_npu.to(self.set_device) if weight is not None else None

            y_origin_npu = deepcopy(y_origin) if y_origin is not None else None
            y_origin_npu = y_origin_npu.to(self.set_device) if y_origin is not None else None
            grad_x_npu, grad_weight_npu = torch_npu.npu_swiglu_group_quant_backward(grad_y_npu, x_npu, weight=weight_npu,
                                                                    y_origin=y_origin_npu, group_index=group_index,
                                                                    clamp_limit=clamp_limit)

            out = grad_x_npu, grad_weight_npu

            # print("pc: npu end")
            # print("pc: idx npu out= ", out)

        return out[0], out[1]

