#!/usr/bin/env python3
from copy import deepcopy
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import torch_npu
import torch.nn as nn
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import logging

def GeluMul(input_tensor, approximate = "none"):
    
    last_dim = input_tensor.shape[-1]
    if last_dim % 2 == 1:
        return "~~~~~~~~~~shape error~~~~~~~~~~"
    d = last_dim // 2
    
    x1 = input_tensor[..., :d]
    x2 = input_tensor[..., d:]

    m = torch.nn.GELU(approximate)
    x1 = m(x1)

    return x1 * x2

@register("function_active_gelu")
class MethodTFgather(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTFgather, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        self.set_device = f"npu:{self.device_id}" if self.device == "npu" else "cpu"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        data = input_data.kwargs["input"]
        approximate = input_data.kwargs["approximateOptional"]

        if self.device == "cpu":
            # CPU golden reference (original CANN logic), unchanged
            return GeluMul(deepcopy(data).cpu(), approximate)
        elif self.device == "npu":
            # PTA path: call torch_npu.npu_gelu_mul
            data_npu = deepcopy(data).to(self.set_device)
            return torch_npu.npu_gelu_mul(data_npu, approximate=approximate)

        return GeluMul(data, approximate)

