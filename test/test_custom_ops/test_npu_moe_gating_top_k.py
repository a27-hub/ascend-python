# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

import itertools
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuMoeGatingTopK(TestCase):

    def cpu_op_exec(self, x: np.ndarray, bias: np.ndarray, k: int, k_group: int = 1, group_count: int = 1,
                    group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, y2_flag: bool = False,
                    routed_scaling_factor: float = 1.0, eps: float = 1e-20) -> tuple:
        if norm_type == 0:
            x = np.exp(
                x - np.expand_dims(np.log(np.sum(np.exp(x), axis=-1, keepdims=True)), axis=-1))  # softmax
        else:
            x = 1 / (1 + np.exp(-x))  # sigmoid
        original_x = x
        if bias is not None:
            x = x + bias

        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = np.amax(x, axis=-1)
            else:
                group_x = np.partition(
                    x, -2, axis=-1)[..., -2:].sum(axis=-1)
            # Indices of top-k_group
            indices = np.argsort(
                group_x, axis=-1, kind='stable')[:, -k_group:]
            mask = np.zeros_like(x[..., 0], dtype=bool)
            np.put_along_axis(mask, indices, True, axis=-1)
            x = (x * mask[..., np.newaxis]).reshape(x.shape[0], -1)
        # Sorting in descending order and getting top-k indices
        indices = np.argsort(-x, axis=-1, kind='stable')
        indices = indices[:, :k]
        y = np.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1:
            y /= (np.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if y2_flag:
            y2 = original_x
        else:
            y2 = None
        return y, indices.astype(np.int32), y2

    def npu_op_exec(self, x, bias, k: int, k_group: int = 1, group_count: int = 1,
                    group_select_mode: int = 0, renorm: int = 0, norm_type: int = 0, y2_flag: bool = False,
                    routed_scaling_factor: float = 1.0, eps: float = 1e-20):
        y, expert_id, out = torch_npu.npu_moe_gating_top_k(x, bias, k=k, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode,
                                                           renorm=renorm, norm_type=norm_type, y2_flag=y2_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
        return y, expert_id, out

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_gating_top_k(self, device="npu"):
        n = 16
        h = 256
        k = 8
        k_group = 4
        group_count = 8
        group_select_mode = 1
        renorm = 0
        norm_type = 1
        y2_flag = False
        routed_scaling_factor = 1.0
        eps = 1e-20

        x = np.random.uniform(-1, 1, size=(n, h)).astype(np.float32)
        bias = np.random.uniform(-1, 1, size=(h,)).astype(np.float32)
        x_npu = torch.from_numpy(x).npu()
        bias_npu = torch.from_numpy(bias).npu()

        y, expert_idx, out = self.cpu_op_exec(x, bias, k=k, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode,
                                              renorm=renorm, norm_type=norm_type, y2_flag=y2_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
        y_npu, expert_idx_npu, out_npu = self.npu_op_exec(x_npu, bias_npu, k=k, k_group=k_group, group_count=group_count, group_select_mode=group_select_mode,
                                                          renorm=renorm, norm_type=norm_type, y2_flag=y2_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)
        self.assertRtolEqual(y, y_npu.cpu().numpy())
        self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())
        if out:
            self.assertRtolEqual(out, out_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
