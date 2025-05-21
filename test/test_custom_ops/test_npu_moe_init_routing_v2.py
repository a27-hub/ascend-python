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


class TestNpuMoeInitRoutingV2(TestCase):
    def adapter_capacity(self, sorted_row_idx, sorted_expert_idx, capacity):
        count = 0
        last = sorted_expert_idx[0]
        for i, val in enumerate(sorted_expert_idx):
            if last != val:
                count = 1
                last = val
            else:
                count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

    def cpu_op_exec(self, input_x, expert_idx, active_num=0, capacity=0, expert_num=0, drop_pad_mode=0, expert_tokens_count_or_cumsum_flag=0, expert_tokens_before_capacity_flag=False):
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]
        sorted_row_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
        sorted_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)
        if drop_pad_mode == 1 and expert_num <= 0:
            print("expert num can not be 0")
            return

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        # expert_token_idx
        expert_idx_hist, bins = np.histogram(
            sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1))
        expert_token_idx = np.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype("int32")
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")

        if drop_pad_mode == 0:
            expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
            expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32)

            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
        else:
            self.adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity)

            sort_row_tmp = np.full((expert_num * capacity), -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * capacity + offset] = sorted_row_idx[i]
                    offset = offset + 1

            # expand_row_idx
            expanded_row_idx = np.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i

            # expanded_x
            expanded_x = np.full((expert_num * capacity, hidden_size), 0, dtype=input_x.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
            expanded_x = expanded_x.reshape(expert_num, capacity, hidden_size)

        return expanded_x, expanded_row_idx.astype("int32"), expert_tokens_count_or_cumsum, expert_tokens_before_capacity

    def npu_op_exec(self, input_x, expert_idx, active_num, capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
        expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = torch_npu.npu_moe_init_routingV2(input_x,
                                                                                                                                      expert_idx,
                                                                                                                                      active_num,
                                                                                                                                      capacity,
                                                                                                                                      expert_num,
                                                                                                                                      drop_pad_mode,
                                                                                                                                      expert_tokens_count_or_cumsum_flag,
                                                                                                                                      expert_tokens_before_capacity_flag)

        return expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_dropless_expandedx_expandedrowidx(self, device="npu"):
        active_num = 0
        capacity = 2
        expert_num = 3
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 0
        expert_tokens_before_capacity_flag = True

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_dropless_expandedx_expandedrowidx1(self, device="npu"):
        active_num = 0
        capacity = 216
        expert_num = 17
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 0
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_dropless_expandedx_expandedrowidx2(self, device="npu"):
        active_num = 0
        capacity = 216
        expert_num = 17
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 0
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_dropless_expandedx_expandedrowidx_experttokenscountcumsum(self, device="npu"):
        active_num = 0
        capacity = 0
        expert_num = 256
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 1
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expert_tokens_count_or_cumsum, expert_tokens_count_or_cumsum_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_droppad_expandedx_expandedrowidx_experttokensbeforecapacity(self, device="npu"):
        active_num = 0
        capacity = 17
        expert_num = 96
        drop_pad_mode = 1
        expert_tokens_count_or_cumsum_flag = 0
        expert_tokens_before_capacity_flag = True

        num_list = [430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expert_tokens_before_capacity, expert_tokens_before_capacity_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_dropless_expandedx_expandedrowidx_experttokenscount(self, device="npu"):
        active_num = 0
        capacity = 1
        expert_num = 58
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 2
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expert_tokens_count_or_cumsum, expert_tokens_count_or_cumsum_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_active_expandedx_expandedrowidx(self, device="npu"):
        active_num = 15
        capacity = 125
        expert_num = 36
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 0
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_active_expandedx_expandedrowidx_experttokenscountcumsum(self, device="npu"):
        active_num = 690
        capacity = 763
        expert_num = 517
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 1
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expert_tokens_count_or_cumsum, expert_tokens_count_or_cumsum_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_active_expandedx_expandedrowidx_experttokenscount(self, device="npu"):
        active_num = 129
        capacity = 152
        expert_num = 949
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 2
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = np.random.uniform(-1, 1, size=(num_rows, h)).astype(dtype)
            expert_idx = np.random.randint(0, expert_num-1, size=(num_rows, k)).astype(np.int32)

            input_x_npu = torch.from_numpy(input_x).npu()
            expert_idx_npu = torch.from_numpy(expert_idx).npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            self.assertRtolEqual(expanded_x, expanded_x_npu.cpu().numpy())
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu.cpu().numpy())
            self.assertRtolEqual(expert_tokens_count_or_cumsum, expert_tokens_count_or_cumsum_npu.cpu().numpy())

    @SupportedDevices(['Ascend910B'])
    def test_npu_moe_init_routing_v2_active_expandedx_expandedrowidx_experttokenscount_bf16(self, device="npu"):
        active_num = 129
        capacity = 152
        expert_num = 949
        drop_pad_mode = 0
        expert_tokens_count_or_cumsum_flag = 2
        expert_tokens_before_capacity_flag = False

        num_list = [10, 430, 520]
        k_list = [2, 4, 5, 9]
        h_list = [200, 1256, 5120]
        dtype_list = [np.float16, np.float32]
        for num_rows, k, h, dtype in itertools.product(num_list, k_list, h_list, dtype_list):
            input_x = torch.rand(num_rows, h, dtype=torch.bfloat16, device="cpu")
            expert_idx = torch.randint(0, expert_num, size=(num_rows, k), dtype=torch.int32, device="cpu")

            input_x_npu = input_x.npu()
            expert_idx_npu = expert_idx.npu()

            expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = self.cpu_op_exec(input_x,
                                                                                                                          expert_idx,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)
            expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_or_cumsum_npu, expert_tokens_before_capacity_npu = self.npu_op_exec(input_x_npu,
                                                                                                                          expert_idx_npu,
                                                                                                                          active_num,
                                                                                                                          capacity,
                                                                                                                          expert_num,
                                                                                                                          drop_pad_mode,
                                                                                                                          expert_tokens_count_or_cumsum_flag,
                                                                                                                          expert_tokens_before_capacity_flag)

            expanded_x = expanded_x.to(expanded_x_npu.dtype)
            expanded_row_idx = torch.from_numpy(expanded_row_idx).to(expanded_row_idx_npu.dtype)
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum).to(expert_tokens_count_or_cumsum_npu.dtype)

            self.assertRtolEqual(expanded_x, expanded_x_npu)
            self.assertRtolEqual(expanded_row_idx, expanded_row_idx_npu)
            self.assertRtolEqual(expert_tokens_count_or_cumsum, expert_tokens_count_or_cumsum_npu)

if __name__ == "__main__":
    run_tests()
