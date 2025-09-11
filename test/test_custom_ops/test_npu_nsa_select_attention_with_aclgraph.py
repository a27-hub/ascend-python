# Copyright (c) 2025, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
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
import tqdm
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestNpuCompress(TestCase):

    def get_tnd_idx(self, actual_q_len, t_idx):
        b_idx = 0
        while t_idx >= actual_q_len[b_idx]:
            b_idx += 1
        if b_idx == 0:
            s1_offset = 0
        else:
            s1_offset = actual_q_len[b_idx - 1]
        s1_idx = t_idx - s1_offset

        return b_idx, s1_idx

    # pylint:disable = huawei-too-many-arguments
    def npu_op_exec(self, query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen):
        
        out = torch_npu.npu_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=None, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
        return out

    def npu_op_exec_gclgraph(self, query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen):
        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        softmax_max_out, softmax_sum_out, attention_out = None, None, None
        handle = None
        workspace = torch_npu._npu_nsa_select_attention_get_max_workspace(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=None, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)

        with torch.npu.graph(g, auto_dispatch_capture=True):
            softmax_max_out = torch.empty((query.shape[0], query.shape[1], 8), dtype=torch.float, device="npu")
            softmax_sum_out = torch.empty((query.shape[0], query.shape[1], 8), dtype=torch.float, device="npu")
            attention_out = torch.empty((query.shape[0], query.shape[1], value.shape[2]), dtype=torch.float16, device="npu")
            torch_npu.npu_nsa_select_attention.out(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, atten_mask=atten_mask, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, out=[attention_out, softmax_max_out, softmax_sum_out])
        g.update(cpu_update_input=[{"actual_seq_kvlen": actual_seq_kvlen}])
        g.replay()
        return [attention_out, softmax_max_out, softmax_sum_out]

    @SupportedDevices(['Ascend910B'])
    def test_npu_nsa_select_attention(self):
        query = torch.randn([1, 115, 192], dtype=torch.float16)
        key = torch.randn([1088, 115, 192], dtype=torch.float16)
        value = torch.randn([1088, 115, 128], dtype=torch.float16)
        scale_value = 1.0
        head_num = query.size(1)
        select_block_count = 16
        select_block_size = 64
        actual_seq_qlen = [1]
        actual_seq_kvlen = [1088]

        topk_indices = torch.zeros(query.size(0), value.size(1), select_block_count,).to(torch.int)
        for i in range(query.size(0)):
            _, s1_idx = self.get_tnd_idx(actual_seq_qlen, i)
            for j in range(value.size(1)):
                if s1_idx < select_block_count * select_block_size:
                    topk_indices[i][j] = torch.arange(select_block_count)
                else:
                    topk_indices[i][j] = torch.randperm(select_block_count)
                    topk_indices[i][j][random.uniform(0, select_block_count)] = (s1_idx + select_block_size - 1) // select_block_size

        npuout = self.npu_op_exec(query.npu(), key.npu(), value.npu(), topk_indices.npu(), scale_value, head_num, select_block_size, select_block_count, None, actual_seq_qlen, actual_seq_kvlen)
        gloden_out = self.npu_op_exec_gclgraph(query.npu(), key.npu(), value.npu(), topk_indices.npu(), scale_value, head_num, select_block_size, select_block_count, None, actual_seq_qlen, actual_seq_kvlen)

        self.assertRtolEqual(gloden_out[0].cpu().numpy(), npuout[0].cpu().numpy())
        self.assertRtolEqual(gloden_out[1].cpu().numpy(), npuout[1].cpu().numpy())
        self.assertRtolEqual(gloden_out[2].cpu().numpy(), npuout[2].cpu().numpy())

if __name__ == "__main__":
    run_tests()
