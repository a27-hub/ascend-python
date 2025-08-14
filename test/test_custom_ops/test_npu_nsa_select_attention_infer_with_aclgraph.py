import unittest
from itertools import chain
import math
import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclgraphUpdate(TestCase):

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_sai_update(self):
        batch_size = 2
        kv_seqlen = 128
        page_block_size = 128
        num_key_value_heads = 2
        num_heads = 32
        selected_block_size = 64
        actual_seq_kvlen = [kv_seqlen] * batch_size
        max_k_seqlen = max(actual_seq_kvlen)
        layout = "BSND"
        select_block_count = 2
        atten_mask = None
        scale_value = 0.0721
        g = 4
        d1 = 192
        d2 = 128
        actual_seq_qlen = [1] * batch_size
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / page_block_size)
        block_table = torch.tensor([[1] * ((max_k_seqlen + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        query = torch.randn([batch_size, 1, num_heads, d1], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads, d1], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads, d2], dtype=torch.float16, device="npu")
        topk_indices = torch.tensor([0, -1, 0, -1, 1, -1, 1, -1], dtype=torch.int32).reshape((2, 2, 2)).npu()

        res = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None
        handle = None
        workspace = torch_npu._npu_nsa_select_attention_infer_get_max_workspace(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)
        size_in_bytes = workspace.numel() * workspace.element_size()
        print(f"{size_in_bytes / 1024 / 1024:.2f} MB, ==============success====================")  # 转换为 MB

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty((batch_size, 1, num_heads, d2), dtype=torch.float16, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res.cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])    
    def test_sai_update_BSH(self):
        batch_size = 2
        kv_seqlen = 128
        page_block_size = 128
        num_key_value_heads = 2
        num_heads = 32
        selected_block_size = 64
        actual_seq_kvlen = [kv_seqlen] * batch_size
        max_k_seqlen = max(actual_seq_kvlen)
        layout = "BSH"
        select_block_count = 2
        atten_mask = None
        scale_value = 0.0721
        g = 4
        d1 = 192
        d2 = 128
        actual_seq_qlen = [1] * batch_size
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / page_block_size)
        block_table = torch.tensor([[1] * ((max_k_seqlen + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        query = torch.randn([batch_size, 1, num_heads * d1], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d1], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d2], dtype=torch.float16, device="npu")
        topk_indices = torch.tensor([0, -1, 0, -1, 1, -1, 1, -1], dtype=torch.int32).reshape((2, 2, 2)).npu()

        res = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None
        handle = None
        workspace = torch_npu._npu_nsa_select_attention_infer_get_max_workspace(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)
        size_in_bytes = workspace.numel() * workspace.element_size()
        print(f"{size_in_bytes / 1024 / 1024:.2f} MB, ==============success====================")  # 转换为 MB

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty((batch_size, 1, num_heads * d2), dtype=torch.float16, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res.cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_sai_update_TND(self):
        batch_size = 2
        kv_seqlen = 128
        page_block_size = 128
        num_key_value_heads = 2
        num_heads = 32
        selected_block_size = 64
        actual_seq_kvlen = [kv_seqlen] * batch_size
        max_k_seqlen = max(actual_seq_kvlen)
        layout = "TND"
        select_block_count = 2
        atten_mask = None
        scale_value = 0.0721
        g = 4
        d1 = 192
        d2 = 128
        actual_seq_qlen = [1] * batch_size
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / page_block_size)
        block_table = torch.tensor([[1] * ((max_k_seqlen + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        query = torch.randn([batch_size, 1 * num_heads, d1], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d1], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d2], dtype=torch.float16, device="npu")
        topk_indices = torch.tensor([0, -1, 0, -1, 1, -1, 1, -1], dtype=torch.int32).reshape((2, 2, 2)).npu()

        res = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None
        handle = None
        workspace = torch_npu._npu_nsa_select_attention_infer_get_max_workspace(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)
        size_in_bytes = workspace.numel() * workspace.element_size()
        print(f"{size_in_bytes / 1024 / 1024:.2f} MB, ==============success====================")  # 转换为 MB

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty((batch_size, 1 * num_heads, d2), dtype=torch.float16, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res.cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_sai_update_with_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        batch_size = 2
        kv_seqlen = 128
        page_block_size = 128
        num_key_value_heads = 2
        num_heads = 32
        selected_block_size = 64
        actual_seq_kvlen = [kv_seqlen] * batch_size
        max_k_seqlen = max(actual_seq_kvlen)
        layout = "TND"
        select_block_count = 2
        atten_mask = None
        scale_value = 0.0721
        g = 4
        d1 = 192
        d2 = 128
        actual_seq_qlen = [1] * batch_size
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / page_block_size)
        block_table = torch.tensor([[1] * ((max_k_seqlen + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        query = torch.randn([batch_size, 1 * num_heads, d1], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d1], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d2], dtype=torch.float16, device="npu")
        topk_indices = torch.tensor([0, -1, 0, -1, 1, -1, 1, -1], dtype=torch.int32).reshape((2, 2, 2)).npu()

        res = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None
        handle = None
        workspace = torch_npu._npu_nsa_select_attention_infer_get_max_workspace(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)
        size_in_bytes = workspace.numel() * workspace.element_size()
        print(f"{size_in_bytes / 1024 / 1024:.2f} MB, ==============success====================")  # 转换为 MB

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty((batch_size, 1 * num_heads, d2), dtype=torch.float16, device="npu")
            torch_npu.npu_nsa_select_attention_infer.out(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen, workspace=workspace, output=output)
         
        g.update(cpu_update_input=[{"actual_seq_kvlen": actual_seq_kvlen}])
        g.replay()
        self.assertEqual(output.cpu(), res.cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_sai_update_with_non_out_and_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        batch_size = 2
        kv_seqlen = 128
        page_block_size = 128
        num_key_value_heads = 2
        num_heads = 32
        selected_block_size = 64
        actual_seq_kvlen = [kv_seqlen] * batch_size
        max_k_seqlen = max(actual_seq_kvlen)
        layout = "TND"
        select_block_count = 2
        atten_mask = None
        scale_value = 0.0721
        g = 4
        d1 = 192
        d2 = 128
        actual_seq_qlen = [1] * batch_size
        max_num_blocks_per_seq = math.ceil(max_k_seqlen / page_block_size)
        block_table = torch.tensor([[1] * ((max_k_seqlen + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        query = torch.randn([batch_size, 1 * num_heads, d1], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d1], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size * max_num_blocks_per_seq, page_block_size, num_key_value_heads * d2], dtype=torch.float16, device="npu")
        topk_indices = torch.tensor([0, -1, 0, -1, 1, -1, 1, -1], dtype=torch.int32).reshape((2, 2, 2)).npu()

        res = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)

        g = torch.npu.NPUGraph()
        output = None
        topKOutput = None
        handle = None

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty((batch_size, 1 * num_heads, d2), dtype=torch.float16, device="npu")
            output = torch_npu.npu_nsa_select_attention_infer(query, key, value, topk_indices, scale_value, num_heads, 
                                                       num_key_value_heads, selected_block_size, select_block_count, 
                                                       page_block_size, layout=layout, atten_mask=atten_mask, 
                                                       block_table=block_table, actual_seq_qlen=actual_seq_qlen, 
                                                       actual_seq_kvlen=actual_seq_kvlen)
        
        g.update(cpu_update_input=[{"actual_seq_kvlen": actual_seq_kvlen}])
        g.replay()
        self.assertEqual(output.cpu(), res.cpu())


if __name__ == "__main__":
    run_tests()