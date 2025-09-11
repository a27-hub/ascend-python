import unittest
from itertools import chain

import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


class TestAclgraphUpdate(TestCase):

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_cai_update(self):
        torch.npu.set_device(0)
        batch_size = 10
        Seq = 1
        head_num = 4
        key_value_head_num = 2
        head_dim = 8
        seq_length = 512
        hidden_dim = head_num * head_dim 
        page_block_size = 32
        block_table = torch.tensor([[1] * ((seq_length + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        head_dim_vo = 6
        query = torch.randn([batch_size * Seq, head_num, head_dim], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim_vo], dtype=torch.float16, device="npu")
        scale_value = 0.01
        select_block_size = 64
        select_block_count = 8
        compress_block_size = 32
        compress_stride = 16
        actual_cmp_seq_kvlen = [95] * batch_size
        atten_mask = None
        topk_mask = None
        actual_seq_qlen = None
        actual_sel_seq_kvlen = None
        res = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None 
        topKOutput = None
        handle = None
        workspace = torch_npu._npu_nsa_compress_attention_infer_get_max_workspace(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty((batch_size, head_num, head_dim_vo), dtype=torch.float16, device="npu")
            topKOutput = torch.empty((batch_size, key_value_head_num, select_block_count), dtype=torch.int32, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_nsa_compress_attention_infer.out(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, atten_mask=None, block_table=block_table,
                                                            topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, 
                                                            actual_sel_seq_kvlen=None, workspace=workspace, out=[output, topKOutput])
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_nsa_compress_attention_infer.out(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, atten_mask=None, block_table=block_table,
                                                            topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, 
                                                            actual_sel_seq_kvlen=None, workspace=workspace, out=[output, topKOutput])
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res[0].cpu())
        self.assertEqual(topKOutput.cpu(), res[1].cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_cai_update_BSND(self):
        layout = "BSND"
        torch.npu.set_device(0)
        batch_size = 10
        Seq = 1
        head_num = 4
        key_value_head_num = 2
        head_dim = 8
        seq_length = 512
        hidden_dim = head_num * head_dim 
        page_block_size = 32
        block_table = torch.tensor([[1] * ((seq_length + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        head_dim_vo = 6
        query = torch.randn([batch_size, Seq, head_num, head_dim], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim_vo], dtype=torch.float16, device="npu")
        scale_value = 0.01
        select_block_size = 64
        select_block_count = 8
        compress_block_size = 32
        compress_stride = 16
        actual_cmp_seq_kvlen = [95] * batch_size
        atten_mask = None
        topk_mask = None
        actual_seq_qlen = None
        actual_sel_seq_kvlen = None

        res = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, layout=layout, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None 
        topKOutput = None
        handle = None
        workspace = torch_npu._npu_nsa_compress_attention_infer_get_max_workspace(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, layout=layout, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        with torch.npu.graph(g):
            stream = torch.npu.current_stream()
            output = torch.empty((batch_size, Seq, head_num, head_dim_vo), dtype=torch.float16, device="npu")
            topKOutput = torch.empty((batch_size, Seq, key_value_head_num, select_block_count), dtype=torch.int32, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_nsa_compress_attention_infer.out(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, layout=layout, atten_mask=None, block_table=block_table,
                                                            topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, 
                                                            actual_sel_seq_kvlen=None, workspace=workspace, out=[output, topKOutput])
            handle = torch.npu.graph_task_group_end(stream)
        
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_nsa_compress_attention_infer.out(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, layout=layout, atten_mask=None, block_table=block_table,
                                                            topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, 
                                                            actual_sel_seq_kvlen=None, workspace=workspace, out=[output, topKOutput])
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)

        g.replay()
        self.assertEqual(output.cpu(), res[0].cpu())
        self.assertEqual(topKOutput.cpu(), res[1].cpu())

    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_cai_update_with_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        batch_size = 10
        Seq = 1
        head_num = 4
        key_value_head_num = 2
        head_dim = 8
        seq_length = 512
        hidden_dim = head_num * head_dim 
        page_block_size = 32
        block_table = torch.tensor([[1] * ((seq_length + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        head_dim_vo = 6
        query = torch.randn([batch_size * Seq, head_num, head_dim], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim_vo], dtype=torch.float16, device="npu")
        scale_value = 0.01
        select_block_size = 64
        select_block_count = 8
        compress_block_size = 32
        compress_stride = 16
        actual_cmp_seq_kvlen = [95] * batch_size
        atten_mask = None
        topk_mask = None
        actual_seq_qlen = None
        actual_sel_seq_kvlen = None

        res_src = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        g = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None 
        topKOutput = None
        handle = None

        workspace = torch_npu._npu_nsa_compress_attention_infer_get_max_workspace(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty((batch_size, head_num, head_dim_vo), dtype=torch.float16, device="npu")
            topKOutput = torch.empty((batch_size, key_value_head_num, select_block_count), dtype=torch.int32, device="npu")
            torch_npu.npu_nsa_compress_attention_infer.out(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, atten_mask=None, block_table=block_table,
                                                            topk_mask=None, actual_seq_qlen=None, actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, 
                                                            actual_sel_seq_kvlen=None, workspace=workspace, out=[output, topKOutput])
         
        g.update(cpu_update_input=[{"actual_cmp_seq_kvlen": actual_cmp_seq_kvlen}])
        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(topKOutput.cpu(), res_src[1].cpu())
    
    @unittest.skip("Skipping due to outdated CANN version; please update CANN to the latest version and remove this skip")
    @SupportedDevices(['Ascend910B'])
    def test_cai_update_with_non_out_and_auto_dispatch_capture(self):
        torch.npu.set_device(0)
        batch_size = 10
        Seq = 1
        head_num = 4
        key_value_head_num = 2
        head_dim = 8
        seq_length = 512
        hidden_dim = head_num * head_dim 
        page_block_size = 32
        block_table = torch.tensor([[1] * ((seq_length + page_block_size - 1) // page_block_size)] * batch_size, device="npu").int()
        head_dim_vo = 6
        query = torch.randn([batch_size * Seq, head_num, head_dim], dtype=torch.float16, device="npu")
        key = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim], dtype=torch.float16, device="npu")
        value = torch.randn([batch_size, page_block_size, key_value_head_num * head_dim_vo], dtype=torch.float16, device="npu")
        scale_value = 0.01
        select_block_size = 64
        select_block_count = 8
        compress_block_size = 32
        compress_stride = 16
        actual_cmp_seq_kvlen = [95] * batch_size
        atten_mask = None
        topk_mask = None
        actual_seq_qlen = None
        actual_sel_seq_kvlen = None

        res_src = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)


        g = torch.npu.NPUGraph()
        output = None 
        topKOutput = None
        handle = None

        with torch.npu.graph(g, auto_dispatch_capture=True):
            output = torch.empty((batch_size, head_num, head_dim_vo), dtype=torch.float16, device="npu")
            topKOutput = torch.empty((batch_size, key_value_head_num, select_block_count), dtype=torch.int32, device="npu")
            output, topKOutput = torch_npu.npu_nsa_compress_attention_infer(query, key, value, scale_value, head_num, key_value_head_num, 
                                                            select_block_size, select_block_count, page_block_size, 
                                                            compress_block_size, compress_stride, block_table=block_table, 
                                                            atten_mask=None, topk_mask=None, actual_seq_qlen=None, 
                                                            actual_cmp_seq_kvlen=actual_cmp_seq_kvlen, actual_sel_seq_kvlen=None)
        
        g.update(cpu_update_input=[{"actual_cmp_seq_kvlen": actual_cmp_seq_kvlen}])
        g.replay()
        self.assertEqual(output.cpu(), res_src[0].cpu())
        self.assertEqual(topKOutput.cpu(), res_src[1].cpu())
    


if __name__ == "__main__":
    run_tests()