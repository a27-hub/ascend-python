import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


"""
测试用例说明：
本测试文件包含20个测试用例，覆盖以下场景：

基础场景（非量化）：
- 用例1-4: float16数据类型，不同world_size和shape组合
- 用例5-8: bfloat16数据类型，不同world_size和shape组合

量化场景：
- 用例9-10: float16输出，int8输入+scale
- 用例11-12: bfloat16输出，int8输入+scale

扩展场景（float8/hifloat8）：
- 用例13-14: float8_e4m3fn数据类型
- 用例15-16: float8_e5m2数据类型
- 用例17-18: hifloat8数据类型
- 用例19-20: 混合精度场景

数据类型支持：
- float16/bfloat16: Ascend910B, Ascend950P
- float8_e4m3fn/float8_e5m2/hifloat8: Ascend950P
"""


class TestQuantMmReduceScatter(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_npu_quant_mm_reduce_scatter(cls, rank, input_list):
        x1, x2, world_size, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, _ = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                        x2,
                                                        hcom_name,
                                                        world_size,
                                                        reduce_op='sum',
                                                        bias=None,
                                                        comm_turn=0)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, world_size = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], world_size, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank], 0.05, 0.05)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, x1_list, x2_list, world_size):
        out = None
        for i in range(world_size):
            x1 = x1_list[i]
            x2 = x2_list[i]
            out_single = torch.matmul(x1.to(torch.float), x2.to(torch.float))
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(x1_list[0].dtype)
        index = x1_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp16_basic(self):
        """
        用例1: 基础场景 - float16, world_size=8
        测试float16数据类型的基础matmul+reduce_scatter融合场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp16_ws4(self):
        """
        用例2: 基础场景 - float16, world_size=4
        测试float16数据类型，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp16_ws2(self):
        """
        用例3: 基础场景 - float16, world_size=2
        测试float16数据类型，最小world_size=2的场景
        """
        world_size = 2
        m, k, n = 32, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp16_large_shape(self):
        """
        用例4: 基础场景 - float16, 大shape
        测试float16数据类型，较大shape的场景
        """
        world_size = 8
        m, k, n = 256, 1024, 512
        x1_list = [torch.randn(m, k, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.float16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_bf16_basic(self):
        """
        用例5: 基础场景 - bfloat16, world_size=8
        测试bfloat16数据类型的基础matmul+reduce_scatter融合场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_bf16_ws4(self):
        """
        用例6: 基础场景 - bfloat16, world_size=4
        测试bfloat16数据类型，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(2)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_bf16_ws2(self):
        """
        用例7: 基础场景 - bfloat16, world_size=2
        测试bfloat16数据类型，最小world_size=2的场景
        """
        world_size = 2
        m, k, n = 32, 512, 256
        x1_list = [torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_bf16_large_shape(self):
        """
        用例8: 基础场景 - bfloat16, 大shape
        测试bfloat16数据类型，较大shape的场景
        """
        world_size = 8
        m, k, n = 256, 1024, 512
        x1_list = [torch.randn(m, k, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        x2_list = [torch.randn(k, n, dtype=torch.bfloat16).uniform_(-1, 1) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result(x1_list, x2_list, world_size)
        self._test_multiprocess(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter,
                                TestQuantMmReduceScatter._init_dist_hccl, [expt_out_list, x1_list, x2_list, world_size])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_quant(cls, rank, input_list):
        x1, x2, x1_scale, x2_scale, world_size, output_dtype, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        x1_scale = x1_scale.npu()
        x2_scale = x2_scale.npu()
        out, _ = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                        x2,
                                                        hcom_name,
                                                        world_size,
                                                        reduce_op='sum',
                                                        bias=None,
                                                        x1_scale=x1_scale,
                                                        x2_scale=x2_scale,
                                                        comm_turn=0,
                                                        y_dtype=output_dtype)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_quant(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], x1_scale_list[i], x2_scale_list[i], world_size, output_dtype, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank], 0.05, 0.05)

        for p in ps:
            p.join()

    def _construct_excepted_result_quant(self, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype):
        out = None
        for i in range(world_size):
            x1 = x1_list[i].float() * x1_scale_list[i]
            x2 = x2_list[i].float() * x2_scale_list[i]
            out_single = torch.matmul(x1, x2)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(output_dtype)
        index = x1_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_int8_fp16_output(self):
        """
        用例9: 量化场景 - int8输入, float16输出
        测试int8量化输入，float16输出的场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.float16
        x1_list = [torch.randint(-10, 10, size=(m, k), dtype=torch.int8) for _ in range(world_size)]
        x2_list = [torch.randint(-10, 10, size=(k, n), dtype=torch.int8) for _ in range(world_size)]
        x1_scale_list = [torch.randn((m, 1), dtype=torch.float32) for _ in range(world_size)]
        x2_scale_list = [torch.randn((1, n), dtype=torch.float32) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype)
        self._test_multiprocess_quant(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_quant,
                                      TestQuantMmReduceScatter._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_int8_fp16_output_ws4(self):
        """
        用例10: 量化场景 - int8输入, float16输出, world_size=4
        测试int8量化输入，float16输出，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        output_dtype = torch.float16
        x1_list = [torch.randint(-10, 10, size=(m, k), dtype=torch.int8) for _ in range(world_size)]
        x2_list = [torch.randint(-10, 10, size=(k, n), dtype=torch.int8) for _ in range(world_size)]
        x1_scale_list = [torch.randn((m, 1), dtype=torch.float32) for _ in range(world_size)]
        x2_scale_list = [torch.randn((1, n), dtype=torch.float32) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype)
        self._test_multiprocess_quant(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_quant,
                                      TestQuantMmReduceScatter._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_int8_bf16_output(self):
        """
        用例11: 量化场景 - int8输入, bfloat16输出
        测试int8量化输入，bfloat16输出的场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch.randint(-10, 10, size=(m, k), dtype=torch.int8) for _ in range(world_size)]
        x2_list = [torch.randint(-10, 10, size=(k, n), dtype=torch.int8) for _ in range(world_size)]
        x1_scale_list = [torch.randn((m, 1), dtype=torch.float32) for _ in range(world_size)]
        x2_scale_list = [torch.randn((1, n), dtype=torch.float32) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype)
        self._test_multiprocess_quant(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_quant,
                                      TestQuantMmReduceScatter._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend910B', 'Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_int8_bf16_output_ws4(self):
        """
        用例12: 量化场景 - int8输入, bfloat16输出, world_size=4
        测试int8量化输入，bfloat16输出，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch.randint(-10, 10, size=(m, k), dtype=torch.int8) for _ in range(world_size)]
        x2_list = [torch.randint(-10, 10, size=(k, n), dtype=torch.int8) for _ in range(world_size)]
        x1_scale_list = [torch.randn((m, 1), dtype=torch.float32) for _ in range(world_size)]
        x2_scale_list = [torch.randn((1, n), dtype=torch.float32) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_quant(x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype)
        self._test_multiprocess_quant(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_quant,
                                      TestQuantMmReduceScatter._init_dist_hccl,
                                      [expt_out_list, x1_list, x2_list, x1_scale_list, x2_scale_list, world_size, output_dtype])

    @classmethod
    def _test_npu_quant_mm_reduce_scatter_fp8(cls, rank, input_list):
        x1, x2, world_size, output_dtype, init_pg, c2p = input_list
        pg = init_pg(rank, world_size)
        group = pg.distributed_c10d._get_default_group()
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)

        x1 = x1.npu()
        x2 = x2.npu()
        out, _ = torch_npu.npu_quant_mm_reduce_scatter(x1,
                                                        x2,
                                                        hcom_name,
                                                        world_size,
                                                        reduce_op='sum',
                                                        bias=None,
                                                        comm_turn=0,
                                                        y_dtype=output_dtype)

        c2p.put((rank, out.cpu()))
        pg.barrier()

    def _test_multiprocess_fp8(self, f, init_pg, input_list):
        expt_out_list, x1_list, x2_list, world_size, output_dtype = input_list
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, [x1_list[i], x2_list[i], world_size, output_dtype, init_pg, c2p]))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output = c2p.get()
            self.assertRtolEqual(output, expt_out_list[rank], 0.05, 0.05)

        for p in ps:
            p.join()

    def _construct_excepted_result_fp8(self, x1_list, x2_list, world_size, output_dtype):
        out = None
        for i in range(world_size):
            x1 = x1_list[i].to(torch.float32)
            x2 = x2_list[i].to(torch.float32)
            out_single = torch.matmul(x1, x2)
            if out is None:
                out = out_single
            else:
                out = torch.add(out, out_single)
        out = out.to(output_dtype)
        index = x1_list[0].shape[0] // world_size
        return [out[index * i:index * (i + 1), :] for i in range(world_size)]

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e4m3fn_basic(self):
        """
        用例13: 扩展场景 - float8_e4m3fn, world_size=8
        测试float8_e4m3fn数据类型的matmul+reduce_scatter融合场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e4m3fn_ws4(self):
        """
        用例14: 扩展场景 - float8_e4m3fn, world_size=4
        测试float8_e4m3fn数据类型，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e5m2_basic(self):
        """
        用例15: 扩展场景 - float8_e5m2, world_size=8
        测试float8_e5m2数据类型的matmul+reduce_scatter融合场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e5m2) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e5m2_ws4(self):
        """
        用例16: 扩展场景 - float8_e5m2, world_size=4
        测试float8_e5m2数据类型，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e5m2) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_hifloat8_basic(self):
        """
        用例17: 扩展场景 - hifloat8, world_size=8
        测试hifloat8数据类型的matmul+reduce_scatter融合场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_int8 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2_int8 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_list = [torch_npu.HiFloat8Tensor.to_hifloat8(x1_int8.clone()) for _ in range(world_size)]
        x2_list = [torch_npu.HiFloat8Tensor.to_hifloat8(x2_int8.clone()) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(4)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_hifloat8_ws4(self):
        """
        用例18: 扩展场景 - hifloat8, world_size=4
        测试hifloat8数据类型，world_size=4的场景
        """
        world_size = 4
        m, k, n = 64, 512, 256
        output_dtype = torch.bfloat16
        x1_int8 = torch.randint(-5, 5, (m, k), dtype=torch.int8)
        x2_int8 = torch.randint(-5, 5, (k, n), dtype=torch.int8)
        x1_list = [torch_npu.HiFloat8Tensor.to_hifloat8(x1_int8.clone()) for _ in range(world_size)]
        x2_list = [torch_npu.HiFloat8Tensor.to_hifloat8(x2_int8.clone()) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e4m3fn_e5m2_mixed(self):
        """
        用例19: 混合精度场景 - x1=float8_e4m3fn, x2=float8_e5m2
        测试float8_e4m3fn和float8_e5m2混合精度的场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e4m3fn) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e5m2) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])

    @skipIfUnsupportMultiNPU(8)
    @SupportedDevices(['Ascend950P'])
    def test_npu_quant_mm_reduce_scatter_fp8_e5m2_e4m3fn_mixed(self):
        """
        用例20: 混合精度场景 - x1=float8_e5m2, x2=float8_e4m3fn
        测试float8_e5m2和float8_e4m3fn混合精度的场景
        """
        world_size = 8
        m, k, n = 128, 512, 256
        output_dtype = torch.bfloat16
        x1_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (m, k)), torch.float8_e5m2) for _ in range(world_size)]
        x2_list = [torch_npu.npu_dtype_cast(torch.randint(-5, 5, (k, n)), torch.float8_e4m3fn) for _ in range(world_size)]
        expt_out_list = self._construct_excepted_result_fp8(x1_list, x2_list, world_size, output_dtype)
        self._test_multiprocess_fp8(TestQuantMmReduceScatter._test_npu_quant_mm_reduce_scatter_fp8,
                                    TestQuantMmReduceScatter._init_dist_hccl,
                                    [expt_out_list, x1_list, x2_list, world_size, output_dtype])


if __name__ == '__main__':
    run_tests()