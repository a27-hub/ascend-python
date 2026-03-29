import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestGroupedMatmul(TestCase):

    def single_matmul(self, y, x1, x2, x1_scale, x2_scale):
        output = np.matmul(x1, x2)
        output = output * x1_scale * x2_scale
        output = output + y
        return output

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_950(self):
        torch.manual_seed(0)
        group_list = torch.tensor([16, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(256, 16), dtype=torch.int8).npu()
        x2 = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(1, 16, 32), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()
        x2_scale = np.random.uniform(low=-5, high=5, size=(1, 32)).astype(np.float32)
        x1_scale = np.random.uniform(low=-5, high=5, size=(1,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()
        x1_dtype = torch_npu.float8_e8m0fnu
        x2_dtype = torch_npu.float8_e8m0fnu
        supported_output = self.single_matmul(y, x1.t().cpu().to(torch.float32), x2.cpu().to(torch.float32), x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                      group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                      x2_dtype=x2_dtype)

        custom_output2 = torch_npu.npu_add_quant_gmm(y2_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                     group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                     x2_dtype=x2_dtype)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_hifloat8_x1(self):
        torch.manual_seed(0)
        group_list = torch.tensor([16, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(256, 16), dtype=torch.int8).npu()
        x2 = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.int8).npu()

        y = torch.randint(2, 3, size=(1, 16, 32), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()

        x2_scale = np.random.uniform(low=-5, high=5, size=(1, 32)).astype(np.float32)
        x1_scale = np.random.uniform(low=-5, high=5, size=(1,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        supported_output = self.single_matmul(y, x1.t().cpu().to(torch.float32), x2.cpu().to(torch.float32), x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                      group_list_type=0, group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                      x2_dtype=torch_npu.hifloat8)
        custom_output2 = torch_npu.npu_add_quant_gmm(y2_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                     group_list_type=0, group_sizes=None, x1_dtype=torch_npu.hifloat8,
                                                     x2_dtype=torch_npu.hifloat8)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e5m2_x1(self):
        torch.manual_seed(0)
        group_list = torch.tensor([16, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(256, 16), dtype=torch.int8).view(torch.float8_e5m2).npu()
        x2 = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.int8).view(torch.float8_e5m2).npu()

        y = torch.randint(2, 3, size=(1, 16, 32), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()

        x2_scale = np.random.uniform(low=-5, high=5, size=(1, 32)).astype(np.float32)
        x1_scale = np.random.uniform(low=-5, high=5, size=(1,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        x1_dtype = torch.float8_e5m2
        x2_dtype = torch.float8_e5m2

        supported_output = self.single_matmul(y, x1.t().cpu().view(torch.float32), x2.cpu().view(torch.float32), x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                      group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                      x2_dtype=x2_dtype)
        custom_output2 = torch_npu.npu_add_quant_gmm(y2_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                     group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                     x2_dtype=x2_dtype)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)

    @SupportedDevices(['Ascend950P'])
    def test_npu_grouped_matmul_quant_float8_e4m3fn_x1(self):
        torch.manual_seed(0)
        group_list = torch.tensor([16, ]).to(torch.int64).npu()

        x1 = torch.randint(2, 3, size=(256, 16), dtype=torch.int8).view(torch.float8_e4m3fn).npu()
        x2 = torch.randint(2, 3, size=(1, 256, 32), dtype=torch.int8).view(torch.float8_e4m3fn).npu()

        y = torch.randint(2, 3, size=(1, 16, 32), dtype=torch.float32)
        y1_clone = y.clone().npu()
        y2_clone = y.clone().npu()

        x2_scale = np.random.uniform(low=-5, high=5, size=(1, 32)).astype(np.float32)
        x1_scale = np.random.uniform(low=-5, high=5, size=(1,)).astype(np.float32)
        x2_scale_clone = torch.from_numpy(x2_scale).npu()
        x1_scale_clone = torch.from_numpy(x1_scale).npu()

        x1_dtype = torch.float8_e4m3fn
        x2_dtype = torch.float8_e4m3fn

        supported_output = self.single_matmul(y, x1.t().cpu().view(torch.float32), x2.cpu().view(torch.float32), x1_scale, x2_scale)
        custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                      group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                      x2_dtype=x2_dtype)
        custom_output2 = torch_npu.npu_add_quant_gmm(y2_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
                                                     group_list_type=0, group_sizes=None, x1_dtype=x1_dtype,
                                                     x2_dtype=x2_dtype)
        self.assertRtolEqual(supported_output, custom_output1.cpu().numpy(), 0.001)
        self.assertRtolEqual(supported_output, custom_output2.cpu().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()


ERROR: test_npu_grouped_matmul_quant_950 (__main__.TestGroupedMatmul)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 39, in test_npu_grouped_matmul_quant_950
    custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
RuntimeError: npu_add_quant_gmm_:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:61 NPU function error: call aclnnQuantGroupedMatmulInplaceAdd failed, error code is 161002
[ERROR] 2047-12-19-01:18:31 (PID:3700729, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 3700729] 2047-12-19-01:18:31.564.897 AclNN_Parameter_Error(EZ1001): Format of x2 should be ND, current format is invalid.


======================================================================
ERROR: test_npu_grouped_matmul_quant_float8_e4m3fn_x1 (__main__.TestGroupedMatmul)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 126, in test_npu_grouped_matmul_quant_float8_e4m3fn_x1
    supported_output = self.single_matmul(y, x1.t().cpu().view(torch.float32), x2.cpu().view(torch.float32), x1_scale, x2_scale)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 16, in single_matmul
    output = np.matmul(x1, x2)
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 256 is different from 64)

======================================================================
ERROR: test_npu_grouped_matmul_quant_float8_e5m2_x1 (__main__.TestGroupedMatmul)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 96, in test_npu_grouped_matmul_quant_float8_e5m2_x1
    supported_output = self.single_matmul(y, x1.t().cpu().view(torch.float32), x2.cpu().view(torch.float32), x1_scale, x2_scale)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 16, in single_matmul
    output = np.matmul(x1, x2)
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 256 is different from 64)

======================================================================
ERROR: test_npu_grouped_matmul_quant_hifloat8_x1 (__main__.TestGroupedMatmul)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch_npu/testing/common_utils.py", line 206, in dep_fn
    return fn(slf, *args, **kwargs)
  File "/home/c00879293/op-plugin/test/test_custom_ops/test_quant_grouped_matmul_inplace_add.py", line 67, in test_npu_grouped_matmul_quant_hifloat8_x1
    custom_output1 = torch_npu.npu_add_quant_gmm_(y1_clone, x1.t(), x2, x2_scale_clone, group_list, x1_scale=x1_scale_clone,
  File "/root/miniconda3/envs/testing/lib/python3.10/site-packages/torch/_ops.py", line 1158, in __call__
    return self._op(*args, **(kwargs or {}))
RuntimeError: npu_add_quant_gmm_:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:61 NPU function error: call aclnnQuantGroupedMatmulInplaceAdd failed, error code is 161002
[ERROR] 2047-12-19-01:18:31 (PID:3700729, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
[PID: 3700729] 2047-12-19-01:18:31.570.665 AclNN_Parameter_Error(EZ1001): Format of x2 should be ND, current format is invalid.
