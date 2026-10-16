import numpy as np
import torch

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuStrideCopy(TestCase):
    def cpu_ref(self, input1, size, stride, storage_offset):
        # 主参考值：CPU 上等价变换（as_strided 视图 -> 连续化）
        return torch.as_strided(input1, size, stride, storage_offset).contiguous().numpy()

    def npu_exec(self, input1, size, stride, storage_offset):
        return torch_npu.npu_stride_copy(input1, size, stride, storage_offset).cpu().numpy()

    def _check(self, item):
        # item: [[dtype, format, shape], size, stride, storage_offset]
        # format 固定为 0(ND)；若为非 base format，在 A2 上会静默回退 acl_op，测不到 op_api 新路径
        cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
        expected = self.cpu_ref(cpu_input, item[1], item[2], item[3])
        for jit in (False, True):  # False -> op_api 新路径；True -> acl_op 回退路径
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, item[1], item[2], item[3])
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def _check_out(self, item):
        # .out 版：预分配输出张量 -> npu_stride_copy.out 原地填充 -> 与参考值比对
        cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
        expected = self.cpu_ref(cpu_input, item[1], item[2], item[3])
        for jit in (False, True):
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            out = torch.empty(item[1], dtype=npu_input.dtype, device=npu_input.device)
            torch.ops.npu.npu_stride_copy.out(npu_input, item[1], item[2], item[3], out=out)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, out.cpu().numpy())

    def test_stride_copy_main_path(self):
        self._check([[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0])
        self._check([[np.int32, 0, [5, 5]], (3, 3), (1, 2), 1])

    def test_stride_copy_small_bytes(self):
        # 覆盖被删的“小张量兜底”分支：nbytes < 32 且为真实变换（转置式取数）
        self._check([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])

    def test_stride_copy_small_bytes_bool(self):
        # 覆盖被删的“小张量兜底”分支：1 字节 bool + nbytes < 32
        cpu_input = torch.arange(25).reshape(5, 5) % 3 == 0
        npu_input = cpu_input.npu()
        size, stride, offset = (5, 5), (1, 5), 0
        expected = self.cpu_ref(cpu_input, size, stride, offset).astype(np.uint8)
        for jit in (False, True):
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, size, stride, offset).astype(np.uint8)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def test_stride_copy_quant_int_repr(self):
        # 量化 view/ravel 的真实调用场景（TensorFactories.cpp / TensorShape.cpp）
        self._check([[np.int8, 0, [4, 4]], (4, 4), (1, 4), 0])

    def test_stride_copy_big_stride(self):
        # 覆盖被删的“大 stride 重排”分支：末维 stride >= 256
        self._check([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0])

    def test_stride_copy_offset(self):
        self._check([[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1])

    def test_stride_copy_broadcast(self):
        # stride 含 0：广播式取数
        self._check([[np.float32, 0, [4]], (3, 4), (0, 1), 0])

    def test_stride_copy_big_numel(self):
        self._check([[np.float32, 0, [1000]], (500, 2), (2, 1), 0])

    def test_stride_copy_ndim(self):
        self._check([[np.float32, 0, [10]], (4,), (2,), 1])
        self._check([[np.float32, 0, [32, 8, 2]], (8, 6, 2), (5, 4, 1), 1])
        self._check([[np.float32, 0, [2, 3, 4, 5, 6]], (2, 3, 4, 5, 3), (360, 120, 30, 6, 2), 0])

    def test_stride_copy_numpy_ref(self):
        # 独立参考值：numpy 手算，避免与算子用同一变换自证
        cpu_input = torch.randn(1, 300)
        npu_input = cpu_input.npu()
        size, stride, offset = (2, 2), (1, 256), 0
        flat = cpu_input.reshape(-1).numpy()
        expected = np.zeros(size, dtype=flat.dtype)
        for idx in np.ndindex(*size):
            expected[idx] = flat[offset + sum(i * s for i, s in zip(idx, stride))]
        for jit in (False, True):
            torch_npu.npu.set_compile_mode(jit_compile=jit)
            got = self.npu_exec(npu_input, size, stride, offset)
            with self.subTest(jit_compile=jit):
                self.assertRtolEqual(expected, got)

    def test_stride_copy_out_variant(self):
        # .out 版才是框架热路径真正调用的那个：
        # copy_d2d_dtype_baseformat -> op_plugin::npu_stride_copy_out
        self._check_out([[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0])
        self._check_out([[np.int8, 0, [5, 5]], (5, 5), (1, 5), 0])
        self._check_out([[np.float16, 0, [1, 300]], (2, 2), (1, 256), 0])

    def test_stride_copy_via_contiguous(self):
        # 间接覆盖（不依赖 .out 的 python 绑定）：A5 上真实 Core 的那条链
        # 非连续张量 .contiguous() -> copy_d2d_dtype_baseformat -> npu_stride_copy_out
        cpu_input = torch.randn(32, 8, 2)
        npu_input = cpu_input.npu()
        cpu_view = cpu_input.transpose(0, 1)
        npu_view = npu_input.transpose(0, 1)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        got = npu_view.contiguous().cpu().numpy()
        self.assertRtolEqual(cpu_view.contiguous().numpy(), got)


if __name__ == "__main__":
    run_tests()
