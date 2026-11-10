import torch

import torch_npu  # noqa: F401 - register the NPU backend
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
INT_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]

def _nan(shape, dtype=torch.float32):
    tensor = torch.randn(shape, dtype=dtype)
    if tensor.numel() > 0:
        flat = tensor.reshape(-1)
        flat[0] = float("nan")
        flat[-1] = float("nan")
    return tensor


def _pair_tensor(dtype, shape):
    """按 dtype 生成含 NaN 的浮点 / 整数 / bool 输入。"""
    if dtype.is_floating_point:
        return _nan(shape, torch.float32).to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool)
    low = 0 if dtype == torch.uint8 else -5
    return torch.randint(low, 5, shape, dtype=dtype)


class TestFmin(TestCase):

    def _assert_same(self, npu_out, cpu_out):
        self.assertEqual(npu_out.device.type, "npu")
        self.assertEqual(npu_out.dtype, cpu_out.dtype)
        self.assertEqual(npu_out.shape, cpu_out.shape)
        torch.testing.assert_close(npu_out.cpu(), cpu_out, rtol=0, atol=0, equal_nan=True)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_nan_matrix(self):
        nan = float("nan")
        inf = float("inf")
        cases = [
            ([nan, 1.0, 2.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, 3.0], [nan, 2.0, 3.0]),
            ([nan, nan], [nan, nan]),
            ([inf, -inf, 0.0], [-inf, inf, -0.0]),
            ([nan, inf], [inf, nan]),
        ]
        for dtype in FLOAT_DTYPES:
            for case, (data_a, data_b) in enumerate(cases):
                for reverse in (False, True):
                    with self.subTest(dtype=dtype, case=case, reverse=reverse):
                        a = torch.tensor(data_a, dtype=dtype)
                        b = torch.tensor(data_b, dtype=dtype)
                        cpu_a, cpu_b = (b, a) if reverse else (a, b)
                        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_float_dtypes(self):
        for dtype in FLOAT_DTYPES:
            cpu_a = _nan((5, 6), dtype)
            cpu_b = _nan((5, 6), dtype)
            cpu_out = torch.fmin(cpu_a, cpu_b)
            npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

            # Check both the CPU fixture and its device round-trip: a flushed
            # input must not be mistaken for a successful fmin subnormal test.
            info = torch.finfo(dtype)
            small = info.tiny * info.eps
            a = torch.tensor([small, -small, 2 * small, -2 * small], dtype=dtype)
            self.assertTrue(bool(((a.abs() > 0) & (a.abs() < info.tiny)).all()))
            na = a.npu()
            torch.testing.assert_close(na.cpu(), a, rtol=0, atol=0)
            b = torch.tensor([0, 0, small, -small], dtype=dtype)
            for left, right in ((a, b), (b, a)):
                self._assert_same(torch.fmin(left.npu(), right.npu()), torch.fmin(left, right))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_integer_and_bool(self):
        for dtype in INT_DTYPES:
            low = 0 if dtype == torch.uint8 else -8
            cpu_a = torch.randint(low, 8, (5, 6), dtype=dtype)
            cpu_b = torch.randint(low, 8, (5, 6), dtype=dtype)
            cpu_out = torch.fmin(cpu_a, cpu_b)
            npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

        cpu_a = torch.randint(0, 2, (4, 4), dtype=torch.bool)
        cpu_b = torch.randint(0, 2, (4, 4), dtype=torch.bool)
        cpu_out = torch.fmin(cpu_a, cpu_b)
        npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_fp64_fallback(self):
        # FP64 follows the fmax CPU fallback policy; values and output device must match.
        # fallback 告警由 TORCH_NPU_WARN_ONCE 发出，走 C++ 日志通道（不进 Python
        # warnings，无法用 assertWarns 断言），日志需由独立测试确认。
        cpu_a = _nan((4, 5), torch.float64)
        cpu_b = _nan((4, 5), torch.float64)
        cpu_out = torch.fmin(cpu_a, cpu_b)
        npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
        self.assertEqual(npu_out.device.type, "npu")
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_complex_error(self):
        for dtype in [torch.complex64, torch.complex128]:
            cpu_a = torch.randn(4, dtype=dtype)
            cpu_b = torch.randn(4, dtype=dtype)
            with self.assertRaisesRegex(TypeError, "fmin not implemented for complex tensors"):
                torch.fmin(cpu_a.npu(), cpu_b.npu())
            with self.assertRaisesRegex(TypeError, "fmin not implemented for complex tensors"):
                out = torch.empty(4, dtype=dtype).npu()
                torch.fmin(cpu_a.npu(), cpu_b.npu(), out=out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_type_promotion(self):
        # 对齐上游 test_maximum_minimum_type_promotion：双输入 dtype 笛卡尔积。
        # fp64（fallback）与 complex（报错）由独立用例覆盖，不进本矩阵。
        dtypes = FLOAT_DTYPES + INT_DTYPES + [torch.bool]
        for dtype_a in dtypes:
            for dtype_b in dtypes:
                cpu_a = _pair_tensor(dtype_a, (4, 4))
                cpu_b = _pair_tensor(dtype_b, (4, 4))
                cpu_out = torch.fmin(cpu_a, cpu_b)
                npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
                self.assertEqual(npu_out.dtype, torch.result_type(cpu_a, cpu_b))
                self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_mixed_dtype_broadcast_and_layout(self):
        # 交叉用例：混合 dtype × 广播（2 组）/ 非连续（2 组）
        cpu_a = torch.randn(4, 1, dtype=torch.float32)
        cpu_b = torch.randn(1, 5, dtype=torch.float16)
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

        cpu_a = torch.randn(8, 1, 3, dtype=torch.bfloat16)
        cpu_b = torch.randint(-5, 5, (1, 4, 3), dtype=torch.int64)
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

        base = torch.randn(6, 6, dtype=torch.float32)
        cpu_a = base.t()
        cpu_b = torch.randn(6, 6, dtype=torch.float16)
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

        base = torch.randn(12, 6, dtype=torch.float16)
        cpu_a = base[::2]
        cpu_b = torch.randn(6, 6, dtype=torch.float32)
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_channels_last(self):
        # 布局：通道格式转换后的输入
        for dtype in (torch.float32, torch.float16):
            cpu_a = torch.randn(2, 3, 4, 5, dtype=dtype).to(memory_format=torch.channels_last)
            cpu_b = torch.randn(2, 3, 4, 5, dtype=dtype).to(memory_format=torch.channels_last)
            self.assertTrue(cpu_a.is_contiguous(memory_format=torch.channels_last))
            self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_extremal(self):
        # 边界：浮点极值与 ±inf、整型极值
        for dtype in FLOAT_DTYPES:
            info = torch.finfo(dtype)
            vals = [info.max, info.min, float("inf"), -float("inf"), 0.0]
            cpu_a = torch.tensor(vals, dtype=dtype)
            cpu_b = torch.tensor(vals[::-1], dtype=dtype)
            self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))
        for dtype in INT_DTYPES:
            info = torch.iinfo(dtype)
            negative_or_zero = 0 if dtype == torch.uint8 else -1
            cpu_a = torch.tensor([info.max, info.min, 0, negative_or_zero], dtype=dtype)
            cpu_b = torch.tensor([info.min, info.max, 0, 1], dtype=dtype)
            self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_broadcast(self):
        cases = [((4, 1), (1, 4)), ((), (4, 4)), ((4, 4), ()), ((0, 3), (0, 3)), ((1, 0, 4), (3, 1, 4))]
        for shape_a, shape_b in cases:
            cpu_a = torch.randn(shape_a)
            cpu_b = torch.randn(shape_b)
            cpu_out = torch.fmin(cpu_a, cpu_b)
            npu_out = torch.fmin(cpu_a.npu(), cpu_b.npu())
            self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_non_contiguous(self):
        base_a = torch.randn(6, 6)
        base_b = torch.randn(6, 6)
        cpu_a = base_a.t()
        cpu_b = base_b.t()
        self.assertFalse(cpu_a.is_contiguous() or cpu_b.is_contiguous())
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_cpu_zero_dim(self):
        for dtype in (torch.float32, torch.float64):
            a = _nan((4, 4), dtype)
            na = a.npu()
            for scalar in (torch.tensor(float("nan")), torch.tensor(0.5, dtype=torch.float64)):
                for reverse in (False, True):
                    with self.subTest(dtype=dtype, scalar_dtype=scalar.dtype, reverse=reverse):
                        inputs = (scalar, na) if reverse else (na, scalar)
                        expected = torch.fmin(scalar, a) if reverse else torch.fmin(a, scalar)
                        self._assert_same(torch.fmin(*inputs), expected)
                        out = torch.empty_like(na, dtype=expected.dtype)
                        self.assertIs(torch.fmin(*inputs, out=out), out)
                        self._assert_same(out, expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_normal(self):
        cpu_a = _nan((4, 4))
        cpu_b = _nan((4, 4))
        cpu_out = torch.empty(4, 4)
        npu_out = torch.empty(4, 4).npu()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        ret = torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self.assertEqual(ret.data_ptr(), npu_out.data_ptr())
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_non_contiguous(self):
        cpu_a = torch.randn(6, 6)
        cpu_b = torch.randn(6, 6)
        cpu_out = torch.empty(12, 6)[::2]
        npu_out = torch.empty(12, 6).npu()[::2]
        storage = npu_out.untyped_storage().data_ptr()
        stride, offset = npu_out.stride(), npu_out.storage_offset()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        returned = torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self.assertIs(returned, npu_out)
        self.assertEqual(npu_out.untyped_storage().data_ptr(), storage)
        self.assertEqual(npu_out.stride(), stride)
        self.assertEqual(npu_out.storage_offset(), offset)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_dtype_mismatch(self):
        # 上游语义：out dtype 与提升结果不同时不报错，按 cast 写入
        cpu_a = _nan((4, 4), torch.float16)
        cpu_b = _nan((4, 4), torch.float16)
        cpu_out = torch.empty(4, 4, dtype=torch.float32)
        npu_out = torch.empty(4, 4, dtype=torch.float32).npu()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_complex(self):
        # out 为 complex 时上游按 cast 写入实部，虚部为 0
        cpu_a = torch.tensor([1.0, -2.0])
        cpu_b = torch.tensor([2.0, -3.0])
        cpu_out = torch.empty(2, dtype=torch.complex64)
        npu_out = torch.empty(2, dtype=torch.complex64).npu()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_shape_mismatch(self):
        # 上游语义：shape 不匹配按 resize 处理并给弃用告警
        cpu_a = torch.randn(4, 4)
        cpu_b = torch.randn(4, 4)
        cpu_out = torch.empty(2, 2)
        npu_out = torch.empty(2, 2).npu()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        # torch_npu 的 assertWarns 是函数式签名 assertWarns(call_fn)
        self.assertWarns(lambda: torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out))
        self.assertEqual(npu_out.shape, cpu_out.shape)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_shape_mismatch_float64_fallback(self):
        # fp64 走 CPU fallback 路径，out 语义必须与原生路径/上游一致：
        # resize + 弃用告警，而不是 "expanded size ... must match" 的报错
        cpu_a = torch.randn(4, 4, dtype=torch.float64)
        cpu_b = torch.randn(4, 4, dtype=torch.float64)
        cpu_out = torch.empty(2, 2, dtype=torch.float64)
        npu_out = torch.empty(2, 2, dtype=torch.float64).npu()
        torch.fmin(cpu_a, cpu_b, out=cpu_out)
        self.assertWarns(
            lambda: torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out))
        self.assertEqual(npu_out.shape, cpu_out.shape)
        self._assert_same(npu_out, cpu_out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_zero_element_resize(self):
        # 零元素 out：上游按 resize 处理且不告警（PyTorch 弃用告警推荐用
        # t.resize_(0) 复用 out），fallback 路径曾因 copy_ 对空张量早退而静默返回空结果
        for dtype in (torch.float32, torch.float64):
            cpu_a = torch.randn(4, 4, dtype=dtype)
            cpu_b = torch.randn(4, 4, dtype=dtype)
            cpu_out = torch.empty(0, dtype=dtype)
            npu_out = torch.empty(0, dtype=dtype).npu()
            torch.fmin(cpu_a, cpu_b, out=cpu_out)
            torch.fmin(cpu_a.npu(), cpu_b.npu(), out=npu_out)
            self.assertEqual(npu_out.shape, cpu_out.shape)
            self.assertEqual(npu_out.numel(), 16)
            self._assert_same(npu_out, cpu_out)
        # 上游 TensorIterator 语义：可安全 cast（int→int、int→float）按 cast 写入；
        # 不可安全 cast（int→bool、float→int）报与 CPU 一致的 RuntimeError
        cpu_a = torch.tensor([2, 1, 3], dtype=torch.int32)
        cpu_b = torch.tensor([1, 2, 3], dtype=torch.int32)

        out64 = torch.empty(3, dtype=torch.int64).npu()
        torch.fmin(cpu_a.npu(), cpu_b.npu(), out=out64)
        self.assertEqual(out64.dtype, torch.int64)
        self._assert_same(out64, torch.fmin(cpu_a, cpu_b).to(torch.int64))

        out_bool = torch.empty(3, dtype=torch.bool).npu()
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.fmin(cpu_a.npu(), cpu_b.npu(), out=out_bool)

        cpu_f = torch.randn(3)
        out_int = torch.empty(3, dtype=torch.int32).npu()
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.fmin(cpu_f.npu(), cpu_f.npu(), out=out_int)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_same_tensor(self):
        # out 与 self 为同一 tensor 时允许，返回即 out
        cpu_a = torch.tensor([1.0, 5.0, float("nan")])
        cpu_b = torch.tensor([2.0, 3.0, 1.0])
        cpu_expected = torch.fmin(cpu_a.clone(), cpu_b)
        for out_index in (0, 1):
            with self.subTest(out_index=out_index):
                inputs = [cpu_a.npu(), cpu_b.npu()]
                out = inputs[out_index]
                storage, stride = out.untyped_storage().data_ptr(), out.stride()
                ret = torch.fmin(*inputs, out=out)
                self.assertIs(ret, out)
                self.assertEqual(out.untyped_storage().data_ptr(), storage)
                self.assertEqual(out.stride(), stride)
                self._assert_same(out, cpu_expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_partial_overlap(self):
        # 部分重叠：复刻上游 RuntimeError，不静默 clone。
        # out 形状必须与结果一致，否则先触发 resize 而非重叠校验。
        base = torch.tensor([1.0, 5.0, 2.0, -1.0, 9.0])
        cpu_b = torch.tensor([2.0, 3.0, 1.0, 0.0])
        cpu_a = base[:4]
        npu_base = base.npu()
        npu_a = npu_base[:4]
        npu_b = cpu_b.npu()
        with self.assertRaisesRegex(RuntimeError, "refer to a single memory location"):
            torch.fmin(cpu_a, cpu_b, out=base[1:])
        with self.assertRaisesRegex(RuntimeError, "refer to a single memory location"):
            torch.fmin(npu_a, npu_b, out=npu_base[1:])

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_zero_sign(self):
        # ±0 不做 bitwise 断言（IEEE 754 未规定，CPU x86/aarch64 本身即不一致）
        cpu_a = torch.tensor([0.0, -0.0, 0.0, -0.0])
        cpu_b = torch.tensor([-0.0, 0.0, 1.0, -1.0])
        self._assert_same(torch.fmin(cpu_a.npu(), cpu_b.npu()), torch.fmin(cpu_a, cpu_b))
        # Ties above are value comparisons only. Here each zero wins strictly,
        # so its sign is determined by the selected input, not a tie convention.
        a = torch.tensor([-0.0, 1.0, 0.0, 1.0])
        b = torch.tensor([1.0, -0.0, 1.0, 0.0])
        actual = torch.fmin(a.npu(), b.npu())
        expected = torch.fmin(a, b)
        self._assert_same(actual, expected)
        self.assertEqual(torch.signbit(actual.cpu()), torch.signbit(expected))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_backward(self):
        nan = float("nan")
        cases = (([nan, 1.0, 5.0, 2.0], [2.0, nan, 1.0, 2.0]),
                 ([[nan], [2.0], [1.0]], [[1.0, nan, 2.0]]))
        for dtype in FLOAT_DTYPES:
            for index, (a, b) in enumerate(cases):
                with self.subTest(dtype=dtype, case=index):
                    cpu_self = torch.tensor(a, dtype=dtype, requires_grad=True)
                    cpu_other = torch.tensor(b, dtype=dtype, requires_grad=True)
                    npu_self = cpu_self.detach().npu().requires_grad_(True)
                    npu_other = cpu_other.detach().npu().requires_grad_(True)
                    # Build a fresh graph twice; leaf gradients must accumulate,
                    # including reduction back to the broadcast input shapes.
                    for step in range(2):
                        cpu_out = torch.fmin(cpu_self, cpu_other)
                        npu_out = torch.fmin(npu_self, npu_other)
                        grad = torch.arange(1, cpu_out.numel() + 1, dtype=dtype).reshape(cpu_out.shape)
                        cpu_out.backward(grad)
                        npu_out.backward(grad.npu())
                        self._assert_same(npu_out, cpu_out.detach())
                        self._assert_same(npu_self.grad, cpu_self.grad)
                        self._assert_same(npu_other.grad, cpu_other.grad)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_backward_tie(self):
        for dtype in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                cpu_self = torch.tensor([1.0, 0.0, -0.0, 0.0, -0.0], dtype=dtype, requires_grad=True)
                cpu_other = torch.tensor([1.0, -0.0, 0.0, 0.0, -0.0], dtype=dtype, requires_grad=True)
                npu_self = cpu_self.detach().npu().requires_grad_(True)
                npu_other = cpu_other.detach().npu().requires_grad_(True)
                torch.fmin(cpu_self, cpu_other).sum().backward()
                torch.fmin(npu_self, npu_other).sum().backward()
                self._assert_same(npu_self.grad, cpu_self.grad)
                self._assert_same(npu_other.grad, cpu_other.grad)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_internal_overlap(self):
        a, b = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.fmin(a, b, out=torch.empty(1).expand(3))
        out = torch.empty(1, device="npu").expand(3)
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.fmin(a.npu(), b.npu(), out=out)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_out_autograd(self):
        for grad_index in (0, 1, 2):
            with self.subTest(grad_index=grad_index):
                cpu = [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]), torch.empty(2)]
                npu = [value.npu() for value in cpu]
                cpu[grad_index].requires_grad_()
                npu[grad_index].requires_grad_()
                with self.assertRaisesRegex(RuntimeError, "automatic differentiation"):
                    torch.fmin(cpu[0], cpu[1], out=cpu[2])
                with self.assertRaisesRegex(RuntimeError, "automatic differentiation"):
                    torch.fmin(npu[0], npu[1], out=npu[2])

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_invalid_inputs(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                cpu = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
                a = cpu.npu()
                out = torch.empty_like(a)
                for inputs in ((a, cpu), (cpu, a)):
                    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                        torch.fmin(*inputs)
                    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                        torch.fmin(*inputs, out=out)
                with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have device"):
                    torch.fmin(a, a, out=torch.empty_like(cpu))
                with self.assertRaises(RuntimeError):
                    torch.fmin(a, torch.ones(4, dtype=dtype, device=a.device))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_cross_device_errors(self):
        if torch.npu.device_count() < 2:
            self.skipTest("requires two visible NPU devices")
        for dtype in (torch.float32, torch.float64):
            cpu = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            a, b = cpu.to("npu:0"), cpu.to("npu:1")
            for first, second in ((a, b), (b, a)):
                with self.subTest(dtype=dtype, device=first.device):
                    out = torch.empty_like(first)
                    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                        torch.fmin(first, second)
                    with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                        torch.fmin(first, second, out=out)
                    with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have device"):
                        torch.fmin(first, first, out=torch.empty_like(second))
                    # Keep the check in this process so a deferred device error
                    # from any rejected call is visible at synchronization.
                    actual = torch.fmin(first, first)
                    torch.npu.synchronize(first.device)
                    self._assert_same(actual, cpu)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_fmin_fp64_precision_and_mixed(self):
        # Additional validation only; no separate production execution path.
        a = torch.tensor([1 + 2**-41, 1 + 2**-40, float("nan"), -float("inf")], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0, 3.0, float("nan")], dtype=torch.float32)
        for left, right in ((a, b), (b, a), (a, b.double())):
            expected = torch.fmin(left, right)
            self._assert_same(torch.fmin(left.npu(), right.npu()), expected)
            out = torch.empty(0, dtype=torch.float64, device="npu")
            self._assert_same(torch.fmin(left.npu(), right.npu(), out=out), expected)


if __name__ == "__main__":
    run_tests()
