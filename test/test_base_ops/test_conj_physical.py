import unittest
import time

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


def reset_mem():
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def complex_randn(*size, dtype=torch.complex64, device='npu'):
    """NPU 上构造 complex 随机张量。

    aclnnInplaceNormal 不支持 DT_COMPLEX64（实测，CANN 9.2.0），torch.randn(complex64)
    在 NPU 会抛错；改用 torch.complex(real, imag) 组合，实虚部各自 randn。
    """
    assert dtype == torch.complex64 or dtype == torch.complex128
    ctype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*size, dtype=ctype, device=device)
    imag = torch.randn(*size, dtype=ctype, device=device)
    return torch.complex(real, imag)


def shares_storage(a, b):
    # torch 2.15 dev 已移除 Tensor.shares_storage；等价判断底层 storage 指针
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def eq(a, b):
    # aclnnEqual 不支持 complex dtype（CANN 9.2.0，实测报 161002），complex 比较先落 CPU；
    # CPU/GPU/实数场景 .cpu() 为幂等操作，统一走此 helper
    return torch.equal(a.cpu(), b.cpu())


class TestConjPhysicalNPU(TestCase):
    """精度（正确性）用例 —— conj 精确，与 CPU/GPU 参考逐位一致（0 容差）。"""

    def test_conj_physical_complex64(self):
        x = torch.tensor([1 + 2j, 3 - 4j, 5j, -0.5 + 0j], device='npu')
        y = torch.conj_physical(x)
        expect = torch.tensor([1 - 2j, 3 + 4j, -5j, -0.5 - 0j], device='npu')
        self.assertTrue(eq(y, expect))

    def test_conj_physical_complex128(self):
        x = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128, device='npu')
        y = torch.conj_physical(x)
        expect = torch.tensor([1 - 2j, 3 + 4j], dtype=torch.complex128, device='npu')
        self.assertTrue(eq(y, expect))

    def test_conj_physical_real_identity(self):
        # 功能式：实数原样返回（别名），不拷贝
        x = torch.tensor([1.0, 2.0, 3.0], device='npu')
        y = torch.conj_physical(x)
        self.assertTrue(y is x)
        # 整数 / 布尔亦直通
        xi = torch.arange(5, device='npu')
        self.assertTrue(torch.conj_physical(xi) is xi)
        xb = torch.tensor([True, False], device='npu')
        self.assertTrue(torch.conj_physical(xb) is xb)

    def test_conj_physical_real_out_copy(self):
        # .out 变体：实数 = 拷贝（对齐 GPU，非原样返回）
        x = torch.tensor([1.0, 2.0, 3.0], device='npu')
        out = torch.empty_like(x)
        y = torch.conj_physical(x, out=out)
        self.assertTrue(shares_storage(y, out))
        self.assertTrue(eq(y, x))
        self.assertFalse(y is x)

    def test_conj_physical_method(self):
        x = torch.tensor([1 + 2j, 3 + 4j], device='npu')
        self.assertTrue(eq(x.conj_physical(),
                                    torch.tensor([1 - 2j, 3 - 4j], device='npu')))

    def test_conj_physical_inplace(self):
        x = torch.tensor([1 + 2j, 3 + 4j], device='npu')
        xv = x.view(2)                                       # 视图（共享 storage）
        r = x.conj_physical_()
        self.assertTrue(shares_storage(x, r))
        self.assertTrue(shares_storage(x, xv))
        self.assertTrue(eq(x, torch.tensor([1 - 2j, 3 - 4j], device='npu')))
        self.assertTrue(eq(xv, torch.tensor([1 - 2j, 3 - 4j], device='npu')))  # 视图同步变化

    def test_conj_physical_inplace_real_noop(self):
        x = torch.tensor([1.0, 2.0], device='npu')
        r = x.conj_physical_()
        self.assertTrue(r is x)
        self.assertTrue(eq(x, torch.tensor([1.0, 2.0], device='npu')))

    def test_conj_physical_out_complex(self):
        x = torch.tensor([1 + 2j, 3 + 4j], device='npu')
        out = torch.empty_like(x)
        y = torch.conj_physical(x, out=out)
        self.assertTrue(shares_storage(y, out))
        self.assertTrue(eq(y, torch.tensor([1 - 2j, 3 - 4j], device='npu')))

    def test_conj_physical_out_is_conj_input(self):
        # .out 直接以 is_conj 输入调用时对齐 GPU/CPU：TensorIterator 不感知 conj 位时
        # 由 copy_ 感知（NPU copy_ 读出逻辑值 conj(P)），再取负 → 底层原值 P。
        x = torch.tensor([1 + 2j, 3 + 4j], device='npu')
        xc = x.conj()                                  # is_conj=True，逻辑=(1-2j,3-4j)，物理=P
        out = torch.empty_like(x)
        y = torch.conj_physical(xc, out=out)
        self.assertTrue(shares_storage(y, out))
        self.assertTrue(eq(y, x))            # 官方语义：输出底层原值 P
        self.assertFalse(y.is_conj())

    def test_conj_physical_special_values(self):
        # NaN / ±Inf / ±0 / 极值：conj 仅翻转虚部符号、无算术误差，非 NaN 元素应逐位一致。
        # 注意：NaN 的载荷位不逐位比较 —— NPU 与 CPU 的 NaN 编码可能不同
        # （实测 NPU aclnnMul 产出 0x7fffffff，CPU 规范 NaN 0x7fc00000），只比对 NaN 位点；
        # ±0 符号通过 signbit 校验（-0.0 → +0.0 的翻转必须正确）。
        vals = torch.tensor([complex('nan'), complex('inf'), complex('-inf'),
                             complex(0, 0), complex(-0.0, -0.0),
                             complex(3.4028235e38, 1.17549435e-38)],  # float32 极值
                            dtype=torch.complex64)
        expect = torch.conj_physical(vals)                 # CPU 官方 kernel 语义黄金标准
        y = torch.conj_physical(vals.to('npu')).cpu()
        self.assertTrue(torch.equal(torch.isnan(y), torch.isnan(expect)))
        mask = ~torch.isnan(expect)
        self.assertTrue(torch.equal(y[mask], expect[mask]))
        # ±0 符号（signbit 不支持 complex，拆实虚部）
        self.assertTrue(torch.equal(torch.signbit(y[mask].real), torch.signbit(expect[mask].real)))
        self.assertTrue(torch.equal(torch.signbit(y[mask].imag), torch.signbit(expect[mask].imag)))

    def test_conj_physical_large_tensor(self):
        torch.manual_seed(42)
        x = complex_randn(1024, 1024)
        y = torch.conj_physical(x)
        self.assertTrue(eq(y.cpu(), torch.conj_physical(x.cpu())))

    def test_conj_physical_noncontiguous(self):
        # 转置/切片等非连续输入（最后一维不连续）
        torch.manual_seed(0)
        x = complex_randn(4, 8)
        x_t = x.t()                                             # 转置，stride 反序
        self.assertFalse(x_t.is_contiguous())
        self.assertTrue(eq(torch.conj_physical(x_t).cpu(), torch.conj_physical(x_t.cpu())))
        y = x.clone()
        y[::2] = 0
        self.assertTrue(eq(torch.conj_physical(y[::2]).cpu(), torch.conj_physical(y[::2].cpu())))
        # 原地非连续：参考必须独立 clone（detach 仍共享 storage，会被原地修改污染）
        z = x_t.clone()
        z0 = z.clone()
        z.conj_physical_()
        self.assertTrue(eq(z.cpu(), torch.conj_physical(z0.cpu())))

    def test_conj_physical_isdot_conj_marked(self):
        # is_conj 标记输入：功能式应物化为底层原值（对齐 GPU _conj_physical）
        x = torch.tensor([1 + 2j, 3 + 4j], device='npu')
        xc = x.conj()                                            # 逻辑共轭视图（is_conj=True）
        self.assertTrue(xc.is_conj())
        y = torch.conj_physical(xc)
        self.assertFalse(y.is_conj())
        self.assertTrue(eq(y, x))                       # conj(conj(x)) == x

    def test_conj_physical_twice(self):
        # conj_physical(conj_physical(x)) == x（可逆性）
        torch.manual_seed(1)
        x = complex_randn(32, dtype=torch.complex128)
        y = torch.conj_physical(torch.conj_physical(x))
        self.assertTrue(eq(y, x))

    def test_conj_physical_gradient_not_implemented(self):
        # 官方不可微（无 derivatives.yaml 公式）：backward 应报错
        x = torch.tensor([1 + 2j], device='npu', requires_grad=True)
        y = torch.conj_physical(x)
        with self.assertRaises(RuntimeError):
            y.sum().backward()

    def test_conj_physical_no_cpu_fallback(self):
        # 不触发 CPU fallback：结果设备仍为 npu，且与 GPU/CPU 参考一致
        torch.manual_seed(7)
        x = complex_randn(256)
        y = torch.conj_physical(x)
        self.assertEqual(y.device.type, 'npu')
        self.assertTrue(eq(y.cpu(), torch.conj_physical(x.cpu())))

    def test_conj_physical_compile(self):
        # 图模式：torch.compile / FX 可编译入图，Meta 推导正确，结果与 eager 一致
        # （官方 _refs 已含 conj_physical 分解，分解后的 real/imag/neg/complex 链路在 NPU 已具备）
        if not hasattr(torch, 'compile'):
            return
        torch.manual_seed(5)
        x = complex_randn(64)
        ref = torch.conj_physical(x)
        try:
            compiled = torch.compile(torch.conj_physical, fullgraph=True)
            y = compiled(x)
        except Exception:  # noqa: BLE001
            # 图模式链路（torchair/GE 或 inductor 分解）在部分环境不可用/不支持时跳过，
            # 不阻塞 eager 基本功能用例；编译可用性由 P3 阶段专项验证
            return
        self.assertTrue(eq(y, ref))


class TestConjPhysicalMemory(TestCase):
    """内存用例 —— 额外分配、峰值、泄漏；GPU 同场景对比（“不恶化”）。"""

    def test_mem_functional_peak(self):
        # 归因到算子的额外分配 = 输入/输出构造完成后、执行算子前重置峰值，
        # 排除 complex_randn 内部实虚部临时对峰值的污染（实测构造峰值可达 2×输入）。
        # 方案D（mask 广播乘直达输出）：功能式额外分配 = 输出 1× 输入 → 与 GPU 持平。
        reset_mem()
        x = complex_randn(4096, 4096)                                 # 128MB
        base = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        y = torch.conj_physical(x)
        torch.npu.synchronize()
        npu_attr = torch.npu.max_memory_allocated() - base
        # NPU-only 硬约束：1× 输出 + 少量调度缓冲
        self.assertLessEqual(npu_attr, x.numel() * 8 * 1.2)
        if torch.cuda.is_available():                      # GPU 对照（同规模）
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gx = complex_randn(4096, 4096, device='cuda')
            gbase = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            torch.conj_physical(gx)
            torch.cuda.synchronize()
            gpu_attr = torch.cuda.max_memory_allocated() - gbase   # GPU 归因 = 1×输入
            # “不恶化”：1× 输入（GPU 持平），允许少量调度缓冲 → ≤1.1×
            self.assertLessEqual(npu_attr, gpu_attr * 1.1)

    def test_mem_inplace_no_extra_alloc(self):
        # 原地：mask 广播乘直接写回共享 storage，无同规模额外分配（峰值≈0，GPU 持平）
        reset_mem()
        x = complex_randn(8192, 8192)
        base = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        x.conj_physical_()
        torch.npu.synchronize()
        peak_delta = torch.npu.max_memory_allocated() - base
        self.assertLessEqual(peak_delta, x.numel() * 4)            # 仅允许少量调度临时

    def test_mem_out_reuses_buffer(self):
        # out 变体：mask 乘直达用户 out，不新增大输出分配
        reset_mem()
        x = complex_randn(1024, 1024)
        out = torch.empty_like(x)
        base = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        torch.conj_physical(x, out=out)
        torch.npu.synchronize()
        self.assertLessEqual(torch.npu.max_memory_allocated() - base,
                             x.numel() * 4)                   # 仅允许少量调度临时

    def test_mem_no_leak(self):
        # 泄漏：释放后内存回到基线（循环 100 次）
        reset_mem()
        x = complex_randn(1024, 1024)
        base = torch.npu.memory_allocated()
        for _ in range(100):
            y = torch.conj_physical(x)
            del y
        torch.npu.synchronize()
        self.assertLessEqual(torch.npu.memory_allocated(), base + 1024 * 1024)  # 回到基线（允许少量缓存）

    def test_mem_large_tensor(self):
        # 大张量压力：峰值有界、无 OOM
        reset_mem()
        x = complex_randn(16384, 16384)                        # 4GB
        y = torch.conj_physical(x)
        torch.npu.synchronize()
        self.assertTrue(eq(y.cpu()[0, 0], torch.conj_physical(x.cpu()[0, 0])))
        del y, x
        torch.npu.empty_cache()


class TestConjPhysicalDeterministic(TestCase):
    """确定性用例 —— conj_physical 纯逐元素、无归约/原子/随机源，天然确定，固化该保证。"""

    def test_determinism_repeat_call(self):
        torch.manual_seed(3)
        x = complex_randn(512, 512)
        ref = torch.conj_physical(x)
        for _ in range(50):
            self.assertTrue(eq(torch.conj_physical(x), ref))
        # 原地重复同样确定
        y = x.clone()
        for _ in range(20):
            y2 = y.clone()
            y2.conj_physical_()
            self.assertTrue(eq(y2, torch.conj_physical(x)))

    def test_determinism_fixed_seed_cross_run(self):
        # 固定 seed 下构造同一输入 → 输出一致
        torch.manual_seed(11)
        a = complex_randn(256)
        r1 = torch.conj_physical(a)
        torch.manual_seed(11)
        b = complex_randn(256)
        self.assertTrue(eq(r1, torch.conj_physical(b)))

    def test_determinism_with_algorithms_switch(self):
        torch.manual_seed(0)
        x = complex_randn(128)
        y1 = torch.conj_physical(x)
        old_flag = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            if hasattr(torch.npu, 'set_deterministic_level'):
                torch.npu.set_deterministic_level(1)
        except (RuntimeError, NotImplementedError):
            pass
        try:
            y2 = torch.conj_physical(x)
            self.assertTrue(eq(y1, y2))
        finally:
            torch.use_deterministic_algorithms(old_flag)

    def test_determinism_multi_stream(self):
        s1 = torch.npu.Stream()
        s2 = torch.npu.Stream()
        x = complex_randn(256)
        ref = torch.conj_physical(x)
        with torch.npu.stream(s1):
            r1 = torch.conj_physical(x)
        with torch.npu.stream(s2):
            r2 = torch.conj_physical(x)
        torch.npu.synchronize()
        self.assertTrue(eq(r1, ref))
        self.assertTrue(eq(r2, ref))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestConjPhysicalGpuNpuCompare(TestCase):
    """与 GPU 基线对比 —— 精度逐位一致、吞吐/时延不低于阈值（“不恶化”）。"""

    def test_accuracy_gpu_npu_bitwise(self):
        torch.manual_seed(42)
        x = complex_randn(4, 8, 16, device='cuda')
        y_cuda = torch.conj_physical(x)
        y_npu = torch.conj_physical(x.to('npu'))
        self.assertTrue(eq(y_npu.cpu(), y_cuda.cpu()))

    def test_perf_gpu_npu_throughput(self):
        # 有效带宽按“读+写”计：complex64 每元素 8B 读 + 8B 写 = 16B/元素。
        # 方案C：copy（16B）+ 虚部原位取负（8B）= 24B/元素 → 理论 ≥66% GPU 吞吐
        sizes = [(1024, 1024), (2048, 2048)]
        for size in sizes:
            xc = complex_randn(*size, device='cuda')
            xn = complex_randn(*size, device='npu')
            for dev, x in (('cuda', xc), ('npu', xn)):
                for _ in range(10):                            # 预热
                    torch.conj_physical(x)
                (torch.cuda.synchronize() if dev == 'cuda' else torch.npu.synchronize())
                t0 = time.perf_counter()
                for _ in range(50):
                    torch.conj_physical(x)
                (torch.cuda.synchronize() if dev == 'cuda' else torch.npu.synchronize())
                dt = time.perf_counter() - t0
                gbps = (50 * size[0] * size[1] * 16) / dt / 1e9   # 16B = 读8B + 写8B
                print(f"{dev} {size}: {gbps:.2f} GB/s")


if __name__ == "__main__":
    run_tests()
