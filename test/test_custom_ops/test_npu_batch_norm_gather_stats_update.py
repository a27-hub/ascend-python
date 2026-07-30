import unittest
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices

torch.npu.set_compile_mode(jit_compile=False)


def _bf16_supported():
    """Per .md: BF16 inputs are only supported on Ascend950 (A5); A2/A3 reject them."""
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


def _npugraph_available():
    """npugraph_ex backend requires torch_npu compiled WITH torchair."""
    try:
        from torch_npu.dynamo import npugraph_ex  # noqa: F401
        return True
    except Exception:
        return False


bf16_only = unittest.skipUnless(_bf16_supported(), "BF16 inputs only supported on Ascend950 (A5)")
graph_only = unittest.skipUnless(_npugraph_available(), "requires torchair/npugraph_ex backend")


def cpu_op_exec(sum_all, square_sum_all, running_mean, running_var, momentum, eps, counts):
    """CPU golden (float64) of aclnnSyncBatchNormGatherStats semantics.

    batch_mean[c]   = sum_i sum_all[i,c] / sum_i counts[i]
    batch_var[c]    = sum_i square_sum_all[i,c] / sum_i counts[i] - batch_mean[c]^2
    batch_invstd[c] = 1 / sqrt(batch_var[c] + eps)
    running_mean    = running_mean*(1-momentum) + momentum*batch_mean       (in-place)
    running_var     = running_var*(1-momentum) + momentum*(batch_var * N/(N-1))  (in-place)
    """
    sa = sum_all.astype(np.float64)
    ss = square_sum_all.astype(np.float64)
    counts_f = counts.astype(np.float64)
    total = counts_f.sum()
    batch_mean = sa.sum(axis=0) / total
    batch_var = ss.sum(axis=0) / total - batch_mean ** 2
    batch_invstd = 1.0 / np.sqrt(batch_var + eps)
    new_rm = None
    new_rv = None
    if running_mean is not None:
        rm = running_mean.astype(np.float64)
        new_rm = rm * (1 - momentum) + momentum * batch_mean
    if running_var is not None:
        rv = running_var.astype(np.float64)
        new_rv = rv * (1 - momentum) + momentum * (batch_var * total / (total - 1))
    return batch_mean, batch_invstd, new_rm, new_rv


def npu_op_exec(input_tensor, sum_all, square_sum_all, running_mean, running_var, momentum, eps, counts):
    inp = input_tensor.npu()
    sa = sum_all.npu()
    ss = square_sum_all.npu()
    rm = running_mean.npu() if running_mean is not None else None
    rv = running_var.npu() if running_var is not None else None
    ct = counts.npu()
    rm_before = rm.clone() if rm is not None else None
    rv_before = rv.clone() if rv is not None else None
    batch_mean, batch_invstd = torch_npu.batch_norm_gather_stats_update(
        inp, sa, ss, rm, rv, momentum, eps, ct)
    rm_after = rm.clone() if rm is not None else None
    rv_after = rv.clone() if rv is not None else None
    return (batch_mean.cpu().numpy(), batch_invstd.cpu().numpy(),
            rm_before.cpu().numpy() if rm_before is not None else None,
            rv_before.cpu().numpy() if rv_before is not None else None,
            rm_after.cpu().numpy() if rm_after is not None else None,
            rv_after.cpu().numpy() if rv_after is not None else None)


# (numpy dtype, torch dtype, assertRtolEqual prec)
_DTYPE_MAP = {
    "bf16": (np.float32, torch.bfloat16, 0.004),  # bf16 down-cast for np; relaxed tol
    "fp16": (np.float16, torch.float16, 0.003),
    "fp32": (np.float32, torch.float32, 1e-4),
}


class TestBatchNormGatherStatsUpdate(TestCase):
    def _make_inputs(self, dtype_key, c_size=8, world=2, with_rm=True, with_rv=True, eps=1e-5,
                     momentum=0.1):
        np_dtype, t_dtype, _ = _DTYPE_MAP[dtype_key]
        np.random.seed(0)
        # generate in float32 then cast (bf16/fp16 have no native numpy random)
        sa = np.random.uniform(1, 10, [world, c_size]).astype(np.float32)
        ss = np.random.uniform(1, 10, [world, c_size]).astype(np.float32)
        rm = np.random.uniform(0, 1, [c_size]).astype(np.float32) if with_rm else None
        rv = np.random.uniform(1, 2, [c_size]).astype(np.float32) if with_rv else None
        counts = np.random.randint(8, 32, [world]).astype(np.int32)
        # input only carries the channel count C = input.size(1); values irrelevant.
        inp = np.random.uniform(-1, 1, [4, c_size]).astype(np.float32)

        sa_t = torch.from_numpy(sa).to(t_dtype)
        ss_t = torch.from_numpy(ss).to(t_dtype)
        rm_t = torch.from_numpy(rm).to(t_dtype) if rm is not None else None
        rv_t = torch.from_numpy(rv).to(t_dtype) if rv is not None else None
        counts_t = torch.from_numpy(counts)
        inp_t = torch.from_numpy(inp).to(t_dtype)
        # golden computed from the float32 source values (cast reference inputs to np_dtype
        # so the division uses the same values the NPU sees after down-cast).
        g_sa = sa.astype(np_dtype) if np_dtype != np.float32 else sa
        g_ss = ss.astype(np_dtype) if np_dtype != np.float32 else ss
        g_rm = (rm.astype(np_dtype) if np_dtype != np.float32 else rm) if rm is not None else None
        g_rv = (rv.astype(np_dtype) if np_dtype != np.float32 else rv) if rv is not None else None
        return inp_t, sa_t, ss_t, rm_t, rv_t, counts_t, g_sa, g_ss, g_rm, g_rv, counts, eps, momentum

    def _check(self, dtype_key, **kwargs):
        _, _, prec = _DTYPE_MAP[dtype_key]
        (inp_t, sa_t, ss_t, rm_t, rv_t, counts_t,
         g_sa, g_ss, g_rm, g_rv, counts, eps, momentum) = self._make_inputs(dtype_key, **kwargs)
        g_bm, g_bi, g_rm_new, g_rv_new = cpu_op_exec(
            g_sa, g_ss, g_rm, g_rv, momentum, eps, counts)
        bm, bi, rm_before, rv_before, rm_after, rv_after = npu_op_exec(
            inp_t, sa_t, ss_t, rm_t, rv_t, momentum, eps, counts_t)
        self.assertRtolEqual(g_bm.astype(np.float32), bm.astype(np.float32), prec)
        self.assertRtolEqual(g_bi.astype(np.float32), bi.astype(np.float32), prec)
        # in-place running_mean update
        if g_rm_new is not None:
            self.assertRtolEqual(g_rm_new.astype(np.float32), rm_after.astype(np.float32), prec)
        # in-place running_var update
        if g_rv_new is not None:
            self.assertRtolEqual(g_rv_new.astype(np.float32), rv_after.astype(np.float32), prec)

    # --- basic dtype coverage (no running stats) ---
    @bf16_only
    @SupportedDevices(['Ascend910B'])
    def test_basic_BF16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("bf16", with_rm=False, with_rv=False)

    @SupportedDevices(['Ascend910B'])
    def test_basic_F16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=False, with_rv=False)

    @SupportedDevices(['Ascend910B'])
    def test_basic_F32(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp32", with_rm=False, with_rv=False)

    # --- optional running_mean + running_var ---
    @bf16_only
    @SupportedDevices(['Ascend910B'])
    def test_running_mean_running_var_BF16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("bf16", with_rm=True, with_rv=True)

    @SupportedDevices(['Ascend910B'])
    def test_running_mean_running_var_F16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=True, with_rv=True)

    # --- optional running_mean only ---
    @bf16_only
    @SupportedDevices(['Ascend910B'])
    def test_running_mean_BF16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("bf16", with_rm=True, with_rv=False)

    @SupportedDevices(['Ascend910B'])
    def test_running_mean_F16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=True, with_rv=False)

    # --- optional running_var only ---
    @bf16_only
    @SupportedDevices(['Ascend910B'])
    def test_running_var_BF16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("bf16", with_rm=False, with_rv=True)

    @SupportedDevices(['Ascend910B'])
    def test_running_var_F16(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=False, with_rv=True)

    # --- epsilon values ---
    @SupportedDevices(['Ascend910B'])
    def test_epsilon_large(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=False, with_rv=False, eps=1e-3)

    @SupportedDevices(['Ascend910B'])
    def test_epsilon_small(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self._check("fp16", with_rm=False, with_rv=False, eps=1e-6)

    # --- output shape + dtype assertion ---
    @SupportedDevices(['Ascend910B'])
    def test_output_shape_dtype(self):
        torch.manual_seed(0)
        np.random.seed(0)
        inp_t, sa_t, ss_t, rm_t, rv_t, counts_t, *_ = self._make_inputs("fp16", with_rm=False, with_rv=False)
        bm, bi = torch_npu.batch_norm_gather_stats_update(
            inp_t.npu(), sa_t.npu(), ss_t.npu(), None, None, 0.1, 1e-5, counts_t.npu())
        self.assertEqual(bm.dtype, torch.float16)
        self.assertEqual(bi.dtype, torch.float16)
        self.assertEqual(tuple(bm.shape), (sa_t.size(1),))
        self.assertEqual(tuple(bi.shape), (sa_t.size(1),))

    # --- graph mode (torch.compile / npugraph_ex) ---
    @graph_only
    @SupportedDevices(['Ascend910B'])
    def test_graph_mode_npugraph_ex(self):
        torch.manual_seed(0)
        np.random.seed(0)
        inp_t, sa_t, ss_t, rm_t, rv_t, counts_t, g_sa, g_ss, g_rm, g_rv, counts, eps, momentum = \
            self._make_inputs("fp32", with_rm=True, with_rv=True)
        g_bm, g_bi, g_rm_new, g_rv_new = cpu_op_exec(g_sa, g_ss, g_rm, g_rv, momentum, eps, counts)

        def fn(inp, sa, ss, rm, rv, ct):
            return torch_npu.batch_norm_gather_stats_update(inp, sa, ss, rm, rv, momentum, eps, ct)

        compiled = torch.compile(fn, backend="npugraph_ex")
        bm, bi = compiled(inp_t.npu(), sa_t.npu(), ss_t.npu(), rm_t.npu(), rv_t.npu(), counts_t.npu())
        self.assertRtolEqual(g_bm.astype(np.float32), bm.cpu().numpy().astype(np.float32), 1e-3)
        self.assertRtolEqual(g_bi.astype(np.float32), bi.cpu().numpy().astype(np.float32), 1e-3)

    # ==================== Negative cases ====================
    @SupportedDevices(['Ascend910B'])
    def test_neg_square_sum_all_shape_mismatch(self):
        # square_sum_all shape != sum_all shape -> rejected (C++ TORCH_CHECK in this build,
        # aclnn ACLNN_ERR_PARAM_INVALID otherwise). Both raise RuntimeError.
        torch.manual_seed(0)
        np.random.seed(0)
        inp_t, sa_t, _, _, _, counts_t, *_ = self._make_inputs("fp32", with_rm=False, with_rv=False)
        bad_ss = torch.randn(2, 4, dtype=torch.float32)  # C=4 != sa_t.size(1)=8
        with self.assertRaises(RuntimeError):
            torch_npu.batch_norm_gather_stats_update(
                inp_t.npu(), sa_t.npu(), bad_ss.npu(), None, None, 0.1, 1e-5, counts_t.npu())

    @SupportedDevices(['Ascend910B'])
    def test_neg_counts_shape_mismatch(self):
        # counts.size(0) != sum_all.size(0) -> rejected (C++ TORCH_CHECK in this build,
        # aclnn ACLNN_ERR_PARAM_INVALID otherwise). Both raise RuntimeError.
        torch.manual_seed(0)
        np.random.seed(0)
        inp_t, sa_t, ss_t, _, _, _, *_ = self._make_inputs("fp32", with_rm=False, with_rv=False)
        bad_counts = torch.tensor([8, 8, 8], dtype=torch.int32)  # len 3 != sa_t.size(0)=2
        with self.assertRaises(RuntimeError):
            torch_npu.batch_norm_gather_stats_update(
                inp_t.npu(), sa_t.npu(), ss_t.npu(), None, None, 0.1, 1e-5, bad_counts.npu())


if __name__ == "__main__":
    run_tests()
