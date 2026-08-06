import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuCrossEntropySumexpAndIndexLogit(TestCase):

    def cpu_op_exec(self, vocab_parallel_logits, target, global_logits_max,
                    vocab_start_index, vocab_end_index):
        """
        CPU golden implementation using FP64 for high precision.

        Returns:
            predicted_logits: target 对应 logit - global_max，不在当前 rank 时为 0
            sum_exp_logits: 本地 exp(logits - global_max) 沿最后一维求和
            exp_logits: exp(logits - global_max)
            target_offset: target - vocab_start_index，不在当前 rank 时为 0
            target_mask: 1=不在当前 rank，0=在
        """
        # 转 FP64 计算
        logits = vocab_parallel_logits.astype(np.float64)
        gmax = global_logits_max.astype(np.float64)
        tgt = target.astype(np.int32)

        # Shape 推导
        target_shape = tgt.shape
        logits_shape = logits.shape
        v_local = logits_shape[-1]

        # 初始化输出
        predicted_logits = np.zeros(target_shape, dtype=np.float32)
        sum_exp_logits = np.zeros(target_shape, dtype=np.float32)
        exp_logits = np.zeros(logits_shape, dtype=np.float32)
        target_offset = np.zeros(target_shape, dtype=np.int32)
        target_mask = np.ones(target_shape, dtype=np.int32)

        # Flatten to 2D for easier processing
        if len(logits_shape) == 3:
            S, B, V = logits_shape
            N = S * B
            logits_2d = logits.reshape(N, V)
            gmax_1d = gmax.reshape(N)
            tgt_1d = tgt.reshape(N)
        else:
            N, V = logits_shape
            logits_2d = logits
            gmax_1d = gmax
            tgt_1d = tgt

        # Process each token
        for i in range(N):
            t = tgt_1d[i]
            gm = gmax_1d[i]

            # Compute mask and offset
            if t >= vocab_start_index and t < vocab_end_index:
                target_mask.flat[i] = 0
                offset = t - vocab_start_index
                target_offset.flat[i] = offset
                # Predicted logit
                predicted_logits.flat[i] = logits_2d[i, offset] - gm
            else:
                target_mask.flat[i] = 1
                target_offset.flat[i] = 0
                predicted_logits.flat[i] = 0.0

            # Compute exp(logits - global_max) and sum
            shifted = logits_2d[i, :] - gm
            exp_vals = np.exp(shifted)
            exp_logits.reshape(N, V)[i, :] = exp_vals.astype(np.float32)
            sum_exp_logits.flat[i] = np.sum(exp_vals).astype(np.float32)

        return predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask

    def npu_op_exec(self, vocab_parallel_logits, target, global_logits_max,
                    vocab_start_index, vocab_end_index):
        predicted, sum_exp, exp_out, offset, mask = torch_npu.npu_cross_entropy_sumexp_and_index_logit(
            vocab_parallel_logits, target, global_logits_max,
            vocab_start_index, vocab_end_index)
        return predicted.cpu(), sum_exp.cpu(), exp_out.cpu(), offset.cpu(), mask.cpu()

    @SupportedDevices(['Ascend910B', 'Ascend910'])
    def test_npu_cross_entropy_sumexp_and_index_logit_fp32_2d(self):
        """测试 FP32 输入，2 维 [N, V_local]"""
        N = 128
        V_local = 2048
        vocab_start = 1000
        vocab_end = vocab_start + V_local

        # 生成输入
        np.random.seed(42)
        logits_np = np.random.randn(N, V_local).astype(np.float32) * 2.0
        target_np = np.random.randint(0, 50000, size=(N,)).astype(np.int32)
        # 确保部分 target 在当前 rank，部分越界
        target_np[:32] = np.random.randint(vocab_start, vocab_end, size=(32,))
        global_max_np = np.max(logits_np, axis=-1).astype(np.float32)

        # CPU golden
        pred_cpu, sum_cpu, exp_cpu, off_cpu, mask_cpu = self.cpu_op_exec(
            logits_np, target_np, global_max_np, vocab_start, vocab_end)

        # NPU 执行
        logits_npu = torch.from_numpy(logits_np).npu()
        target_npu = torch.from_numpy(target_np).npu()
        gmax_npu = torch.from_numpy(global_max_np).npu()

        pred_npu, sum_npu, exp_npu, off_npu, mask_npu = self.npu_op_exec(
            logits_npu, target_npu, gmax_npu, vocab_start, vocab_end)

        # 精度比对 (L2 标准)
        self.assertRtolEqual(pred_cpu, pred_npu.numpy(), prec=1e-5)
        self.assertRtolEqual(sum_cpu, sum_npu.numpy(), prec=1e-5)
        self.assertRtolEqual(exp_cpu, exp_npu.numpy(), prec=1e-5)
        self.assertEqual(off_cpu.tolist(), off_npu.numpy().tolist())
        self.assertEqual(mask_cpu.tolist(), mask_npu.numpy().tolist())

    @SupportedDevices(['Ascend910B', 'Ascend910'])
    def test_npu_cross_entropy_sumexp_and_index_logit_bf16_3d(self):
        """测试 BF16 输入，3 维 [S, B, V_local]"""
        S, B = 32, 16
        V_local = 4096
        vocab_start = 2000
        vocab_end = vocab_start + V_local

        # 生成输入
        np.random.seed(123)
        logits_np = np.random.randn(S, B, V_local).astype(np.float32)
        target_np = np.random.randint(0, 50000, size=(S, B)).astype(np.int32)
        # 部分在范围内
        target_np[:8, :8] = np.random.randint(vocab_start, vocab_end, size=(8, 8))
        global_max_np = np.max(logits_np, axis=-1).astype(np.float32)

        # CPU golden
        pred_cpu, sum_cpu, exp_cpu, off_cpu, mask_cpu = self.cpu_op_exec(
            logits_np, target_np, global_max_np, vocab_start, vocab_end)

        # NPU 执行 (BF16 输入)
        logits_npu = torch.from_numpy(logits_np).to(torch.bfloat16).npu()
        target_npu = torch.from_numpy(target_np).npu()
        gmax_npu = torch.from_numpy(global_max_np).to(torch.bfloat16).npu()

        pred_npu, sum_npu, exp_npu, off_npu, mask_npu = self.npu_op_exec(
            logits_npu, target_npu, gmax_npu, vocab_start, vocab_end)

        # BF16 输入精度放宽
        self.assertRtolEqual(pred_cpu, pred_npu.numpy(), prec=1e-3)
        self.assertRtolEqual(sum_cpu, sum_npu.numpy(), prec=1e-3)
        self.assertRtolEqual(exp_cpu, exp_npu.numpy(), prec=1e-3)
        self.assertEqual(off_cpu.tolist(), off_npu.numpy().tolist())
        self.assertEqual(mask_cpu.tolist(), mask_npu.numpy().tolist())

    @SupportedDevices(['Ascend910B', 'Ascend910'])
    def test_npu_cross_entropy_sumexp_and_index_logit_large_vocab(self):
        """测试大 V_local (37888, 典型网络 shape)"""
        N = 4096
        V_local = 37888
        vocab_start = 10000
        vocab_end = vocab_start + V_local

        np.random.seed(456)
        logits_np = np.random.randn(N, V_local).astype(np.float32) * 1.5
        target_np = np.random.randint(0, 100000, size=(N,)).astype(np.int32)
        target_np[:1024] = np.random.randint(vocab_start, vocab_end, size=(1024,))
        global_max_np = np.max(logits_np, axis=-1).astype(np.float32)

        # CPU golden
        pred_cpu, sum_cpu, exp_cpu, off_cpu, mask_cpu = self.cpu_op_exec(
            logits_np, target_np, global_max_np, vocab_start, vocab_end)

        # NPU
        logits_npu = torch.from_numpy(logits_np).npu()
        target_npu = torch.from_numpy(target_np).npu()
        gmax_npu = torch.from_numpy(global_max_np).npu()

        pred_npu, sum_npu, exp_npu, off_npu, mask_npu = self.npu_op_exec(
            logits_npu, target_npu, gmax_npu, vocab_start, vocab_end)

        self.assertRtolEqual(pred_cpu, pred_npu.numpy(), prec=1e-5)
        self.assertRtolEqual(sum_cpu, sum_npu.numpy(), prec=1e-5)
        self.assertRtolEqual(exp_cpu, exp_npu.numpy(), prec=1e-5)
        self.assertEqual(off_cpu.tolist(), off_npu.numpy().tolist())
        self.assertEqual(mask_cpu.tolist(), mask_npu.numpy().tolist())

    @SupportedDevices(['Ascend910B', 'Ascend910'])
    def test_npu_cross_entropy_sumexp_and_index_logit_boundary(self):
        """测试边界场景：所有 target 越界"""
        N = 64
        V_local = 1024
        vocab_start = 5000
        vocab_end = vocab_start + V_local

        np.random.seed(789)
        logits_np = np.random.randn(N, V_local).astype(np.float32)
        # 所有 target 都不在 [vocab_start, vocab_end) 范围
        target_np = np.random.randint(0, vocab_start, size=(N,)).astype(np.int32)
        global_max_np = np.max(logits_np, axis=-1).astype(np.float32)

        pred_cpu, sum_cpu, exp_cpu, off_cpu, mask_cpu = self.cpu_op_exec(
            logits_np, target_np, global_max_np, vocab_start, vocab_end)

        logits_npu = torch.from_numpy(logits_np).npu()
        target_npu = torch.from_numpy(target_np).npu()
        gmax_npu = torch.from_numpy(global_max_np).npu()

        pred_npu, sum_npu, exp_npu, off_npu, mask_npu = self.npu_op_exec(
            logits_npu, target_npu, gmax_npu, vocab_start, vocab_end)

        # 所有 predicted_logits 应为 0
        self.assertTrue(np.all(pred_npu.numpy() == 0.0))
        # 所有 target_mask 应为 1
        self.assertTrue(np.all(mask_npu.numpy() == 1))
        # 所有 target_offset 应为 0
        self.assertTrue(np.all(off_npu.numpy() == 0))

        self.assertRtolEqual(sum_cpu, sum_npu.numpy(), prec=1e-5)
        self.assertRtolEqual(exp_cpu, exp_npu.numpy(), prec=1e-5)


if __name__ == "__main__":
    run_tests()
