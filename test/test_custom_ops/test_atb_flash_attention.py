# ============================================================================
# 测试文件：Flash Attention 算子测试（普通 Mask 模式）
# 功能：验证 op-plugin 中 _npu_flash_attention 算子在 Ascend 910B 上的正确性
# 测试方式：用 numpy 手写 golden 参考计算，与 NPU 算子输出对比精度
#
# 测试框架：基于 torch_npu.testing.TestCase
#   - TestCase：测试基类，提供 assertRtolEqual 精度对比方法
#   - run_tests：测试运行入口，替代 unittest.main()
#   - SupportedDevices：装饰器，限制测试仅在指定设备上运行
#
# 运行方式：python test/test_custom_ops/test_atb_flash_attention.py
# ============================================================================

import math
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests       # 测试基类 + 运行器
from torch_npu.testing.common_utils import SupportedDevices       # 设备限制装饰器


# ============================================================================
# 测试类：TestFA
# 继承 torch_npu.testing.TestCase，验证 Flash Attention 算子正确性
# 包含三个方法：
#   1. gen_seq_len     - 辅助函数：生成序列长度
#   2. group_matmul    - 辅助函数：GQA 分组矩阵乘法
#   3. calc_expect_func - 核心函数：Golden 参考计算
#   4. test_flash_attention - 测试入口：生成数据 → NPU计算 → 精度对比
# ============================================================================
class TestFA(TestCase):

    # ========================================================================
    # 辅助函数：生成序列长度
    # 功能：为每个 batch 生成实际序列长度和对齐后的序列长度
    # 参数：
    #   batch - 批次数
    #   max_seq - 最大序列长度
    #   variate_seq - 是否使用变长序列（True: 每个 batch 长度不同; False: 固定长度）
    # 返回：
    #   seqlen - 实际序列长度数组 [batch]
    #   seqlen_aligned - 16 对齐后的序列长度数组 [batch]（硬件要求 16 对齐）
    #   ntokens - 所有 batch 的总 token 数
    # ========================================================================
    def gen_seq_len(self, batch, max_seq, variate_seq=False):
        if variate_seq:
            # --- 变长模式 ---
            # 生成不同的序列长度，对齐到 16 的倍数后随机减少 0~15
            num = max_seq // 16
            # 生成对齐后的长度序列 [16, 32, 48, ...]
            seqlen_aligned_arange = np.arange(1, num) * 16
            if batch > num:
                # batch 数大于预设对齐长度数量时，随机补充
                seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
                # 向上对齐到 16 的倍数：(x + 15) // 16 * 16
                seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
            else:
                seqlen_aligned = seqlen_aligned_arange
            # 对齐长度随机减少 0~15，模拟实际长度（不一定是 16 的倍数）
            sp_list = np.random.randint(0, 15, size=(num - 1))
            seqlen = seqlen_aligned - sp_list
            # 取最后 batch 个（防止数组长度超过 batch 数）
            seqlen = seqlen[-batch:]
            seqlen_aligned = seqlen_aligned[-batch:]
        else:
            # --- 固定长度模式 ---
            # 所有 batch 使用相同的 max_seq
            # 向上对齐到 16 的倍数：(max_seq + 15) // 16 * 16
            max_seq_aligned = (max_seq + 15) // 16 * 16
            # 填充量 = 对齐长度 - 实际长度
            sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
            sp_list = sp_list.astype(np.int32)
            # 所有 batch 的实际序列长度 = max_seq
            seqlen = np.ones((batch,)) * max_seq
            seqlen = seqlen.astype(np.int32)
            # 所有 batch 的对齐序列长度 = max_seq_aligned
            seqlen_aligned = np.ones((batch,)) * max_seq_aligned
            seqlen_aligned = seqlen_aligned.astype(np.int32)

        # 总 token 数 = 所有 batch 序列长度之和
        ntokens = seqlen.sum()
        return seqlen, seqlen_aligned, ntokens

    # ========================================================================
    # 辅助函数：分组矩阵乘法（支持 GQA）
    # 功能：实现 GQA 中的分组 QK^T 或 PV 计算
    #       每个 KV head 对应 group_size 个 Q head，共享同一组 K/V
    # 参数：
    #   heads - Q 的 head 总数
    #   group_num - KV 的 head 数（= num_kv_heads）
    #   A - 左矩阵，shape [heads, seq_q, dim]
    #   B - 右矩阵，shape [group_num, dim, seq_kv]
    # 返回：
    #   score - 结果矩阵，shape [heads, seq_q, seq_kv]
    # ========================================================================
    def group_matmul(self, heads, group_num, A, B):
        group_head = heads // group_num  # 每组包含的 Q head 数
        score = None
        for i in range(group_num):
            # 第 i 组的 Q head 范围：[i*group_head, (i+1)*group_head)
            # 第 i 组的 KV head：固定为第 i 个
            # 转 float32 计算提高精度，结果转回 float16
            group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                    B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                # 按第 0 维（head 维度）拼接各组结果
                score = np.concatenate((score, group_score), 0)
        return score

    # ============================================================================
    # 核心函数：Golden 参考计算（普通 Mask 版本）
    # 功能：用 numpy 手写完整的 Self-Attention 计算流程，作为正确性基准
    #       支持 GQA、普通 float16 mask（上三角加 -10000）
    # 计算流程：QK^T → Scale → Mask → Softmax → PV
    #
    # 参数：
    #   batch     - 批次数
    #   seqlen    - 序列长度
    #   heads     - Query head 数 (num_heads)
    #   embed     - Q/K 的 head 维度 (head_dim)
    #   group_num - KV head 数 (num_kv_heads)，GQA 时 < heads
    #
    # 返回：
    #   (q, k, v, mask, q_len, tor, heads, out_expect) 元组
    #   索引： 0    1    2     3      4      5     6       7
    # ============================================================================
    def calc_expect_func(self, batch, seqlen, heads, embed, group_num=32):
        # --- 配置参数 ---
        is_mask = True         # 是否使用因果 mask（上三角遮蔽）
        variate_seq = False    # 是否使用变长序列（False=所有 batch 相同长度）
        is_decoder = False     # 是否为解码器模式（False=编码器，Q/KV 长度相同）
        max_seq = 2048         # mask 矩阵的最大尺寸
        src_type = 'float16'   # 输入输出数据类型
        fp32 = True            # softmax 是否在 float32 下计算（True=精度更高）

        # --- Step 1: 生成序列长度 ---
        # 编码器模式：Q 和 KV 序列长度相同
        # 解码器模式：Q 长度为 1（逐 token 生成），KV 长度为 seqlen
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens

        max_s = np.max(q_seqlen)                   # 所有 batch 中的最大序列长度
        ntokens2 = (q_seqlen * kv_seqlen).sum()     # Q*KV 总元素数（未使用，仅调试）
        embed_v = np.random.randint(1, embed)       # Value 的 head 维度，随机生成（可以不等于 embed）

        # --- Step 2: 生成随机输入数据 ---
        # 布局为 TND（total_tokens, num_heads, head_dim），先以 2D 生成再 reshape
        # Q: [total_tokens, heads * embed]，后续 reshape 为 [total_tokens, heads, embed]
        # K: [total_tokens, group_num * embed]
        # V: [total_tokens, group_num * embed_v]，注意 V 的维度可以与 Q/K 不同
        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v)).astype(np.float16)

        # --- Step 3: 生成因果 mask ---
        # 普通模式：float16 格式，上三角为 -10000（softmax 后趋近 0），其余为 0
        # np.triu(matrix, k=1)：取上三角（不含对角线），k=1 表示从主对角线上一条开始
        # 与压缩 mask（int8 格式）不同，这里直接用加法 score = score + mask
        mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)
        mask = np.triu(mask, 1)
        mask *= -10000.0

        # --- 偏移量初始化 ---
        # 用于在展平的 TND 数据中定位每个 batch 的数据起始位置
        q_offset = 0
        k_offset = 0
        v_offset = 0

        # --- 累积变量 ---
        s = None       # 累积的注意力分数（未使用，仅调试）
        _p = None      # 累积的 softmax exp 值（未使用，仅调试）
        out = None     # 累积的输出结果

        # --- Step 4: 逐 batch 计算 Self-Attention ---
        for idx in range(batch):
            q_s = q_seqlen[idx]    # 当前 batch 的 Q 序列长度
            kv_s = kv_seqlen[idx]  # 当前 batch 的 KV 序列长度

            # 切取当前 batch 的 Q 切片
            # reshape: [q_s, heads*embed] → [q_s, heads, embed]
            # transpose: [q_s, heads, embed] → [heads, q_s, embed]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))

            # 切取当前 batch 的 K 切片
            # reshape: [kv_s, group_num*embed] → [kv_s, group_num, embed]
            # transpose: [kv_s, group_num, embed] → [group_num, kv_s, embed]
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            # K^T: [group_num, kv_s, embed] → [group_num, embed, kv_s]
            k_slice_t = np.transpose(k_slice, (0, 2, 1))

            # 切取当前 batch 的 V 切片
            # reshape: [kv_s, group_num*embed_v] → [kv_s, group_num, embed_v]
            # transpose: [kv_s, group_num, embed_v] → [group_num, kv_s, embed_v]
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed_v)
            v_slice = np.transpose(v_slice, (1, 0, 2))

            # --- Step 4a: QK^T 注意力分数 ---
            # 使用 group_matmul 支持 GQA（每组 Q head 共享同一个 KV head）
            # score: [heads, q_s, kv_s]
            score = self.group_matmul(heads, group_num, q_slice, k_slice_t)
            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            # --- Step 4b: Scale 缩放 ---
            # 缩放因子 = 1/√d_k，防止点积值过大导致 softmax 梯度消失
            tor = np.float16(1.0 / math.sqrt(1.0 * embed))
            score = score * tor

            # --- Step 4c: Apply Mask（因果遮蔽）---
            # 普通模式：score = score + mask
            # 上三角位置加 -10000，softmax 后趋近 0，实现"看不到未来"的因果约束
            # mask 形状: [1, max_s, max_s] → 切片取 [1, q_s, kv_s] →广播到 [heads, q_s, kv_s]
            if is_mask:
                score = score + mask[:, :q_s, :kv_s]

            # --- Step 4d: Softmax（数值稳定版）---
            # 1. 减最大值（防止 exp 溢出）
            # 2. 取 exp
            # 3. 除以 sum（归一化）
            score_max = np.max(score, axis=-1)                          # 每行最大值
            score = score - score_max.reshape((heads, q_s, 1))         # 减最大值
            score_exp = np.exp(score.astype(np.float32))                # 转 float32 计算 exp

            if not fp32:
                # float16 模式：直接在 float16 下计算 softmax（精度较低）
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = self.group_matmul(heads, group_num, p, v_slice)
            else:
                # float32 模式：exp 在 float32 下计算，归一化后再转回 float16（精度更高）
                # 注意：这里先乘 V 再除 sum，等价于 softmax(V)，但避免了先除再乘的精度损失
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)                        # 先转 float16
                out_sub = self.group_matmul(heads, group_num, p, v_slice)  # 先乘 V
                out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)  # 再除 sum

            # --- Step 4e: 整理输出形状 ---
            # [heads, q_s, embed_v] → [q_s, heads, embed_v]
            out_sub = out_sub.reshape(heads, q_s, embed_v)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)  # 确保内存连续

            # 拼接到总输出（按 batch 维度拼接）
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            # 更新偏移量，移到下一个 batch
            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        # --- Step 5: 整理最终输出格式 ---
        # reshape 为 TND 布局 [total_tokens, num_heads, head_dim]
        # mask 从 [1, max_s, max_s] reshape 为 [max_s, max_s]（去掉 batch 维度）
        q = q.astype(src_type).reshape(-1, heads, embed)
        k = k.astype(src_type).reshape(-1, group_num, embed)
        v = v.astype(src_type).reshape(-1, group_num, embed_v)
        mask = mask.astype(src_type).reshape(max_s, max_s)
        q_len = q_seqlen.astype(np.int32)
        out_expect = out.astype(src_type).reshape(-1, heads, embed_v)

        # 返回元组，索引对应：
        # 0: q          - Query 张量 [total_tokens, heads, embed]
        # 1: k          - Key 张量 [total_tokens, group_num, embed]
        # 2: v          - Value 张量 [total_tokens, group_num, embed_v]
        # 3: mask       - 因果掩码 [max_s, max_s]
        # 4: q_len      - 各 batch 序列长度 [batch]
        # 5: tor        - 缩放因子 (1/√embed)
        # 6: heads      - Query 头数
        # 7: out_expect - Golden 参考输出 [total_tokens, heads, embed_v]
        ret_data = q, k, v, mask, q_len, tor, heads, out_expect
        return ret_data

    # ========================================================================
    # 测试入口：验证 _npu_flash_attention 算子的正确性
    # 流程：
    #   1. 用 calc_expect_func 生成 golden 数据（numpy 计算）
    #   2. numpy → torch.Tensor → .npu() 移到 NPU
    #   3. 调用 NPU 算子 torch_npu._npu_flash_attention()
    #   4. assertRtolEqual 对比 NPU 输出 vs golden 输出
    # ========================================================================
    @SupportedDevices(['Ascend910B'])  # 装饰器：仅在 Ascend 910B 设备上运行此测试
    def test_flash_attention(self):
        # --- Step 1: 生成 golden 数据 ---
        # 参数：batch=16, seqlen=128, heads=32, embed=128, group_num=32（MHA 模式）
        kv_head = 32
        data = self.calc_expect_func(16, 128, 32, 128, group_num=kv_head)
        # data = (q, k, v, mask, q_len, tor, heads, out_expect)
        # 索引：  0   1   2    3      4      5     6        7

        # param_seqlen: 序列长度列表（本测试未使用，预留接口）
        param_seqlen = data[4].tolist()

        # --- Step 2: numpy → torch.Tensor → NPU ---
        in_tensors = []
        for tensor in data:
            if isinstance(tensor, np.ndarray):
                # numpy 数组 → torch.Tensor（共享内存，零拷贝）
                in_tensors.append(torch.from_numpy(tensor))
            else:
                # 标量值（如 tor, heads）→ torch.Tensor
                in_tensors.append(torch.tensor(tensor))
        # 所有张量移到 NPU 设备
        in_tensors = [tensor.npu() for tensor in in_tensors]

        # --- Step 3: 解包输入参数 ---
        query = in_tensors[0]          # Q 张量 [total_tokens, heads, embed]
        key = in_tensors[1]            # K 张量 [total_tokens, group_num, embed]
        value = in_tensors[2]          # V 张量 [total_tokens, group_num, embed_v]
        mask = in_tensors[3]           # 因果掩码 [max_s, max_s]
        seq_len = in_tensors[4].cpu()  # 序列长度，需在 CPU 上（NPU 算子要求）
        tor = data[5]                  # 缩放因子 (float16 标量)
        heads = data[6]                # Query 头数 (int 标量)
        group_num = kv_head            # KV 头数 = 32（MHA 模式）
        cal_out = in_tensors[7]        # golden 参考输出（用于精度对比）

        # --- Step 4: 分配输出张量并调用 NPU 算子 ---
        # torch.empty_like：创建与 golden 输出形状/类型相同的空张量（不初始化）
        # .npu()：移到 NPU 设备
        # 注意：out 参数是原地写入模式，NPU 算子直接将结果写入 out
        out = torch.empty_like(in_tensors[7]).npu()
        torch_npu._npu_flash_attention(query, key, value, mask, seq_len,
                                       tor, heads, group_num, out)

        # --- Step 5: 精度对比 ---
        # assertRtolEqual：TestCase 基类提供的方法
        # 对比 NPU 算子输出 (out) 与 golden 参考输出 (cal_out)
        # 内部使用 torch.allclose() 判断相对误差是否在允许范围内
        self.assertRtolEqual(cal_out, out)


# ============================================================================
# 测试入口
# run_tests()：torch_npu 提供的测试运行器，替代 unittest.main()
# 自动发现并运行类中所有 test_ 开头的方法
# ============================================================================
if __name__ == '__main__':
    run_tests()
