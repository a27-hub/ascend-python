import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu

import atb_cpp_interface_extension as atb_ext


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cpu_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_value(item) for item in value)
    return value


def summarize_tensor(tensor):
    cpu = tensor.detach().cpu()
    if cpu.numel() <= 16:
        return str(cpu)
    return f"Tensor(shape={tuple(cpu.shape)}, dtype={cpu.dtype}, device={tensor.device})"


def summarize_value(value):
    if isinstance(value, torch.Tensor):
        return summarize_tensor(value)
    if isinstance(value, (tuple, list)):
        return type(value).__name__ + "(" + ", ".join(summarize_value(item) for item in value) + ")"
    return repr(value)


def tensor_close(actual, expected, *, rtol=1e-3, atol=1e-3):
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if actual_cpu.dtype != expected_cpu.dtype:
        actual_cpu = actual_cpu.to(expected_cpu.dtype)
    if actual_cpu.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ):
        return torch.equal(actual_cpu, expected_cpu)
    return torch.allclose(actual_cpu.float(), expected_cpu.float(), rtol=rtol, atol=atol)


def compare_value(actual, expected, *, rtol=1e-3, atol=1e-3):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return tensor_close(actual, expected, rtol=rtol, atol=atol)
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(
            compare_value(a, e, rtol=rtol, atol=atol) for a, e in zip(actual, expected)
        )
    return actual == expected


def compare_workspace(actual, expected):
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    return (
        actual_cpu.dtype == expected_cpu.dtype
        and actual.device == expected.device
        and tuple(actual_cpu.shape) == tuple(expected_cpu.shape)
        and actual_cpu.numel() == expected_cpu.numel()
    )


def print_case_result(name, passed, baseline, actual):
    print(f"[CASE] {name}")
    print("  python_result:", summarize_value(baseline))
    print("  cpp_wrapper_result:", summarize_value(actual))
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")


def run_value_case(name, python_fn, cpp_fn, *, rtol=1e-3, atol=1e-3):
    baseline = python_fn()
    actual = cpp_fn()
    passed = compare_value(actual, baseline, rtol=rtol, atol=atol)
    print_case_result(name, passed, baseline, actual)
    return passed


def run_workspace_case(name, workspace_python_fn, workspace_cpp_fn, paired_python_fn, paired_cpp_fn, *, rtol=1e-3, atol=1e-3):
    baseline_workspace = workspace_python_fn()
    actual_workspace = workspace_cpp_fn()
    workspace_ok = compare_workspace(actual_workspace, baseline_workspace)
    print(f"[CASE] {name}")
    print("  python_workspace:", summarize_value(baseline_workspace))
    print("  cpp_workspace:", summarize_value(actual_workspace))
    if not workspace_ok:
        print(f"  FAIL: {name} (workspace metadata mismatch)")
        return False

    baseline = paired_python_fn(baseline_workspace)
    actual = paired_cpp_fn(actual_workspace)
    passed = compare_value(actual, baseline, rtol=rtol, atol=atol)
    print("  python_result:", summarize_value(baseline))
    print("  cpp_wrapper_result:", summarize_value(actual))
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    return passed


def build_quantize_inputs():
    input_x = torch.tensor(
        [
            -2.0, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0,
            -1.5, -0.25, 0.25, 0.75,
            1.25, 1.75, 2.25, -2.25,
        ],
        dtype=torch.float16,
    ).view(1, 4, 4).npu()
    scale = torch.full((4, 4), 0.5, dtype=torch.float16).npu()
    offset = torch.ones((4, 4), dtype=torch.int8).npu()
    return input_x, scale, offset


def build_quant_rms_norm_inputs():
    torch.manual_seed(12)
    x = torch.randn(4, 1, 128).half().npu()
    gamma = torch.randn(1, 128).half().npu()
    beta = torch.randn(1, 128).half().npu()
    scale = torch.tensor([0.3], dtype=torch.float16).npu()
    offset = torch.tensor([2], dtype=torch.int8).npu()
    eps = 1e-5
    return x, gamma, beta, scale, offset, eps


def build_group_topk_inputs():
    torch.manual_seed(1)
    scores = torch.empty((4, 16), dtype=torch.float16, device="npu").uniform_(-2, 2)
    return scores, 2, 4, 2


def build_matmul_add_fp32_inputs():
    torch.manual_seed(2)
    x = torch.rand((16, 32), dtype=torch.float16).npu()
    weight = torch.rand((16, 32), dtype=torch.float16).npu()
    c = torch.rand((32, 32), dtype=torch.float32).npu()
    return x, weight, c


def build_fused_add_topk_div_inputs():
    torch.manual_seed(3)
    x = torch.from_numpy(np.random.uniform(1, 10, (16, 256))).to(torch.float16).npu()
    add_num = torch.from_numpy(np.random.uniform(1, 10, 256)).to(torch.float16).npu()
    mapping_num = torch.ones(256, dtype=torch.int32).npu() * 2
    mapping_table = torch.randint(0, 256, (256, 2), dtype=torch.int32).npu()
    kwargs = dict(
        mapping_num=mapping_num,
        mapping_table=mapping_table,
        activation_type="activation_sigmoid",
        group_num=8,
        group_topk=4,
        n=2,
        k=8,
        is_norm=True,
        scale=1.0,
        enable_expert_mapping=True,
    )
    return x, add_num, kwargs


def build_mla_preprocess_inputs():
    token_num = 1
    head_num = 128
    n_7168 = 7168
    block_num = 192
    block_size = 128
    dtype = torch.bfloat16
    device = "npu:0"

    input1 = torch.randn((token_num, n_7168), dtype=dtype, device=device)
    gamma0 = torch.randn((n_7168,), dtype=dtype, device=device)
    beta0 = torch.randn((n_7168,), dtype=dtype, device=device)
    quant_scale0 = torch.randn((1,), dtype=dtype, device=device)
    quant_offset0 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

    wdqkv = torch.randint(0, 7, (1, 224, 2112, 32), dtype=torch.int8, device=device)
    wdqkv = torch_npu.npu_format_cast(wdqkv, 29)
    de_scale0 = torch.rand((2112,), dtype=torch.float, device=device)
    bias0 = torch.randint(0, 7, (2112,), dtype=torch.int32, device=device)

    gamma1 = torch.randn((1536,), dtype=dtype, device=device)
    beta1 = torch.randn((1536,), dtype=dtype, device=device)
    quant_scale1 = torch.randn((1,), dtype=dtype, device=device)
    quant_offset1 = torch.randint(0, 7, (1,), dtype=torch.int8, device=device)

    wuq = torch.randint(0, 7, (1, 48, head_num * 192, 32), dtype=torch.int8, device=device)
    wuq = torch_npu.npu_format_cast(wuq, 29)
    de_scale1 = torch.rand((head_num * 192,), dtype=torch.float, device=device)
    bias1 = torch.randint(0, 7, (head_num * 192,), dtype=torch.int32, device=device)

    gamma2 = torch.randn((512,), dtype=dtype, device=device)
    cos = torch.randn((token_num, 64), dtype=dtype, device=device)
    sin = torch.randn((token_num, 64), dtype=dtype, device=device)
    wuk = torch.randn((head_num, 128, 512), dtype=dtype, device=device)
    wuk = torch_npu.npu_format_cast(wuk, 29)

    kv_cache = torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device)
    kv_cache = torch_npu.npu_format_cast(kv_cache, 29)
    kv_cache_rope = torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device)
    kv_cache_rope = torch_npu.npu_format_cast(kv_cache_rope, 29)
    slotmapping = torch.randint(0, 7, (token_num,), dtype=torch.int32, device=device)
    ctkv_scale = torch.randn((1,), dtype=dtype, device=device)
    qnope_scale = torch.randn((head_num,), dtype=dtype, device=device)

    kwargs = dict(
        quant_scale0=quant_scale0,
        quant_offset0=quant_offset0,
        bias0=bias0,
        quant_scale1=quant_scale1,
        quant_offset1=quant_offset1,
        bias1=bias1,
        ctkv_scale=ctkv_scale,
        q_nope_scale=qnope_scale,
        cache_mode="int8_nzcache",
        quant_mode="per_tensor_quant_asymm",
    )
    outputs = dict(
        q_out0=torch.randint(0, 7, (token_num, head_num, 512), dtype=torch.int8, device=device),
        kv_cache_out0=torch_npu.npu_format_cast(
            torch.randint(0, 7, (block_num, head_num * 512 // 32, block_size, 32), dtype=torch.int8, device=device), 29),
        q_out1=torch.randn((token_num, head_num, 64), dtype=dtype, device=device),
        kv_cache_out1=torch_npu.npu_format_cast(
            torch.randn((block_num, head_num * 64 // 16, block_size, 16), dtype=dtype, device=device), 29),
    )
    return (
        input1,
        gamma0,
        beta0,
        wdqkv,
        de_scale0,
        gamma1,
        beta1,
        wuq,
        de_scale1,
        gamma2,
        cos,
        sin,
        wuk,
        kv_cache,
        kv_cache_rope,
        slotmapping,
        kwargs,
        outputs,
    )


def build_multi_head_latent_attention_inputs():
    from test.test_custom_ops.test_atb_mla_with_lse import TestAtbAttention

    helper = TestAtbAttention(methodName="runTest")
    helper.is_ring = 0
    helper.calc_data(
        32,
        32,
        1,
        576,
        512,
        128,
        64,
        256,
        torch.float16,
        0,
        torch.float16,
        is_kv_combined=True,
        is_nz_in=False,
    )
    common = dict(
        q_nope=helper.q_split1.npu(),
        q_rope=helper.q_split2.npu(),
        ctkv=helper.key_cache_split1.npu(),
        k_rope=helper.key_cache_split2.npu(),
        block_tables=torch.tensor(helper.block_tables).int().npu(),
        context_lens=[int(x) for x in helper.contex_lens.tolist()],
        head_num=32,
        qk_scale=1.0 / math.sqrt(576.0),
        kv_headnum=1,
        mask=None,
        qseqlen=None,
        qk_descale=None,
        pv_descale=None,
        mask_type=None,
        calc_type="calc_type_ring",
        cache_mode=None,
    )
    return common


def build_paged_attention_inputs():
    from test.test_custom_ops.test_atb_paged_attention import TestPagedAttention

    helper = TestPagedAttention(methodName="runTest")
    query_np, key_cache_np, value_cache_np, block_tables_np, context_lens_np = helper.prepare_inputs()
    return dict(
        query=torch.from_numpy(query_np).npu(),
        key_cache=torch.from_numpy(key_cache_np).npu(),
        value_cache=torch.from_numpy(value_cache_np).npu(),
        num_kv_heads=1,
        num_heads=2,
        scale_value=1.0 / math.sqrt(128.0),
        block_table=torch.from_numpy(block_tables_np).npu(),
        context_lens=torch.from_numpy(context_lens_np),
    )


def build_paged_attention_v2_inputs():
    from test.test_custom_ops.test_atb_paged_attention_v2 import TestPagedAttentionAlibi

    helper = TestPagedAttentionAlibi(methodName="runTest")
    q, key_cache, value_cache, block_tables, contex_lens, alib_mask, golden_out = helper.calc_data(
        1, 32, 32, 288, 128, 64, 128, "float16", 4, torch.float16
    )
    return dict(
        query=q.npu(),
        key_cache=key_cache.npu(),
        block_table=torch.from_numpy(block_tables).npu(),
        context_lens=[int(x) for x in contex_lens.tolist()],
        value_cache=value_cache.npu(),
        mask=alib_mask.npu(),
        num_kv_heads=32,
        num_heads=32,
        scale_value=1.0 / math.sqrt(288.0),
        mask_type=2,
    )


def build_paged_attention_mla_inputs():
    from test.test_custom_ops.test_atb_paged_attention_mla import TestPagedAttention

    helper = TestPagedAttention(methodName="runTest")
    query, key_cache, value_cache, block_tables, context_lens = helper.init_data()
    return dict(
        query=query.npu(),
        key_cache=key_cache.npu(),
        num_kv_heads=1,
        num_heads=4,
        scale_value=1.0 / math.sqrt(576.0),
        block_table=torch.tensor(block_tables, dtype=torch.int32).npu(),
        context_lens=torch.tensor(context_lens, dtype=torch.int32),
        mla_vheadsize=512,
    )


def build_paged_attention_quant_inputs():
    from test.test_custom_ops.test_atb_paged_attention_quant import TestQuantRmsNorm

    helper = TestQuantRmsNorm(methodName="runTest")
    helper.calc_data(96, 8, 1, 128, 128, 480, 7, torch.bfloat16, 0, torch.bfloat16, False, 7, is_quant_flag=1)
    return dict(
        query=helper.q.npu(),
        key_cache=helper.key_cache.npu(),
        value_cache=helper.value_cache.npu(),
        num_kv_heads=1,
        num_heads=8,
        scale_value=1.0 / math.sqrt(128.0),
        block_table=torch.from_numpy(helper.block_tables).npu(),
        context_lens=torch.from_numpy(helper.contex_lens),
        quant_type=3,
        outdata_type=27,
        k_descale=helper.de_scale1_fp32.npu(),
        v_descale=helper.de_scale2_fp32.npu(),
    )


def build_paged_attention_splitfuse_inputs():
    from test.test_custom_ops.test_atb_paged_attention_splitfuse import GenerateDataParams, TestFA

    helper = TestFA(methodName="runTest")
    data = helper.generate_data(GenerateDataParams())
    return dict(
        query=data.query.npu(),
        key_cache=data.key_cache.npu(),
        value_cache=data.value_cache.npu(),
        block_table=data.block_tables.npu(),
        context_lens=data.seq_lens,
        mask=helper.make_attention_mask(data.max_seq_len, data.seq_lens, data.query_lens).to(torch.bfloat16).npu(),
        seq_len=data.query_lens,
        num_kv_heads=32,
        num_heads=32,
        scale_value=128 ** -0.5,
    )


def build_paged_cache_load_inputs():
    from test.test_custom_ops.test_atb_paged_cache_load import TestPagedCacheLoadParams, TestPagedCacheLoadSeqStarts

    helper = TestPagedCacheLoadSeqStarts(methodName="runTest")
    prepared = helper._prepare_data(
        TestPagedCacheLoadParams(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_size=16,
            num_blocks=1024,
            max_seq_len=512,
            batch_size=8,
            device="npu",
        )
    )
    return dict(
        key_cache=prepared.cached_kv_c,
        value_cache=prepared.cached_k_pe,
        block_table=prepared.block_table,
        context_lens=prepared.seq_len_tensor.int(),
        seq_starts=prepared.seq_starts,
        cumsum=False,
    )


def build_rotary_embedding_inputs():
    from test.test_custom_ops.test_atb_rope import TestRope

    helper = TestRope(methodName="runTest")
    cos_sin_cache = helper.compute_cos_sin_cache()
    positions = torch.arange(helper.seq_length).repeat(helper.batch_size).npu()
    query = torch.rand(helper.batch_size * helper.seq_length, helper.num_heads * helper.head_dim, dtype=torch.float16).npu()
    key = torch.rand(helper.batch_size * helper.seq_length, helper.num_heads * helper.head_dim, dtype=torch.float16).npu()
    return positions, query, key, helper.head_dim, cos_sin_cache, False


def build_flash_attention_inputs():
    from test.test_custom_ops.test_atb_flash_attention import TestFA

    helper = TestFA(methodName="runTest")
    data = helper.calc_expect_func(16, 128, 32, 128, group_num=32)
    tensors = [torch.from_numpy(item) if isinstance(item, np.ndarray) else torch.tensor(item) for item in data]
    tensors = [tensor.npu() for tensor in tensors]
    return dict(
        query=tensors[0],
        key=tensors[1],
        value=tensors[2],
        mask=tensors[3],
        seq_len=tensors[4].cpu(),
        scale_value=float(data[5]),
        num_heads=int(data[6]),
        num_kv_heads=32,
    )


def build_flash_attention_v2_inputs():
    from test.test_custom_ops.test_atb_flash_attention_v2 import SelfAttentionV2Test

    helper = SelfAttentionV2Test(methodName="runTest")
    data = helper.t_flash_attention_case_fa_encoder_nocache_bf16_alibi()
    seq_len = [int(x) for x in data[4]]
    return dict(
        query=data[0].npu().contiguous(),
        key=data[1].npu().contiguous(),
        value=data[2].npu().contiguous(),
        seq_len=seq_len,
        mask=data[3].npu().contiguous(),
        slopes=None,
        kernel_type=0,
        mask_type=2,
        scale_value=1.0,
        num_heads=12,
        num_kv_heads=1,
    )


def build_flash_attention_prefix_v2_inputs():
    from test.test_custom_ops.test_atb_flash_attention_prefix_v2 import TestFlashAttentionPrefixEncoder

    helper = TestFlashAttentionPrefixEncoder(methodName="runTest")
    helper.set_data_params(
        dynamic_batch=False,
        is_decoder=0,
        batch=1,
        kv_head=1,
        heads=12,
        embeddim=128,
        max_seq=4096,
        kv_seqLen=[512],
        data_type=torch.bfloat16,
        is_alibi=True,
        op_type=10,
        mask_type=5,
        no_cache=True,
        is_sqrt=True,
        tor=1.0 / math.sqrt(128.0),
        q_seqlens=[128],
        num_blocks=1024,
        block_size=128,
    )
    helper.gen_out_tensor()
    helper.alibi_slopes *= -1
    mask = np.ones((256, 256)) * float("inf")
    mask = np.triu(mask, 1)
    helper.mask = helper.bias[:256, :256] * -1 + mask
    helper.mask = helper.mask.to(torch.bfloat16)
    q_seqlen = [128]
    kv_seq_len = [512]
    q = helper.q.view(sum(q_seqlen), 12, 128).npu()
    k_cache = helper.k_cache.view(1024, 128, 128).npu()
    v_cache = helper.v_cache.view(1024, 128, 128).npu()
    return dict(
        query=q,
        key_cache=k_cache,
        value_cache=v_cache,
        block_table=helper.block_tables.npu(),
        mask=helper.mask.npu(),
        seq_len=q_seqlen,
        context_lens=kv_seq_len,
        slopes=helper.alibi_slopes.to(torch.float32).npu(),
        kernel_type=1,
        mask_type=5,
        num_kv_heads=1,
        num_heads=12,
        scale_value=1.0 / math.sqrt(128.0),
    )


def build_self_attention_prefix_encoder_inputs():
    from test.test_custom_ops.test_atb_selfattention_prefix_encoder import TestFlashAttentionPrefixEncoder

    helper = TestFlashAttentionPrefixEncoder(methodName="runTest")
    q_seqlens = [128 * 26 + 13] * 2
    kv_seq_lens = [128 * 28 + 13] * 2
    helper.set_data_params(
        dynamic_batch=False,
        is_decoder=0,
        batch=2,
        kv_head=2,
        heads=4,
        embeddim=128,
        max_seq=128 * 100,
        kv_seqLen=kv_seq_lens,
        is_clamp=0,
        clamp_max=0,
        clamp_min=0,
        data_type=torch.bfloat16,
        op_type=10,
        mask_type=3,
        no_cache=True,
        tor=1.0 / math.sqrt(128.0),
        q_seqlens=q_seqlens,
        num_blocks=1024,
        block_size=128,
        is_triu_mask=True,
        is_mask=True,
    )
    helper.gen_out_tensor()
    return dict(
        query=helper.q.npu(),
        key=helper.k_cache.npu(),
        value=helper.v_cache.npu(),
        block_tables=helper.block_tables.npu(),
        seqlen=q_seqlens,
        kv_seqlen=kv_seq_lens,
        q_headnum=4,
        qk_scale=1.0 / math.sqrt(128.0),
        kv_headnum=2,
        mask=None,
        slopes=None,
        mask_type=None,
    )


def build_ring_mla_inputs(is_ring):
    from test.test_custom_ops.test_atb_ring_mla import MASK_TYPE_NO_MASK, TestAtbRingMla

    helper = TestAtbRingMla(methodName="runTest")
    kv_seq_lens = [200] * 2
    heads = 16
    kv_head = 16
    helper.set_data_params(
        dynamic_batch=False,
        is_decoder=0,
        batch=2,
        kv_head=kv_head,
        heads=heads,
        embeddim=192,
        embeddimv=128,
        max_seq=200,
        kv_seqLen=kv_seq_lens,
        is_clamp=0,
        clamp_max=0,
        clamp_min=0,
        data_type=torch.float16,
        is_alibi=False,
        is_triu_mask=False,
        op_type=1,
        mask_type=MASK_TYPE_NO_MASK,
        tor=1.0 / math.sqrt(192.0),
        long_seq=False,
        lse=(torch.rand((sum(kv_seq_lens), heads)) * 10).to(torch.float32) if is_ring else torch.zeros((sum(kv_seq_lens), heads), dtype=torch.float32),
        last_o=torch.rand((sum(kv_seq_lens), heads, 128)).to(torch.float16) if is_ring else torch.zeros((sum(kv_seq_lens), heads, 128), dtype=torch.float16),
        isring=1 if is_ring else 0,
    )
    helper.gen_out_tensor()
    q_seqlen = torch.from_numpy(np.array(helper.q_seqlen)).to(torch.int32)
    kv_seq_tensor = torch.from_numpy(np.array(helper.kv_seqLen)).to(torch.int32)
    seqlen = torch.stack([q_seqlen, kv_seq_tensor]).npu()
    pre_out = helper.last_o.reshape(sum(helper.q_seqlen), heads, 128).npu() if is_ring else None
    prev_lse = helper.input_lse.transpose(1, 0).reshape(heads, sum(helper.q_seqlen)).npu() if is_ring else None
    return dict(
        q_nope=helper.q_split1.reshape(sum(helper.q_seqlen), heads, 128).npu(),
        q_rope=helper.q_split2.reshape(sum(helper.q_seqlen), heads, 64).npu(),
        k_nope=helper.k_split1.reshape(sum(helper.kv_seqLen), kv_head, 128).npu(),
        k_rope=helper.k_split2.reshape(sum(helper.kv_seqLen), kv_head, 64).npu(),
        value=helper.v.reshape(sum(helper.kv_seqLen), kv_head, 128).npu(),
        mask=helper.mask.to(torch.float16).reshape(200, 200).npu(),
        seqlen=seqlen,
        head_num=heads,
        kv_head_num=kv_head,
        pre_out=pre_out,
        prev_lse=prev_lse,
        qk_scale=1.0 / math.sqrt(192.0),
        kernel_type="kernel_type_high_precision",
        mask_type="no_mask",
        input_layout="type_bsnd",
        calc_type="calc_type_default" if is_ring else "calc_type_first_ring",
    )


def build_flash_attention_qlens_inputs():
    from test.test_custom_ops.test_atb_flash_attention_qlens import GenerateDataParams, TestFA

    helper = TestFA(methodName="runTest")
    result = helper.generate_data(GenerateDataParams())
    mask_compress = torch.ones(size=(128, 128), dtype=torch.bfloat16)
    mask_compress = torch.triu(mask_compress, 1)
    mask_compress *= -10000.0
    return dict(
        query=result.query.npu(),
        key_cache=result.key_cache.npu(),
        value_cache=result.value_cache.npu(),
        block_table=result.block_tables.npu(),
        mask=mask_compress.npu(),
        seq_len=result.query_lens,
        context_lens=result.seq_lens,
        num_kv_heads=32,
        num_heads=32,
        scale_value=128 ** -0.5,
    )


def build_flash_attention_unpad_inputs():
    from test.test_custom_ops.test_atb_flash_attention_unpad import TestFaUnPad

    helper = TestFaUnPad(methodName="runTest")
    data = helper.calc_expect_func(16, 128, 32, 128, group_num=32)
    tensors = [torch.from_numpy(item) if isinstance(item, np.ndarray) else torch.tensor(item) for item in data]
    tensors = [tensor.npu() for tensor in tensors]
    return dict(
        query=tensors[0],
        key=tensors[1],
        value=tensors[2],
        seq_len=tensors[3].cpu(),
        scale_value=float(data[4]),
        num_heads=int(data[5]),
        num_kv_heads=32,
    )


def case_quantize_per_tensor():
    x, scale, offset = build_quantize_inputs()
    return run_value_case(
        "_npu_quantize_per_tensor",
        lambda: (lambda y: (torch.ops.atb._npu_quantize_per_tensor(x, scale, offset, y), y)[1])(torch.zeros_like(x, dtype=torch.int8)),
        lambda: atb_ext._npu_quantize_per_tensor(x, scale, offset),
        atol=0.0,
        rtol=0.0,
    )


def case_quant_rms_norm():
    x, gamma, beta, scale, offset, eps = build_quant_rms_norm_inputs()
    return run_value_case(
        "_npu_quant_rms_norm",
        lambda: (lambda out: (torch.ops.atb._npu_quant_rms_norm(x, gamma, beta, scale, offset, out, eps), out)[1])(torch.zeros_like(x, dtype=torch.int8)),
        lambda: atb_ext._npu_quant_rms_norm(x, gamma, beta, scale, offset, eps),
        atol=0.0,
        rtol=0.0,
    )


def case_group_topk():
    scores, k, group_num, inner_n = build_group_topk_inputs()
    return run_value_case(
        "_npu_group_topk",
        lambda: (lambda out: (torch.ops.atb._npu_group_topk(out, k, group_num, inner_n), out)[1])(scores.clone()),
        lambda: atb_ext._npu_group_topk(scores, k, group_num, inner_n),
    )


def case_matmul_add_fp32():
    x, weight, c = build_matmul_add_fp32_inputs()
    return run_value_case(
        "_npu_matmul_add_fp32",
        lambda: (lambda out: (torch.ops.atb._npu_matmul_add_fp32(x, weight, out), out)[1])(c.clone()),
        lambda: atb_ext._npu_matmul_add_fp32(x, weight, c),
        rtol=1e-3,
        atol=1e-3,
    )


def case_fused_add_topk_div():
    x, add_num, kwargs = build_fused_add_topk_div_inputs()
    return run_value_case(
        "npu_fused_add_topk_div",
        lambda: torch.ops.atb.npu_fused_add_topk_div(x, add_num, **kwargs),
        lambda: atb_ext.npu_fused_add_topk_div(x, add_num, **kwargs),
    )


def case_fused_add_topk_div_out():
    x, add_num, kwargs = build_fused_add_topk_div_inputs()
    y0 = torch.empty((16, 8), dtype=torch.float32).npu()
    i0 = torch.empty((16, 8), dtype=torch.int32).npu()
    y1 = torch.empty_like(y0)
    i1 = torch.empty_like(i0)
    return run_value_case(
        "npu_fused_add_topk_div_out",
        lambda: torch.ops.atb.npu_fused_add_topk_div.out(x, add_num, **kwargs, y=y0, indices=i0),
        lambda: atb_ext.npu_fused_add_topk_div_out(x, add_num, **kwargs, y=y1, indices=i1),
    )


def case_mla_preprocess():
    (input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1, gamma2, cos, sin,
     wuk, kv_cache, kv_cache_rope, slotmapping, kwargs, outputs) = build_mla_preprocess_inputs()
    return run_value_case(
        "npu_mla_preprocess",
        lambda: torch.ops.atb.npu_mla_preprocess(
            input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping, **kwargs),
        lambda: atb_ext.npu_mla_preprocess(
            input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping, **kwargs),
    )


def case_mla_preprocess_out():
    (input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1, gamma2, cos, sin,
     wuk, kv_cache, kv_cache_rope, slotmapping, kwargs, outputs) = build_mla_preprocess_inputs()
    q_out0 = outputs["q_out0"].clone()
    kv_cache_out0 = outputs["kv_cache_out0"].clone()
    q_out1 = outputs["q_out1"].clone()
    kv_cache_out1 = outputs["kv_cache_out1"].clone()
    q_out0_cpp = outputs["q_out0"].clone()
    kv_cache_out0_cpp = outputs["kv_cache_out0"].clone()
    q_out1_cpp = outputs["q_out1"].clone()
    kv_cache_out1_cpp = outputs["kv_cache_out1"].clone()
    return run_value_case(
        "npu_mla_preprocess_out",
        lambda: torch.ops.atb.npu_mla_preprocess.out(
            input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping, **kwargs,
            q_out0=q_out0, kv_cache_out0=kv_cache_out0, q_out1=q_out1, kv_cache_out1=kv_cache_out1),
        lambda: atb_ext.npu_mla_preprocess_out(
            input1, gamma0, beta0, wdqkv, de_scale0, gamma1, beta1, wuq, de_scale1,
            gamma2, cos, sin, wuk, kv_cache, kv_cache_rope, slotmapping, **kwargs,
            q_out0=q_out0_cpp, kv_cache_out0=kv_cache_out0_cpp, q_out1=q_out1_cpp, kv_cache_out1=kv_cache_out1_cpp),
    )


def case_mha():
    common = build_multi_head_latent_attention_inputs()
    return run_value_case(
        "npu_multi_head_latent_attention",
        lambda: torch.ops.atb.npu_multi_head_latent_attention(**common),
        lambda: atb_ext.npu_multi_head_latent_attention(**common),
    )


def case_mha_out():
    common = build_multi_head_latent_attention_inputs()
    output0 = torch.empty_like(common["q_nope"])
    output1 = torch.empty_like(common["q_nope"])
    return run_value_case(
        "npu_multi_head_latent_attention_out",
        lambda: torch.ops.atb.npu_multi_head_latent_attention.out(**common, output=output0),
        lambda: atb_ext.npu_multi_head_latent_attention_out(**common, output=output1),
    )


def case_mha_lse():
    common = build_multi_head_latent_attention_inputs()
    lse_kwargs = dict(common, return_lse=True, workspace=None)
    return run_value_case(
        "npu_multi_head_latent_attention_lse",
        lambda: torch.ops.atb.npu_multi_head_latent_attention.lse(**lse_kwargs),
        lambda: atb_ext.npu_multi_head_latent_attention_lse(**lse_kwargs),
    )


def case_mha_lse_get_workspace():
    common = build_multi_head_latent_attention_inputs()
    ws_kwargs = dict(common, return_lse=True)

    def paired_python(workspace):
        output = torch.empty_like(common["q_nope"])
        lse = torch.empty((common["q_nope"].size(0), common["q_nope"].size(1), 1), dtype=common["q_nope"].dtype, device=common["q_nope"].device)
        return torch.ops.atb.npu_multi_head_latent_attention.lse_out(**common, return_lse=True, workspace=workspace, output=output, lse=lse)

    def paired_cpp(workspace):
        output = torch.empty_like(common["q_nope"])
        lse = torch.empty((common["q_nope"].size(0), common["q_nope"].size(1), 1), dtype=common["q_nope"].dtype, device=common["q_nope"].device)
        return atb_ext.npu_multi_head_latent_attention_lse_out(**common, return_lse=True, workspace=workspace, output=output, lse=lse)

    return run_workspace_case(
        "_npu_multi_head_latent_attention_lse_get_workspace",
        lambda: torch.ops.atb._npu_multi_head_latent_attention_get_workspace.lse(**ws_kwargs),
        lambda: atb_ext._npu_multi_head_latent_attention_lse_get_workspace(**ws_kwargs),
        paired_python,
        paired_cpp,
    )


def case_mha_lse_out():
    common = build_multi_head_latent_attention_inputs()
    output0 = torch.empty_like(common["q_nope"])
    lse0 = torch.empty((common["q_nope"].size(0), common["q_nope"].size(1), 1), dtype=common["q_nope"].dtype, device=common["q_nope"].device)
    output1 = torch.empty_like(common["q_nope"])
    lse1 = torch.empty((common["q_nope"].size(0), common["q_nope"].size(1), 1), dtype=common["q_nope"].dtype, device=common["q_nope"].device)
    return run_value_case(
        "npu_multi_head_latent_attention_lse_out",
        lambda: torch.ops.atb.npu_multi_head_latent_attention.lse_out(**common, return_lse=True, workspace=None, output=output0, lse=lse0),
        lambda: atb_ext.npu_multi_head_latent_attention_lse_out(**common, return_lse=True, workspace=None, output=output1, lse=lse1),
    )


def case_paged_attention():
    kwargs = build_paged_attention_inputs()
    return run_value_case(
        "_npu_paged_attention",
        lambda: (lambda out: (torch.ops.atb._npu_paged_attention(**kwargs, out=out, workspace=None), out)[1])(
            torch.zeros((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["query"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_paged_attention(**kwargs, workspace=None),
    )


def case_paged_attention_get_workspace():
    kwargs = build_paged_attention_inputs()

    def ws_python():
        out = torch.zeros((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["query"].dtype, device=kwargs["query"].device)
        return torch.ops.atb._npu_paged_attention_get_workspace(**kwargs, out=out)

    def paired_python(workspace):
        out = torch.zeros((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["query"].dtype, device=kwargs["query"].device)
        torch.ops.atb._npu_paged_attention(**kwargs, out=out, workspace=workspace)
        return out

    return run_workspace_case(
        "_npu_paged_attention_get_workspace",
        ws_python,
        lambda: atb_ext._npu_paged_attention_get_workspace(**kwargs),
        paired_python,
        lambda workspace: atb_ext._npu_paged_attention(**kwargs, workspace=workspace),
    )


def case_paged_attention_v2():
    kwargs = build_paged_attention_v2_inputs()
    out0 = torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["value_cache"].dtype, device=kwargs["query"].device)
    return run_value_case(
        "_npu_paged_attention_v2",
        lambda: torch.ops.atb._npu_paged_attention_v2.out(**kwargs, workspace=None, out=out0),
        lambda: atb_ext._npu_paged_attention_v2(**kwargs, workspace=None),
    )


def case_paged_attention_v2_get_workspace():
    kwargs = build_paged_attention_v2_inputs()

    def ws_python():
        out = torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["value_cache"].dtype, device=kwargs["query"].device)
        return torch.ops.atb._npu_paged_attention_v2_get_workspace(**kwargs, out=out)

    def paired_python(workspace):
        out = torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["value_cache"].dtype, device=kwargs["query"].device)
        return torch.ops.atb._npu_paged_attention_v2.out(**kwargs, workspace=workspace, out=out)

    return run_workspace_case(
        "_npu_paged_attention_v2_get_workspace",
        ws_python,
        lambda: atb_ext._npu_paged_attention_v2_get_workspace(**kwargs),
        paired_python,
        lambda workspace: atb_ext._npu_paged_attention_v2(**kwargs, workspace=workspace),
    )


def case_paged_attention_mla():
    kwargs = build_paged_attention_mla_inputs()
    return run_value_case(
        "_npu_paged_attention_mla",
        lambda: (lambda out: (torch.ops.atb._npu_paged_attention_mla(**kwargs, out=out), out)[1])(
            torch.zeros((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["mla_vheadsize"]), dtype=kwargs["query"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_paged_attention_mla(**kwargs),
    )


def case_paged_attention_splitfuse():
    kwargs = build_paged_attention_splitfuse_inputs()
    return run_value_case(
        "_npu_paged_attention_splitfuse",
        lambda: (lambda out: (torch.ops.atb._npu_paged_attention_splitfuse(**kwargs, out=out), out)[1])(
            torch.zeros((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["query"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_paged_attention_splitfuse(**kwargs),
    )


def case_paged_attention_quant():
    kwargs = build_paged_attention_quant_inputs()
    return run_value_case(
        "_npu_paged_attention_quant",
        lambda: (lambda out: (torch.ops.atb._npu_paged_attention_quant(**kwargs, out=out), out)[1])(
            torch.zeros_like(kwargs["query"], dtype=torch.bfloat16)),
        lambda: atb_ext._npu_paged_attention_quant(**kwargs),
    )


def case_paged_cache_load():
    kwargs = build_paged_cache_load_inputs()
    return run_value_case(
        "npu_paged_cache_load",
        lambda: torch.ops.atb.npu_paged_cache_load(**kwargs),
        lambda: atb_ext.npu_paged_cache_load(**kwargs),
    )


def case_paged_cache_load_out():
    kwargs = build_paged_cache_load_inputs()
    total_tokens = int(kwargs["context_lens"].sum().item())
    key0 = torch.empty((total_tokens, 1, kwargs["key_cache"].size(3)), dtype=kwargs["key_cache"].dtype, device=kwargs["key_cache"].device)
    value0 = torch.empty((total_tokens, 1, kwargs["value_cache"].size(3)), dtype=kwargs["value_cache"].dtype, device=kwargs["value_cache"].device)
    key1 = torch.empty_like(key0)
    value1 = torch.empty_like(value0)
    return run_value_case(
        "npu_paged_cache_load_out",
        lambda: torch.ops.atb.npu_paged_cache_load.out(**kwargs, key=key0, value=value0),
        lambda: atb_ext.npu_paged_cache_load_out(**kwargs, key=key1, value=value1),
    )


def case_reshape_and_cache():
    torch.manual_seed(5)
    key = torch.rand((4, 1, 8), dtype=torch.float16).npu()
    value = torch.rand((4, 1, 8), dtype=torch.float16).npu()
    key_cache = torch.rand((8, 4, 1, 8), dtype=torch.float16).npu()
    value_cache = torch.rand((8, 4, 1, 8), dtype=torch.float16).npu()
    slot_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    return run_value_case(
        "_npu_reshape_and_cache",
        lambda: (lambda k, v: (torch.ops.atb._npu_reshape_and_cache(key, value, k, v, slot_indices), (k, v))[1])(key_cache.clone(), value_cache.clone()),
        lambda: atb_ext._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_indices),
    )


def case_reshape_and_cache_siso():
    torch.manual_seed(6)
    key = torch.rand((4, 2, 8), dtype=torch.float16).npu()
    key_cache = torch.rand((8, 4, 2, 8), dtype=torch.float16).npu()
    slot_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32).npu()
    return run_value_case(
        "_npu_reshape_and_cache_siso",
        lambda: (lambda out: (torch.ops.atb._npu_reshape_and_cache_siso(key, out, slot_indices), out)[1])(key_cache.clone()),
        lambda: atb_ext._npu_reshape_and_cache_siso(key, key_cache, slot_indices),
    )


def case_rotary_embedding():
    positions, query, key, head_size, cache, is_neox = build_rotary_embedding_inputs()
    return run_value_case(
        "_npu_rotary_embedding",
        lambda: (lambda q, k: (torch.ops.atb._npu_rotary_embedding(positions, q, k, head_size, cache, is_neox), (q, k))[1])(query.clone(), key.clone()),
        lambda: atb_ext._npu_rotary_embedding(positions, query, key, head_size, cache, is_neox),
    )


def case_flash_attention():
    kwargs = build_flash_attention_inputs()
    return run_value_case(
        "_npu_flash_attention",
        lambda: (lambda out: (torch.ops.atb._npu_flash_attention(**kwargs, out=out), out)[1])(
            torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value"].size(2)), dtype=kwargs["value"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_flash_attention(**kwargs),
    )


def case_flash_attention_v2():
    kwargs = build_flash_attention_v2_inputs()
    out = torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value"].size(2)), dtype=kwargs["value"].dtype, device=kwargs["query"].device)
    return run_value_case(
        "_npu_flash_attention_v2",
        lambda: torch.ops.atb._npu_flash_attention_v2.out(**kwargs, out=out),
        lambda: atb_ext._npu_flash_attention_v2(**kwargs),
    )


def case_flash_attention_prefix_v2():
    kwargs = build_flash_attention_prefix_v2_inputs()
    out = torch.empty_like(kwargs["query"])
    return run_value_case(
        "_npu_flash_attention_prefix_v2",
        lambda: torch.ops.atb._npu_flash_attention_prefix_v2.out(**kwargs, out=out),
        lambda: atb_ext._npu_flash_attention_prefix_v2(**kwargs),
    )


def case_self_attention_prefix_encoder():
    kwargs = build_self_attention_prefix_encoder_inputs()
    return run_value_case(
        "npu_self_attention_prefix_encoder",
        lambda: torch.ops.atb.npu_self_attention_prefix_encoder(**kwargs),
        lambda: atb_ext.npu_self_attention_prefix_encoder(**kwargs),
    )


def case_self_attention_prefix_encoder_out():
    kwargs = build_self_attention_prefix_encoder_inputs()
    out0 = torch.empty_like(kwargs["query"])
    out1 = torch.empty_like(kwargs["query"])
    return run_value_case(
        "npu_self_attention_prefix_encoder_out",
        lambda: torch.ops.atb.npu_self_attention_prefix_encoder.out(**kwargs, output=out0),
        lambda: atb_ext.npu_self_attention_prefix_encoder_out(**kwargs, output=out1),
    )


def case_ring_mla():
    kwargs = build_ring_mla_inputs(is_ring=True)
    return run_value_case(
        "npu_ring_mla",
        lambda: torch.ops.atb.npu_ring_mla(**kwargs),
        lambda: atb_ext.npu_ring_mla(**kwargs),
    )


def case_ring_mla_out():
    kwargs = build_ring_mla_inputs(is_ring=True)
    output0 = torch.empty((kwargs["q_nope"].size(0), kwargs["q_nope"].size(1), kwargs["value"].size(2)), dtype=kwargs["value"].dtype, device=kwargs["q_nope"].device)
    lse0 = torch.empty((kwargs["q_nope"].size(1), kwargs["q_nope"].size(0)), dtype=torch.float32, device=kwargs["q_nope"].device)
    output1 = torch.empty_like(output0)
    lse1 = torch.empty_like(lse0)
    return run_value_case(
        "npu_ring_mla_out",
        lambda: torch.ops.atb.npu_ring_mla.out(**kwargs, output=output0, softmax_lse=lse0),
        lambda: atb_ext.npu_ring_mla_out(**kwargs, output=output1, softmax_lse=lse1),
    )


def case_flash_attention_qlens():
    kwargs = build_flash_attention_qlens_inputs()
    return run_value_case(
        "_npu_flash_attention_qlens",
        lambda: (lambda out: (torch.ops.atb._npu_flash_attention_qlens(**kwargs, out=out), out)[1])(
            torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value_cache"].size(3)), dtype=kwargs["value_cache"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_flash_attention_qlens(**kwargs),
    )


def case_flash_attention_unpad():
    kwargs = build_flash_attention_unpad_inputs()
    return run_value_case(
        "_npu_flash_attention_unpad",
        lambda: (lambda out: (torch.ops.atb._npu_flash_attention_unpad(**kwargs, out=out), out)[1])(
            torch.empty((kwargs["query"].size(0), kwargs["query"].size(1), kwargs["value"].size(2)), dtype=kwargs["value"].dtype, device=kwargs["query"].device)),
        lambda: atb_ext._npu_flash_attention_unpad(**kwargs),
    )


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cases = [
        case_quantize_per_tensor,
        case_quant_rms_norm,
        case_group_topk,
        case_matmul_add_fp32,
        case_fused_add_topk_div,
        case_fused_add_topk_div_out,
        case_mla_preprocess,
        case_mla_preprocess_out,
        case_mha,
        case_mha_out,
        case_mha_lse,
        case_mha_lse_get_workspace,
        case_mha_lse_out,
        case_paged_attention,
        case_paged_attention_get_workspace,
        case_paged_attention_v2,
        case_paged_attention_v2_get_workspace,
        case_paged_attention_mla,
        case_paged_attention_splitfuse,
        case_paged_attention_quant,
        case_paged_cache_load,
        case_paged_cache_load_out,
        case_reshape_and_cache,
        case_reshape_and_cache_siso,
        case_ring_mla,
        case_ring_mla_out,
        case_rotary_embedding,
        case_flash_attention,
        case_flash_attention_v2,
        case_flash_attention_prefix_v2,
        case_self_attention_prefix_encoder,
        case_self_attention_prefix_encoder_out,
        case_flash_attention_qlens,
        case_flash_attention_unpad,
    ]

    passed = 0
    failed = 0
    for case in cases:
        try:
            ok = case()
        except Exception as exc:
            ok = False
            print(f"[CASE] {case.__name__}")
            print(f"  FAIL: raised {type(exc).__name__}: {exc}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"[SUMMARY] passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
