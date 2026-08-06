# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical coverage for MXFP4 mode-6 PageAttention on Ascend950.

The CPU oracle owns a canonical [P, N, Bs, D] FP4 representation and resolves
the block table independently.  Device tensors are separately packed into NZ,
BNBD, or BBH storage, so a shared layout/packing bug cannot validate itself.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch_npu

from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
GROUP_SIZE = 32
D0 = 16


@dataclass(frozen=True)
class Case:
    name: str
    layout: str
    q_dtype: torch.dtype
    asymmetric: bool
    b: int
    q_heads: int
    kv_heads: int
    s1: int
    d: int
    block_size: int
    physical_pages: int
    kv_lens: Tuple[int, ...]
    block_table: Tuple[Tuple[int, ...], ...]
    seed: int


def _core_cases():
    cases = []
    table = ((9, 2, 5), (1, 8, 3), (7, 0, 6))
    for layout in ("nz", "bnbd", "bbh"):
        for q_dtype in (torch.float16, torch.bfloat16):
            for asymmetric in (False, True):
                dtype_name = "fp16" if q_dtype == torch.float16 else "bf16"
                mode_name = "asymmetric" if asymmetric else "symmetric"
                cases.append(Case(
                    f"{layout}_{dtype_name}_{mode_name}_three_page_tail",
                    layout, q_dtype, asymmetric, 3, 3, 3, 1, 128, 128, 10,
                    (257, 130, 1), table, 1729,
                ))
    return cases


def _generalization_cases():
    cases = []
    gqa_table = ((9, 2, 5), (1, 8, 3), (7, 0, 6))
    for layout in ("nz", "bnbd", "bbh"):
        cases.append(Case(
            f"{layout}_bf16_asymmetric_gqa_page_boundaries",
            layout, torch.bfloat16, True, 3, 4, 2, 2, 128, 128, 10,
            (128, 256, 257), gqa_table, 999983,
        ))
    for layout in ("bnbd", "bbh"):
        cases.append(Case(
            f"{layout}_bf16_asymmetric_group1_d32_boundaries",
            layout, torch.bfloat16, True, 3, 2, 2, 1, 32, 128, 12,
            (1, 128, 129), ((11, 3), (8, 1), (10, 4)), 2097143,
        ))
        cases.append(Case(
            f"{layout}_bf16_asymmetric_group6_d192_three_page_tail",
            layout, torch.bfloat16, True, 2, 3, 3, 1, 192, 128, 10,
            (257, 129), ((9, 2, 5), (1, 8, 3)), 8191,
        ))
        cases.append(Case(
            f"{layout}_bf16_asymmetric_group8_d256_three_page_tail",
            layout, torch.bfloat16, True, 2, 3, 3, 1, 256, 128, 10,
            (257, 129), ((9, 2, 5), (1, 8, 3)), 32749,
        ))
        cases.append(Case(
            f"{layout}_bf16_asymmetric_group16_d512_boundaries",
            layout, torch.bfloat16, True, 2, 2, 2, 1, 512, 128, 12,
            (127, 129), ((11, 3), (8, 1)), 4194301,
        ))
    for layout in ("nz", "bnbd", "bbh"):
        cases.append(Case(
            f"{layout}_fp16_symmetric_single_page_b1n1",
            layout, torch.float16, False, 1, 1, 1, 1, 128, 128, 3,
            (127,), ((2,),), 2026,
        ))
    return cases


POSITIVE_CASES = tuple(_core_cases() + _generalization_cases())
REFERENCE_CASES = tuple(case for case in POSITIVE_CASES if case.layout == "nz" and "three_page_tail" in case.name)


def _canonical_inputs(case):
    if case.d % GROUP_SIZE != 0 or case.q_heads % case.kv_heads != 0:
        raise ValueError(f"invalid case contract: {case.name}")
    p = np.arange(case.physical_pages, dtype=np.int64)[:, None, None, None]
    n = np.arange(case.kv_heads, dtype=np.int64)[None, :, None, None]
    s = np.arange(case.block_size, dtype=np.int64)[None, None, :, None]
    d = np.arange(case.d, dtype=np.int64)[None, None, None, :]
    key_codes = ((11 * p + 7 * n + 5 * s + 3 * d + case.seed) & 0xF).astype(np.uint8)
    value_codes = ((11 * p + 7 * n + 5 * s + 3 * d + case.seed + 9) & 0xF).astype(np.uint8)

    g = np.arange(case.d // GROUP_SIZE, dtype=np.int64)[None, None, None, :]
    key_scale = (125 + ((5 * p + 3 * n + s + 7 * g + case.seed) & 0x3)).astype(np.uint8)
    value_scale = (125 + ((5 * p + 3 * n + s + 7 * g + case.seed + 1) & 0x3)).astype(np.uint8)
    key_offset = (0.03125 * (((7 * p + 5 * n + s + 3 * g + case.seed) % 17) - 8)).astype(np.float32)
    value_offset = (0.03125 * (((7 * p + 5 * n + s + 3 * g + case.seed + 1) % 17) - 8)).astype(np.float32)

    b = np.arange(case.b, dtype=np.int64)[:, None, None, None]
    qh = np.arange(case.q_heads, dtype=np.int64)[None, :, None, None]
    qs = np.arange(case.s1, dtype=np.int64)[None, None, :, None]
    qd = np.arange(case.d, dtype=np.int64)[None, None, None, :]
    query = (0.03125 * (((5 * b + 3 * qh + 7 * qs + qd + case.seed) % 15) - 7)).astype(np.float32)
    return query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset


def _pack_last_axis(logical_codes):
    if logical_codes.shape[-1] % 2 != 0:
        raise ValueError("packed FP4 axis must be even")
    packed = logical_codes[..., 0::2] | (logical_codes[..., 1::2] << 4)
    return np.ascontiguousarray(packed, dtype=np.uint8)


def _pack_kv(canonical_codes, layout):
    p, n, bs, d = canonical_codes.shape
    if layout == "nz":
        logical = canonical_codes.reshape(p, n, bs, d // D0, D0).transpose(0, 1, 3, 2, 4)
    elif layout == "bnbd":
        logical = canonical_codes
    elif layout == "bbh":
        logical = canonical_codes.transpose(0, 2, 1, 3).reshape(p, bs, n * d)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return _pack_last_axis(logical)


def _layout_param(canonical_param, layout):
    p, n, bs, g = canonical_param.shape
    if layout == "nz":
        physical = canonical_param.transpose(0, 1, 3, 2)
    elif layout == "bnbd":
        physical = canonical_param
    elif layout == "bbh":
        physical = canonical_param.transpose(0, 2, 1, 3).reshape(p, bs, n * g)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return np.ascontiguousarray(physical)


def _dequantize(codes, scale_codes, offsets, asymmetric, out_dtype):
    code_indices = torch.from_numpy(codes.astype(np.int64, copy=False))
    values = FP4_E2M1[code_indices].to(torch.bfloat16)
    scales = torch.from_numpy(np.ldexp(np.ones_like(scale_codes, dtype=np.float32),
                                       scale_codes.astype(np.int32) - 127)).to(torch.bfloat16)
    scales = scales.repeat_interleave(GROUP_SIZE, dim=-1)
    if asymmetric:
        offset_tensor = torch.from_numpy(offsets).to(out_dtype).to(torch.bfloat16)
        offset_tensor = offset_tensor.repeat_interleave(GROUP_SIZE, dim=-1)
        values = (values + offset_tensor).to(torch.bfloat16)
    return (values * scales).to(torch.bfloat16).to(out_dtype)


def _golden(case, query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset):
    query_tensor = torch.from_numpy(query).to(case.q_dtype).to(torch.float32)
    key_dequant = _dequantize(key_codes, key_scale, key_offset, case.asymmetric, case.q_dtype)
    value_dequant = _dequantize(value_codes, value_scale, value_offset, case.asymmetric, case.q_dtype)
    result = torch.empty((case.b, case.q_heads, case.s1, case.d), dtype=case.q_dtype)
    q_per_kv = case.q_heads // case.kv_heads
    block_table = torch.tensor(case.block_table, dtype=torch.int64)
    for batch in range(case.b):
        seq_len = case.kv_lens[batch]
        physical = block_table[batch, torch.arange(seq_len) // case.block_size]
        token = torch.arange(seq_len) % case.block_size
        for q_head in range(case.q_heads):
            kv_head = q_head // q_per_kv
            key_seq = key_dequant[physical, kv_head, token].to(torch.float32)
            value_seq = value_dequant[physical, kv_head, token].to(torch.float32)
            scores = torch.matmul(query_tensor[batch, q_head], key_seq.transpose(0, 1)) * (case.d ** -0.5)
            probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
            result[batch, q_head] = torch.matmul(probability, value_seq).to(case.q_dtype)
    return result


def _fia_precision(actual, expected, q_dtype):
    """Repository FIA gate used by the C++ numerical example and FIA tests."""
    actual_f32 = actual.to(torch.float32).reshape(-1)
    expected_f32 = expected.to(torch.float32).reshape(-1)
    if actual_f32.numel() != expected_f32.numel():
        return False, f"element-count mismatch: {actual_f32.numel()} != {expected_f32.numel()}"
    if actual_f32.numel() == 0:
        return True, "empty output"

    rtol = 0.0078125 if q_dtype == torch.bfloat16 else 0.005
    atol = 0.0001 if q_dtype == torch.bfloat16 else 0.000025
    absolute_error = torch.abs(actual_f32 - expected_f32)
    both_nan = torch.isnan(actual_f32) & torch.isnan(expected_f32)
    exactly_equal = actual_f32 == expected_f32
    both_finite = torch.isfinite(actual_f32) & torch.isfinite(expected_f32)
    close = both_nan | exactly_equal | (both_finite & (absolute_error <= atol + rtol * torch.abs(expected_f32)))
    failed = ~close
    failed_count = int(failed.sum().item())
    fulfill_percent = 100.0 * (actual_f32.numel() - failed_count) / actual_f32.numel()

    max_relative_error = 0.0
    if failed_count:
        relative_floor = (1.0 / (1 << 14)) / 0.005
        denominator = torch.maximum(torch.maximum(torch.abs(actual_f32), torch.abs(expected_f32)),
                                    torch.tensor(relative_floor)) + 1.0e-10
        relative_error = torch.where(both_finite, absolute_error / denominator,
                                     torch.full_like(absolute_error, float("inf")))
        max_relative_error = float(relative_error[failed].max().item())
    passed = fulfill_percent >= 99.5 and max_relative_error < 10.0
    detail = (f"rtol={rtol} atol={atol} fulfill_percent={fulfill_percent:.6f} "
              f"failed={failed_count}/{actual_f32.numel()} max_relative_error={max_relative_error:.7f}")
    return passed, detail


class TestFusedInferAttentionScoreV2Mxfp4Pa(TestCase):
    def _logical_dtypes(self):
        if not hasattr(torch_npu, "float4_e2m1fn_x2"):
            self.fail("Ascend950 MXFP4 PA prerequisite missing dtype: float4_e2m1fn_x2")
        if hasattr(torch_npu, "float8_e8m0fnu"):
            e8m0_dtype = torch_npu.float8_e8m0fnu
        elif hasattr(torch_npu, "float8_e8m0"):
            # Older torch_npu branches expose the same ACL_FLOAT8_E8M0 logical
            # dtype under this compatibility name.
            e8m0_dtype = torch_npu.float8_e8m0
        else:
            self.fail("Ascend950 MXFP4 PA prerequisite missing dtype: float8_e8m0fnu/float8_e8m0")
        return torch_npu.float4_e2m1fn_x2, e8m0_dtype

    def _run_case(self, case, override_key_scale=None):
        fp4_dtype, e8m0_dtype = self._logical_dtypes()
        canonical = _canonical_inputs(case)
        query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset = canonical
        key_bytes = torch.from_numpy(_pack_kv(key_codes, case.layout)).npu()
        value_bytes = torch.from_numpy(_pack_kv(value_codes, case.layout)).npu()
        key_scale_physical = _layout_param(key_scale, case.layout) if override_key_scale is None else override_key_scale
        key_scale_npu = torch.from_numpy(key_scale_physical).npu()
        value_scale_npu = torch.from_numpy(_layout_param(value_scale, case.layout)).npu()
        key_offset_npu = None
        value_offset_npu = None
        if case.asymmetric:
            key_offset_npu = torch.from_numpy(_layout_param(key_offset, case.layout)).to(case.q_dtype).npu()
            value_offset_npu = torch.from_numpy(_layout_param(value_offset, case.layout)).to(case.q_dtype).npu()
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            torch.from_numpy(query).to(case.q_dtype).npu(), key_bytes, value_bytes,
            actual_seq_qlen=[case.s1] * case.b,
            actual_seq_kvlen=list(case.kv_lens),
            block_table=torch.tensor(case.block_table, dtype=torch.int32).npu(),
            dequant_scale_key=key_scale_npu,
            dequant_offset_key=key_offset_npu,
            dequant_scale_value=value_scale_npu,
            dequant_offset_value=value_offset_npu,
            num_query_heads=case.q_heads,
            num_key_value_heads=case.kv_heads,
            softmax_scale=case.d ** -0.5,
            pre_tokens=65536,
            next_tokens=65536,
            input_layout="BNSD",
            sparse_mode=0,
            block_size=case.block_size,
            query_quant_mode=0,
            key_quant_mode=6,
            value_quant_mode=6,
            inner_precise=1,
            return_softmax_lse=False,
            key_dtype=fp4_dtype,
            value_dtype=fp4_dtype,
            dequant_scale_key_dtype=e8m0_dtype,
            dequant_scale_value_dtype=e8m0_dtype,
        )
        expected = _golden(case, *canonical)
        return output.cpu(), expected

    @SupportedDevices(["Ascend950"])
    def test_mxfp4_mode6_page_attention(self, device="npu"):
        selected = os.getenv("MXFP4_PA_PTA_CASE")
        matched = 0
        for case in REFERENCE_CASES:
            if selected and selected != case.name:
                continue
            matched += 1
            with self.subTest(case=case.name):
                actual, expected = self._run_case(case)
                passed, detail = _fia_precision(actual, expected, case.q_dtype)
                self.assertTrue(passed, f"{case.name}: {detail}")
        if selected and matched == 0:
            self.fail(f"MXFP4_PA_PTA_CASE did not match a case: {selected}")

    @SupportedDevices(["Ascend950"])
    def test_mxfp4_mode6_page_attention_layout_equivalence(self, device="npu"):
        peer_groups = {}
        for case in POSITIVE_CASES:
            signature = (case.q_dtype, case.asymmetric, case.b, case.q_heads, case.kv_heads,
                         case.s1, case.d, case.block_size, case.physical_pages,
                         case.kv_lens, case.block_table, case.seed)
            peer_groups.setdefault(signature, []).append(case)
        for signature, peers in peer_groups.items():
            if len(peers) < 2:
                continue
            baseline_output = None
            baseline_expected = None
            for case in peers:
                actual, expected = self._run_case(case)
                if baseline_output is None:
                    baseline_output = actual
                    baseline_expected = expected
                    continue
                self.assertTrue(torch.equal(baseline_expected, expected),
                                f"CPU oracle differs across layouts for {signature}")
                self.assertTrue(torch.equal(baseline_output, actual),
                                f"device output differs across layouts for {signature}: {case.layout}")

    @SupportedDevices(["Ascend950"])
    def test_mxfp4_mode6_page_attention_rejects_wrong_param_layout(self, device="npu"):
        negative_cases = (
            next(case for case in POSITIVE_CASES if case.name == "bnbd_bf16_asymmetric_group1_d32_boundaries"),
            next(case for case in POSITIVE_CASES if case.name == "bbh_bf16_asymmetric_group16_d512_boundaries"),
        )
        for case in negative_cases:
            canonical = _canonical_inputs(case)
            key_scale = canonical[3]
            wrong_layout = (key_scale.transpose(0, 1, 3, 2) if case.layout == "bnbd" else key_scale)
            with self.subTest(case=case.name), self.assertRaises(RuntimeError):
                self._run_case(case, np.ascontiguousarray(wrong_layout))


if __name__ == "__main__":
    run_tests()
