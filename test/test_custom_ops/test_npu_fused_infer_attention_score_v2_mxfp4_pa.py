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

import csv
import os
from dataclasses import dataclass
from pathlib import Path
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
    peer_group: str
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


def _case_file():
    configured = os.getenv("MXFP4_PA_PTA_CASE_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "data" / "fused_infer_attention_score_mxfp4_pa_business_cases.csv"


def _parse_int_tuple(text):
    return tuple(int(value) for value in text.split(";") if value)


def _parse_block_table(text):
    return tuple(_parse_int_tuple(row) for row in text.split("|") if row)


def _validate_case(case):
    if case.layout not in ("nz", "bnbd", "bbh"):
        raise ValueError(f"{case.name}: unsupported layout {case.layout}")
    if case.b <= 0 or case.q_heads <= 0 or case.kv_heads <= 0 or case.s1 <= 0:
        raise ValueError(f"{case.name}: B/QH/KVH/S1 must be positive")
    if case.q_heads % case.kv_heads != 0:
        raise ValueError(f"{case.name}: QH must be divisible by KVH")
    if case.d <= 0 or case.d % GROUP_SIZE != 0 or case.d % D0 != 0:
        raise ValueError(f"{case.name}: D must be divisible by {GROUP_SIZE} and {D0}")
    if case.block_size <= 0 or case.physical_pages <= 0:
        raise ValueError(f"{case.name}: block_size and physical_pages must be positive")
    if len(case.kv_lens) != case.b or len(case.block_table) != case.b:
        raise ValueError(f"{case.name}: kv_lens and block_table rows must equal B")
    if not case.block_table or not case.block_table[0]:
        raise ValueError(f"{case.name}: block_table must not be empty")
    table_width = len(case.block_table[0])
    if any(len(row) != table_width for row in case.block_table):
        raise ValueError(f"{case.name}: block_table must be rectangular")
    if any(length <= 0 or length > table_width * case.block_size for length in case.kv_lens):
        raise ValueError(f"{case.name}: kv_lens exceed block_table capacity")
    if any(page < 0 or page >= case.physical_pages for row in case.block_table for page in row):
        raise ValueError(f"{case.name}: block_table contains an invalid physical page")
    if case.seed <= 0:
        raise ValueError(f"{case.name}: seed must be positive")


def _load_cases():
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    path = _case_file()
    cases = []
    names = set()
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(line for line in csv_file if not line.lstrip().startswith("#")):
            name = row["name"].strip()
            if name in names:
                raise ValueError(f"duplicate case name: {name}")
            dtype_name = row["q_dtype"].strip().lower()
            if dtype_name not in dtype_map:
                raise ValueError(f"{name}: unsupported q_dtype {dtype_name}")
            asymmetric_text = row["asymmetric"].strip()
            if asymmetric_text not in ("0", "1"):
                raise ValueError(f"{name}: asymmetric must be 0 or 1")
            case = Case(
                name=name,
                peer_group=row["peer_group"].strip(),
                layout=row["layout"].strip().lower(),
                q_dtype=dtype_map[dtype_name],
                asymmetric=asymmetric_text == "1",
                b=int(row["b"]),
                q_heads=int(row["q_heads"]),
                kv_heads=int(row["kv_heads"]),
                s1=int(row["s1"]),
                d=int(row["d"]),
                block_size=int(row["block_size"]),
                physical_pages=int(row["physical_pages"]),
                kv_lens=_parse_int_tuple(row["kv_lens"]),
                block_table=_parse_block_table(row["block_table"]),
                seed=int(row["seed"]),
            )
            _validate_case(case)
            names.add(name)
            cases.append(case)
    if not cases:
        raise ValueError(f"no cases loaded from {path}")
    peer_groups = {}
    for case in cases:
        if not case.peer_group:
            raise ValueError(f"{case.name}: peer_group must not be empty")
        peer_groups.setdefault(case.peer_group, []).append(case)
    for peer_group, peers in peer_groups.items():
        if len(peers) != 3 or {peer.layout for peer in peers} != {"nz", "bnbd", "bbh"}:
            raise ValueError(f"{peer_group}: peer group must contain exactly one NZ, BNBD and BBH case")
        signatures = {
            (peer.q_dtype, peer.asymmetric, peer.b, peer.q_heads, peer.kv_heads, peer.s1, peer.d,
             peer.block_size, peer.physical_pages, peer.kv_lens, peer.block_table, peer.seed)
            for peer in peers
        }
        if len(signatures) != 1:
            raise ValueError(f"{peer_group}: logical inputs differ across layouts")
    return tuple(cases)


POSITIVE_CASES = _load_cases()


def _selected_cases():
    selected_name = os.getenv("MXFP4_PA_PTA_CASE")
    selected_group = os.getenv("MXFP4_PA_PTA_GROUP")
    if selected_name and selected_group:
        raise ValueError("set only one of MXFP4_PA_PTA_CASE and MXFP4_PA_PTA_GROUP")
    if selected_name:
        selected = tuple(case for case in POSITIVE_CASES if case.name == selected_name)
    elif selected_group:
        selected = tuple(case for case in POSITIVE_CASES if case.peer_group == selected_group)
    else:
        selected = POSITIVE_CASES
    if not selected:
        requested = selected_name or selected_group
        raise ValueError(f"no case matched selection: {requested}")
    return selected


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


def _layout_predequant_kv(canonical_kv, layout):
    p, n, bs, d = canonical_kv.shape
    if layout == "nz":
        physical = canonical_kv.reshape(p, n, bs, d // D0, D0).permute(0, 1, 3, 2, 4)
    elif layout == "bnbd":
        physical = canonical_kv
    elif layout == "bbh":
        physical = canonical_kv.permute(0, 2, 1, 3).reshape(p, bs, n * d)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return physical.contiguous()


def _packed_kv_with_storage_mode(packed, storage_mode, case_name, tensor_name):
    packed_npu = torch.from_numpy(packed).npu()
    if storage_mode == "direct":
        return packed_npu

    padded_shape = list(packed_npu.shape)
    padded_shape[0] += 1
    storage = torch.empty(padded_shape, dtype=packed_npu.dtype, device=packed_npu.device)
    view = storage.narrow(0, 1, packed_npu.size(0))
    view.copy_(packed_npu)
    if storage_mode == "clone":
        view = view.clone(memory_format=torch.contiguous_format)
    elif storage_mode != "slice":
        raise ValueError(f"unsupported packed KV storage mode: {storage_mode}")

    print(f"[MXFP4_STORAGE_OFFSET_INPUT][{case_name}][{tensor_name}] mode={storage_mode} "
          f"dtype={view.dtype} element_size={view.element_size()} shape={tuple(view.shape)} "
          f"stride={view.stride()} storage_offset={view.storage_offset()} "
          f"is_contiguous={str(view.is_contiguous()).lower()}")
    if storage_mode == "slice":
        assert view.is_contiguous() and view.storage_offset() > 0
    else:
        assert view.is_contiguous() and view.storage_offset() == 0
    return view


def _paged_to_contiguous_kv(case, canonical_kv):
    s2_max = max(case.kv_lens)
    contiguous = torch.zeros((case.b, case.kv_heads, s2_max, case.d), dtype=case.q_dtype)
    for batch, seq_len in enumerate(case.kv_lens):
        copied = 0
        logical_page = 0
        while copied < seq_len:
            physical_page = case.block_table[batch][logical_page]
            copy_tokens = min(case.block_size, seq_len - copied)
            contiguous[batch, :, copied:copied + copy_tokens] = \
                canonical_kv[physical_page, :, :copy_tokens]
            copied += copy_tokens
            logical_page += 1
    return contiguous


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
                                    torch.tensor(relative_floor)) + 2.0e-9
        relative_error = torch.where(both_finite, absolute_error / denominator,
                                     torch.full_like(absolute_error, float("inf")))
        max_relative_error = float(relative_error[failed].max().item())
    passed = fulfill_percent >= 99.5 and max_relative_error < 10.0
    detail = (f"rtol={rtol} atol={atol} fulfill_percent={fulfill_percent:.6f} "
              f"failed={failed_count}/{actual_f32.numel()} max_relative_error={max_relative_error:.7f}")
    return passed, detail


class TestFusedInferAttentionScoreV2Mxfp4Pa(TestCase):
    _result_cache = {}
    _predequant_result_cache = {}
    _nonpa_result_cache = {}

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

    def _run_case(self, case, override_key_scale=None, kv_storage_mode="direct"):
        fp4_dtype, e8m0_dtype = self._logical_dtypes()
        canonical = _canonical_inputs(case)
        query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset = canonical
        key_bytes = _packed_kv_with_storage_mode(
            _pack_kv(key_codes, case.layout), kv_storage_mode, case.name, "key")
        value_bytes = _packed_kv_with_storage_mode(
            _pack_kv(value_codes, case.layout), kv_storage_mode, case.name, "value")
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

    def _run_cached_case(self, case):
        if case.name not in self._result_cache:
            self._result_cache[case.name] = self._run_case(case)
        return self._result_cache[case.name]

    def _run_predequant_case(self, case):
        canonical = _canonical_inputs(case)
        query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset = canonical
        key_dequant = _dequantize(key_codes, key_scale, key_offset, case.asymmetric, case.q_dtype)
        value_dequant = _dequantize(value_codes, value_scale, value_offset, case.asymmetric, case.q_dtype)
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            torch.from_numpy(query).to(case.q_dtype).npu(),
            _layout_predequant_kv(key_dequant, case.layout).npu(),
            _layout_predequant_kv(value_dequant, case.layout).npu(),
            actual_seq_qlen=[case.s1] * case.b,
            actual_seq_kvlen=list(case.kv_lens),
            block_table=torch.tensor(case.block_table, dtype=torch.int32).npu(),
            num_query_heads=case.q_heads,
            num_key_value_heads=case.kv_heads,
            softmax_scale=case.d ** -0.5,
            pre_tokens=65536,
            next_tokens=65536,
            input_layout="BNSD",
            sparse_mode=0,
            block_size=case.block_size,
            query_quant_mode=0,
            key_quant_mode=0,
            value_quant_mode=0,
            inner_precise=1,
            return_softmax_lse=False,
        )
        return output.cpu()

    def _run_cached_predequant_case(self, case):
        if case.name not in self._predequant_result_cache:
            self._predequant_result_cache[case.name] = self._run_predequant_case(case)
        return self._predequant_result_cache[case.name]

    def _run_nonpa_case(self, case):
        canonical = _canonical_inputs(case)
        query, key_codes, value_codes, key_scale, value_scale, key_offset, value_offset = canonical
        key_dequant = _dequantize(key_codes, key_scale, key_offset, case.asymmetric, case.q_dtype)
        value_dequant = _dequantize(value_codes, value_scale, value_offset, case.asymmetric, case.q_dtype)
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            torch.from_numpy(query).to(case.q_dtype).npu(),
            _paged_to_contiguous_kv(case, key_dequant).npu(),
            _paged_to_contiguous_kv(case, value_dequant).npu(),
            actual_seq_qlen=[case.s1] * case.b,
            actual_seq_kvlen=list(case.kv_lens),
            num_query_heads=case.q_heads,
            num_key_value_heads=case.kv_heads,
            softmax_scale=case.d ** -0.5,
            pre_tokens=65536,
            next_tokens=65536,
            input_layout="BNSD",
            sparse_mode=0,
            block_size=0,
            query_quant_mode=0,
            key_quant_mode=0,
            value_quant_mode=0,
            inner_precise=1,
            return_softmax_lse=False,
        )
        return output.cpu()

    def _run_cached_nonpa_case(self, case):
        if case.peer_group not in self._nonpa_result_cache:
            self._nonpa_result_cache[case.peer_group] = self._run_nonpa_case(case)
        return self._nonpa_result_cache[case.peer_group]

    @SupportedDevices(["Ascend950", "Ascend950P"])
    def test_mxfp4_mode6_page_attention_peer_parity(self, device="npu"):
        selected = _selected_cases()
        print(f"[MXFP4_PTA_PEER] batch_cases={len(selected)} "
              "gates=mode6_vs_predequant_pa,predequant_pa_vs_nonpa,mode6_pa_vs_nonpa")
        for case in selected:
            with self.subTest(case=case.name):
                actual, _ = self._run_cached_case(case)
                predequant = self._run_cached_predequant_case(case)
                nonpa = self._run_cached_nonpa_case(case)
                comparisons = (
                    ("mode6_vs_predequant_pa", actual, predequant),
                    ("predequant_pa_vs_nonpa", predequant, nonpa),
                    ("mode6_pa_vs_nonpa", actual, nonpa),
                )
                for comparison, lhs, rhs in comparisons:
                    passed, detail = _fia_precision(lhs, rhs, case.q_dtype)
                    exact = torch.equal(lhs, rhs)
                    max_abs = float((lhs.to(torch.float32) - rhs.to(torch.float32)).abs().max().item())
                    print(f"[MXFP4_PTA_PEER][{case.name}] {comparison} "
                          f"{'PASS' if passed else 'FAIL'} exact={str(exact).lower()} "
                          f"max_abs_difference={max_abs:.7f} {detail}")
                    self.assertTrue(passed, f"{case.name}: {comparison}: {detail}")

    @SupportedDevices(["Ascend950", "Ascend950P"])
    def test_mxfp4_mode6_page_attention_cpu_diagnostic(self, device="npu"):
        if os.getenv("MXFP4_PA_PTA_RUN_CPU_DIAGNOSTIC") != "1":
            self.skipTest("CPU oracle diagnostic requires MXFP4_PA_PTA_RUN_CPU_DIAGNOSTIC=1")
        run_nonpa_peer = os.getenv("MXFP4_PA_PTA_NONPA_PEER") == "1"
        run_predequant_peer = os.getenv("MXFP4_PA_PTA_PREDEQUANT_PEER") == "1" or run_nonpa_peer
        for case in _selected_cases():
            with self.subTest(case=case.name):
                actual, expected = self._run_cached_case(case)
                passed, detail = _fia_precision(actual, expected, case.q_dtype)
                print(f"[MXFP4_PTA_CPU_DIAGNOSTIC][{case.name}] {'PASS' if passed else 'FAIL'} "
                      f"acceptance=DIAGNOSTIC_ONLY {detail}")
                if run_predequant_peer:
                    predequant = self._run_cached_predequant_case(case)
                    predequant_passed, predequant_detail = _fia_precision(predequant, expected, case.q_dtype)
                    peer_passed, peer_detail = _fia_precision(actual, predequant, case.q_dtype)
                    exact = torch.equal(actual, predequant)
                    max_abs = float((actual.to(torch.float32) - predequant.to(torch.float32)).abs().max().item())
                    print(f"[MXFP4_PTA_BASELINE][{case.name}] predequant_vs_cpu "
                          f"{'PASS' if predequant_passed else 'FAIL'} {predequant_detail}")
                    print(f"[MXFP4_PTA_BASELINE][{case.name}] mode6_vs_predequant "
                          f"{'PASS' if peer_passed else 'FAIL'} exact={str(exact).lower()} "
                          f"max_abs_difference={max_abs:.7f} {peer_detail}")
                    if run_nonpa_peer:
                        nonpa = self._run_cached_nonpa_case(case)
                        nonpa_passed, nonpa_detail = _fia_precision(nonpa, expected, case.q_dtype)
                        pa_nonpa_passed, pa_nonpa_detail = _fia_precision(predequant, nonpa, case.q_dtype)
                        mode6_nonpa_passed, mode6_nonpa_detail = _fia_precision(actual, nonpa, case.q_dtype)
                        pa_nonpa_exact = torch.equal(predequant, nonpa)
                        pa_nonpa_max_abs = float(
                            (predequant.to(torch.float32) - nonpa.to(torch.float32)).abs().max().item())
                        mode6_nonpa_max_abs = float(
                            (actual.to(torch.float32) - nonpa.to(torch.float32)).abs().max().item())
                        print(f"[MXFP4_PTA_NONPA][{case.name}] nonpa_vs_cpu "
                              f"{'PASS' if nonpa_passed else 'FAIL'} {nonpa_detail}")
                        print(f"[MXFP4_PTA_NONPA][{case.name}] predequant_pa_vs_nonpa "
                              f"{'PASS' if pa_nonpa_passed else 'FAIL'} "
                              f"exact={str(pa_nonpa_exact).lower()} max_abs_difference={pa_nonpa_max_abs:.7f} "
                              f"{pa_nonpa_detail}")
                        print(f"[MXFP4_PTA_NONPA][{case.name}] mode6_pa_vs_nonpa "
                              f"{'PASS' if mode6_nonpa_passed else 'FAIL'} "
                              f"max_abs_difference={mode6_nonpa_max_abs:.7f} {mode6_nonpa_detail}")

    @SupportedDevices(["Ascend950", "Ascend950P"])
    def test_mxfp4_mode6_page_attention_layout_equivalence(self, device="npu"):
        selected = _selected_cases()
        selected_names = {case.name for case in selected}
        peer_groups = {}
        for case in POSITIVE_CASES:
            peer_groups.setdefault(case.peer_group, []).append(case)
        tested_group = False
        for peer_group, peers in peer_groups.items():
            if len(peers) != 3 or not all(peer.name in selected_names for peer in peers):
                continue
            tested_group = True
            baseline_output = None
            baseline_expected = None
            for case in peers:
                actual, expected = self._run_cached_case(case)
                if baseline_output is None:
                    baseline_output = actual
                    baseline_expected = expected
                    continue
                self.assertTrue(torch.equal(baseline_expected, expected),
                                f"CPU oracle differs across layouts for {peer_group}")
                self.assertTrue(torch.equal(baseline_output, actual),
                                f"device output differs across layouts for {peer_group}: {case.layout}")
            print(f"[MXFP4_PTA_EQUIV][{peer_group}] PASS layouts=nz,bnbd,bbh bitwise_exact=yes")
        if not tested_group:
            self.skipTest("layout equivalence needs a complete peer group; use MXFP4_PA_PTA_GROUP")

    @SupportedDevices(["Ascend950", "Ascend950P"])
    def test_mxfp4_mode6_page_attention_nonzero_storage_offset(self, device="npu"):
        if os.getenv("MXFP4_PA_PTA_STORAGE_OFFSET") != "1":
            self.skipTest("nonzero storage_offset regression requires MXFP4_PA_PTA_STORAGE_OFFSET=1")
        for case in _selected_cases():
            with self.subTest(case=case.name):
                sliced, _ = self._run_case(case, kv_storage_mode="slice")
                cloned, _ = self._run_case(case, kv_storage_mode="clone")
                exact = torch.equal(sliced, cloned)
                max_abs = float((sliced.to(torch.float32) - cloned.to(torch.float32)).abs().max().item())
                print(f"[MXFP4_STORAGE_OFFSET_RESULT][{case.name}] "
                      f"{'PASS' if exact else 'FAIL'} slice_vs_clone_bitwise_exact={str(exact).lower()} "
                      f"max_abs_difference={max_abs:.7f}")
                self.assertTrue(exact, f"{case.name}: nonzero storage_offset output differs from clone")

    @SupportedDevices(["Ascend950", "Ascend950P"])
    def test_mxfp4_mode6_page_attention_rejects_wrong_param_layout(self, device="npu"):
        if os.getenv("MXFP4_PA_PTA_CASE") or os.getenv("MXFP4_PA_PTA_GROUP"):
            self.skipTest("negative layout checks run only without a case/group filter")
        negative_cases = (
            next(case for case in POSITIVE_CASES if case.name ==
                 "bnbd_d256_bf16_asym_mha_qh4_kvh4_s1x1_tail"),
            next(case for case in POSITIVE_CASES if case.name ==
                 "bbh_d256_fp16_asym_gqa_qh8_kvh2_s1x2_boundaries"),
        )
        for case in negative_cases:
            canonical = _canonical_inputs(case)
            key_scale = canonical[3]
            wrong_layout = (key_scale.transpose(0, 1, 3, 2) if case.layout == "bnbd" else key_scale)
            with self.subTest(case=case.name), self.assertRaises(RuntimeError):
                self._run_case(case, np.ascontiguousarray(wrong_layout))


if __name__ == "__main__":
    run_tests()
