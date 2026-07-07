"""
Probe script: compare npu_trans_quant_param(round_mode=0) vs round_mode=1).

Run on NPU machine (example):
  cd /path/to/op-plugin
  python test/test_custom_ops/test_npu_trans_quant_param_round_mode_probe.py

Or:
  python -m pytest test/test_custom_ops/test_npu_trans_quant_param_round_mode_probe.py -s

Copy the full stdout and send back for analysis.
"""
import struct
import sys
import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


def f32_bits(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", np.float32(x)))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def pack_int64_from_f32_bits(u32: np.ndarray) -> np.int64:
    u64 = np.uint64(u32) | (np.uint64(1) << 46)
    return np.int64(u64)


def golden_truncate_only(scale: np.ndarray) -> np.ndarray:
    """Hypothesis A: round_mode=0 — direct truncate to high 19 bits (existing test golden)."""
    u32 = scale.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return pack_int64_from_f32_bits(u32)


def golden_rint_then_truncate(scale: np.ndarray) -> np.ndarray:
    """Hypothesis B: round_mode=1 — numpy rint then truncate (rough CPU model)."""
    rounded = np.rint(scale.astype(np.float64)).astype(np.float32)
    u32 = rounded.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return pack_int64_from_f32_bits(u32)


def golden_round_then_truncate(scale: np.ndarray) -> np.ndarray:
    """Hypothesis C: round_mode=1 — numpy round then truncate."""
    rounded = np.round(scale.astype(np.float64)).astype(np.float32)
    u32 = rounded.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return pack_int64_from_f32_bits(u32)


def run_npu(scale_np: np.ndarray, round_mode: int, offset_np=None):
    scale_t = torch.from_numpy(scale_np.astype(np.float32)).npu()
    if offset_np is None:
        out = torch_npu.npu_trans_quant_param(scale_t, round_mode=round_mode)
    else:
        offset_t = torch.from_numpy(offset_np.astype(np.float32)).npu()
        out = torch_npu.npu_trans_quant_param(scale_t, offset_t, round_mode=round_mode)
    return out.cpu().numpy().astype(np.int64)


def fmt_row(scale_val, out0, out1, g0, g1_rint, g1_round):
    bits = f32_bits(scale_val)
    same_01 = "SAME" if out0 == out1 else "DIFF"
    m0_trunc = "Y" if out0 == g0 else "N"
    m1_trunc = "Y" if out1 == g0 else "N"
    m1_rint = "Y" if out1 == g1_rint else "N"
    m1_round = "Y" if out1 == g1_round else "N"
    return (
        f"scale={scale_val:12.8g}  f32_bits=0x{bits:08X}  "
        f"mode0=0x{int(out0):016X}  mode1=0x{int(out1):016X}  "
        f"0vs1={same_01}  "
        f"m0==trunc:{m0_trunc}  m1==trunc:{m1_trunc}  m1==rint+trunc:{m1_rint}  m1==round+trunc:{m1_round}"
    )


# Values chosen so low mantissa bits may differ between truncate vs round-then-truncate.
PROBE_SCALES = [
    2.0,
    2.5,
    2.5000001,
    2.4999999,
    1.0000001,
    1.9999999,
    3.1415927,
    0.125,
    0.126,
    100.0,
    100.00003,
    -2.5,
    -1.0000001,
    1e-3,
    1e-3 + 1e-7,
    65504.0,   # near fp16 max, still valid fp32
    1.1754944e-38,  # ~fp32 min normal
]


class TestTransQuantParamRoundModeProbe(TestCase):
    @SupportedDevices(["Ascend910B", "Ascend950"])
    def test_round_mode_probe(self, device="npu"):
        print("\n" + "=" * 100)
        print("npu_trans_quant_param round_mode probe")
        print("torch:", torch.__version__)
        print("device:", torch.npu.get_device_name(0) if torch.npu.is_available() else "N/A")
        print("=" * 100)

        scale_batch = np.array(PROBE_SCALES, dtype=np.float32)

        try:
            out0_batch = run_npu(scale_batch, round_mode=0)
            out1_batch = run_npu(scale_batch, round_mode=1)
        except Exception as e:
            print("ERROR calling npu_trans_quant_param with round_mode=1:")
            print(repr(e))
            print("\nLikely CANN only supports round_mode=0 (TransQuantParamV2). Send this output back.")
            raise

        g0_batch = golden_truncate_only(scale_batch)
        g1_rint_batch = golden_rint_then_truncate(scale_batch)
        g1_round_batch = golden_round_then_truncate(scale_batch)

        print("\n--- per-value results (scale shape=1,) ---")
        diff_count = 0
        for i, s in enumerate(PROBE_SCALES):
            out0 = out0_batch[i]
            out1 = out1_batch[i]
            if out0 != out1:
                diff_count += 1
            print(fmt_row(s, out0, out1, g0_batch[i], g1_rint_batch[i], g1_round_batch[i]))

        print(f"\nSummary: {diff_count}/{len(PROBE_SCALES)} values differ between mode0 and mode1")

        mode0_match_trunc = int(np.sum(out0_batch == g0_batch))
        mode1_match_trunc = int(np.sum(out1_batch == g0_batch))
        mode1_match_rint = int(np.sum(out1_batch == g1_rint_batch))
        mode1_match_round = int(np.sum(out1_batch == g1_round_batch))

        print(f"mode0 matches truncate-only golden: {mode0_match_trunc}/{len(PROBE_SCALES)}")
        print(f"mode1 matches truncate-only golden: {mode1_match_trunc}/{len(PROBE_SCALES)}")
        print(f"mode1 matches rint+truncate golden:  {mode1_match_rint}/{len(PROBE_SCALES)}")
        print(f"mode1 matches round+truncate golden: {mode1_match_round}/{len(PROBE_SCALES)}")

        # batch shape (N,) — same API shape as existing tests
        print("\n--- batch shape test: scale.shape=(8,) ---")
        batch = np.linspace(0.5, 4.0, 8, dtype=np.float32)
        b0 = run_npu(batch, round_mode=0)
        b1 = run_npu(batch, round_mode=1)
        bg0 = golden_truncate_only(batch)
        print(f"batch mode0==truncate golden all: {bool(np.all(b0 == bg0))}")
        print(f"batch mode0==mode1 all: {bool(np.all(b0 == b1))}")
        if not np.all(b0 == b1):
            for i in range(len(batch)):
                if b0[i] != b1[i]:
                    print(f"  batch[{i}] scale={batch[i]:.6f} mode0=0x{int(b0[i]):016X} mode1=0x{int(b1[i]):016X}")

        # with offset
        print("\n--- offset test: scale.shape=(4,), offset.shape=(4,) ---")
        scale4 = np.array([1.1, 2.5, 3.3, 4.7], dtype=np.float32)
        offset4 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        o0 = run_npu(scale4, round_mode=0, offset_np=offset4)
        o1 = run_npu(scale4, round_mode=1, offset_np=offset4)
        print(f"offset case mode0==mode1 all: {bool(np.all(o0 == o1))}")
        for i in range(4):
            tag = "SAME" if o0[i] == o1[i] else "DIFF"
            print(f"  [{i}] scale={scale4[i]} offset={offset4[i]} mode0=0x{int(o0[i]):016X} mode1=0x{int(o1[i]):016X} {tag}")

        print("\n--- interpretation guide ---")
        print("1) If mode0 matches truncate golden for all values -> round_mode=0 is truncate-only (expected).")
        print("2) If mode0==mode1 always -> round_mode may have no effect on this CANN version.")
        print("3) If mode1 differs from mode0 but mode1==truncate golden -> ticket swap (mode1 no-op) is WRONG.")
        print("4) If mode1 differs and matches rint/round+truncate golden -> mode1 adds rounding before pack.")
        print("=" * 100 + "\n")

        # Do not fail the run; this is a data-collection probe.
        self.assertTrue(True)


if __name__ == "__main__":
    # Allow plain `python .../test_npu_trans_quant_param_round_mode_probe.py`
    if len(sys.argv) == 1:
        sys.argv.append(__name__)
    run_tests()
