"""Precision checker for torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale.

The script builds inputs with ``make_inputs``, computes the CPU numpy golden
with ``golden.py``, runs the op path, and checks that the number of elements
whose absolute error is greater than 1e-3 is no more than 1e-3 of the total
element count.

The torch_npu call is temporarily disabled: ``run_npu_op`` returns zero-filled
outputs so the golden calculation and compare/reporting path can be tested
without executing the NPU kernel.

Example:
    python3 test/precision_test/qkv_rms_norm_rope_cache_with_kscale_precision.py
    python3 test/precision_test/qkv_rms_norm_rope_cache_with_kscale_precision.py --case smoke --mode inplace

The default asset directory is this script's directory. Override it with
``--assets-dir`` if the assets are elsewhere.
"""

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_npu
except ImportError:
    torch_npu = None


DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent
ATOL = 1e-3
MAX_BAD_RATIO = 1e-3
USE_ZERO_STUB = False


@dataclass
class CompareResult:
    name: str
    passed: bool
    total: int
    bad_count: int
    bad_ratio: float
    max_abs_diff: float
    first_bad: Optional[Tuple[int, ...]]
    first_actual: Optional[float]
    first_expected: Optional[float]


@dataclass
class InputCases:
    """Records parameters for ``make_inputs`` plus case metadata.

    ``post_func`` is called after ``make_inputs`` returns, allowing per-case
    modifications of the input dict (e.g. overriding slot_mapping or zeroing
    specific tensor regions).
    """
    case_name: str
    B: int
    seq_lens: list
    Nq: int
    Nk: int
    Nv: int
    D: int = 128
    block_size: int = 16
    seed: int = 0
    max_slot: Optional[int] = None
    post_func: Optional[Callable] = None


def load_golden(assets_dir):
    assets_dir = Path(assets_dir).resolve()
    golden_file = assets_dir / "golden.py"
    if not golden_file.is_file():
        raise FileNotFoundError(f"golden.py must exist under {assets_dir}")

    sys.path.insert(0, str(assets_dir))
    importlib.invalidate_caches()
    return importlib.import_module("golden")


def make_inputs(golden_mod, B, seq_lens, Nq, Nk, Nv,
                D=128, block_size=16, seed=0, max_slot=None):
    """Build one set of numpy inputs for the golden + NPU op.

    k_cache and v_cache are non-contiguous views sliced from a single
    oversized buffer so the test exercises realistic strided memory layouts.
    The raw buffer is returned alongside so the caller can transfer the
    whole storage to the NPU device and reconstruct the views via
    ``torch_slice_from_np_view``.
    """
    rng = np.random.default_rng(seed)
    assert len(seq_lens) == B
    T = int(sum(seq_lens))
    cu_seq_len = np.zeros(B + 1, dtype=np.int32)
    cu_seq_len[1:] = np.cumsum(seq_lens)
    max_seq_len = int(max(seq_lens))

    H = Nq + Nk + Nv
    qkv = golden_mod.to_bf16(rng.standard_normal((H, T, D)).astype(np.float32))
    q_gamma = rng.standard_normal(D).astype(np.float32)
    k_gamma = rng.standard_normal(D).astype(np.float32)
    cos_sin = rng.standard_normal((max_seq_len, D)).astype(np.float32)
    q_rotation = golden_mod.to_bf16(
        rng.standard_normal((D, D)).astype(np.float32) * 0.1)
    k_rotation = golden_mod.to_bf16(
        rng.standard_normal((D, D)).astype(np.float32) * 0.1)
    v_scale = rng.uniform(0.1, 1.0, size=Nv).astype(np.float32)

    # cache big enough for all referenced slots (default: T contiguous slots)
    highest_slot = (T - 1) if max_slot is None else int(max_slot)
    block_num = highest_slot // block_size + 2
    slot_mapping = np.arange(T, dtype=np.int32)

    # -- non-contiguous k_cache / v_cache from oversized buffer ---------------
    buf_shape = (block_num * 5, (Nk + Nv) * 5, block_size * 5, D)
    buffer = golden_mod.cast_to_e4m3fn_sat_rint(
        rng.standard_normal(buf_shape).astype(np.float32))

    # Slice views from the buffer — any valid offset works; the key point is
    # that the resulting views are non-contiguous in memory.
    k_cache = buffer[:block_num, :Nk, :block_size, :D]
    v_cache = buffer[:block_num, Nk:Nk + Nv, :block_size, :D]

    k_scale_cache = rng.standard_normal(
        (block_num, Nk, block_size, 1)).astype(np.float32)

    return dict(
        qkv=qkv, q_gamma=q_gamma, k_gamma=k_gamma, cos_sin=cos_sin,
        slot_mapping=slot_mapping, k_cache=k_cache, v_cache=v_cache,
        k_scale_cache=k_scale_cache, cu_seq_len=cu_seq_len,
        q_rotation=q_rotation, k_rotation=k_rotation, v_scale=v_scale,
        head_nums=[Nq, Nk, Nv], epsilon=1e-6,
        buffer=buffer,
    )


def build_case(golden_mod, ic):
    inp = make_inputs(
        golden_mod,
        B=ic.B, seq_lens=ic.seq_lens,
        Nq=ic.Nq, Nk=ic.Nk, Nv=ic.Nv,
        D=ic.D, block_size=ic.block_size,
        seed=ic.seed, max_slot=ic.max_slot,
    )
    if ic.post_func is not None:
        ic.post_func(inp)
    return inp


def get_input_cases():
    cases = [
        InputCases(
            case_name="smoke",
            B=1, seq_lens=[1], Nq=16, Nk=2, Nv=2,
        ),
        InputCases(
            case_name="normal",
            B=1, seq_lens=[128], Nq=32, Nk=4, Nv=4, seed=1,
        ),
        InputCases(
            case_name="multi_batch",
            B=2, seq_lens=[2, 7], Nq=16, Nk=2, Nv=2, seed=2,
        ),
        InputCases(
            case_name="head_combo",
            B=1, seq_lens=[64], Nq=64, Nk=8, Nv=8, seed=3,
        ),
        InputCases(
            case_name="noncontig_slots",
            B=1, seq_lens=[8], Nq=16, Nk=2, Nv=2, seed=4, max_slot=60,
            post_func=lambda inp: inp.update(
                {"slot_mapping": np.array(
                    [3, 31, 5, 48, 0, 17, 9, 60], dtype=np.int32)}),
        ),
        InputCases(
            case_name="v_scale_zero",
            B=1, seq_lens=[4], Nq=16, Nk=2, Nv=2, seed=5,
            post_func=lambda inp: inp["v_scale"].__setitem__(slice(None), 0.0),
        ),
        InputCases(
            case_name="max_abs_zero",
            B=1, seq_lens=[4], Nq=16, Nk=2, Nv=2, seed=6,
            post_func=lambda inp: inp["qkv"].__setitem__(
                (slice(None), 0, slice(None)), 0.0),
        ),
        InputCases(
            case_name=f"network_0",
            B=1, seq_lens=[32768], Nq=8, Nk=1, Nv=1, block_size=256
        ),
    ]
    for i, T_i in enumerate(range(1024, 32768, 1024)):
        cases.append(InputCases(
            case_name=f"test_{i}_T{T_i}",
            B=1, seq_lens=[T_i], Nq=8, Nk=1, Nv=1, block_size=256
        ))
    return cases


def expected_outputs(inp, golden_mod):
    golden_kwargs = {k: v for k, v in inp.items() if k != "buffer"}
    q_out, q_scale, k_cache, v_cache, k_scale_cache = (
        golden_mod.qkv_rms_norm_rope_cache_with_kscale_golden(**golden_kwargs)
    )
    return {
        # golden.py uses [Nq, T, D] / [Nq, T], while torch_npu returns
        # [T, Nq, D] / [T, Nq]. Keep the layout conversion explicit here.
        "q_out": np.asarray(q_out).astype(np.float32),
        "q_scale": np.asarray(q_scale, dtype=np.float32),
        # "q_out": np.transpose(np.asarray(q_out).astype(np.float32), (1, 0, 2)),
        # "q_scale": np.transpose(np.asarray(q_scale, dtype=np.float32), (1, 0)),
        "k_cache": np.asarray(k_cache).astype(np.float32),
        "v_cache": np.asarray(v_cache).astype(np.float32),
        "k_scale_cache": np.asarray(k_scale_cache, dtype=np.float32),
    }


def fp8_numpy_to_torch(array, golden_mod, device):
    raw = golden_mod.e4m3fn_to_uint8(array)
    storage = np.ascontiguousarray(raw.view(np.int8))
    return torch.from_numpy(storage).view(torch.float8_e4m3fn).to(device)


def numpy_to_torch(array, device, dtype=None):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(device)


def torch_slice_from_np_view(torch_dev, np_view, np_buf):
    """Create a torch tensor that mirrors a numpy view's non-contiguous layout.

    ``torch_dev`` is the torch representation of the *full* buffer (already on
    the target device).  ``np_view`` is the numpy view whose shape / strides /
    offset we want to reproduce.  ``np_buf`` is the original contiguous numpy
    buffer that ``np_view`` was sliced from.
    """
    offset_bytes = np_view.ctypes.data - np_buf.ctypes.data
    offset_elems = offset_bytes // np_buf.itemsize
    torch_strides = tuple(s // np_buf.itemsize for s in np_view.strides)
    return torch.as_strided(
        torch_dev,
        size=tuple(np_view.shape),
        stride=torch_strides,
        storage_offset=offset_elems,
    )


def make_torch_inputs(inp, golden_mod, device):
    # Transfer the entire fp8 buffer to the device, then reconstruct the
    # non-contiguous k_cache / v_cache views via torch.as_strided.
    buf_torch = fp8_numpy_to_torch(inp["buffer"], golden_mod, device)

    return {
        # "qkv": numpy_to_torch(
        #     np.transpose(inp["qkv"], (1, 0, 2)), device, torch.bfloat16),
        "qkv": numpy_to_torch(inp["qkv"], device, torch.bfloat16),
        "q_gamma": numpy_to_torch(inp["q_gamma"], device, torch.float32),
        "k_gamma": numpy_to_torch(inp["k_gamma"], device, torch.float32),
        "cos_sin": numpy_to_torch(inp["cos_sin"], device, torch.float32),
        "slot_mapping": numpy_to_torch(
            inp["slot_mapping"], device, torch.int32),
        "k_cache": torch_slice_from_np_view(
            buf_torch, inp["k_cache"], inp["buffer"]),
        "v_cache": torch_slice_from_np_view(
            buf_torch, inp["v_cache"], inp["buffer"]),
        "k_scale_cache": numpy_to_torch(
            inp["k_scale_cache"], device, torch.float32),
        "cu_seq_len": numpy_to_torch(inp["cu_seq_len"], device, torch.int32),
        "head_nums": list(inp["head_nums"]),
        "q_rotation": numpy_to_torch(
            inp["q_rotation"], device, torch.bfloat16),
        "k_rotation": numpy_to_torch(
            inp["k_rotation"], device, torch.bfloat16),
        "v_scale": numpy_to_torch(inp["v_scale"], device, torch.float32),
        "epsilon": float(inp.get("epsilon", 1e-6)),
    }


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def zero_outputs(inp):
    n_q, _, _ = inp["head_nums"]
    total_tokens = int(inp["qkv"].shape[1])
    head_dim = int(inp["qkv"].shape[2])
    return {
        "q_out": np.zeros((total_tokens, n_q, head_dim), dtype=np.float32),
        "q_scale": np.zeros((total_tokens, n_q), dtype=np.float32),
        "k_cache": np.zeros_like(
            np.asarray(inp["k_cache"], dtype=np.float32)),
        "v_cache": np.zeros_like(
            np.asarray(inp["v_cache"], dtype=np.float32)),
        "k_scale_cache": np.zeros_like(
            np.asarray(inp["k_scale_cache"], dtype=np.float32)),
    }


def run_npu_op(inp, golden_mod, mode, device):
    if USE_ZERO_STUB:
        return zero_outputs(inp)

    if torch is None or torch_npu is None:
        raise RuntimeError(
            "torch and torch_npu are required when USE_ZERO_STUB is False")

    t = make_torch_inputs(inp, golden_mod, device)
    with torch.no_grad():
        if mode == "inplace":
            q_out, q_scale = torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale(
                t["qkv"], t["q_gamma"], t["k_gamma"], t["cos_sin"],
                t["slot_mapping"],
                t["k_cache"], t["v_cache"], t["k_scale_cache"],
                t["cu_seq_len"], t["head_nums"],
                q_rotation=t["q_rotation"], k_rotation=t["k_rotation"],
                v_scale=t["v_scale"],
                epsilon=t["epsilon"],
            )
            outputs = (q_out, q_scale,
                       t["k_cache"], t["v_cache"], t["k_scale_cache"])
        elif mode == "functional":
            outputs = torch_npu.npu_qkv_rms_norm_rope_cache_with_kscale_functional(
                t["qkv"], t["q_gamma"], t["k_gamma"], t["cos_sin"],
                t["slot_mapping"],
                t["k_cache"], t["v_cache"], t["k_scale_cache"],
                t["cu_seq_len"], t["head_nums"],
                q_rotation=t["q_rotation"], k_rotation=t["k_rotation"],
                v_scale=t["v_scale"],
                epsilon=t["epsilon"],
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    q_out, q_scale, k_cache, v_cache, k_scale_cache = outputs
    return {
        "q_out": tensor_to_numpy(q_out),
        "q_scale": tensor_to_numpy(q_scale),
        "k_cache": tensor_to_numpy(k_cache),
        "v_cache": tensor_to_numpy(v_cache),
        "k_scale_cache": tensor_to_numpy(k_scale_cache),
    }


def compare_array(name, actual, expected, atol, max_bad_ratio):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        return CompareResult(
            name, False, 1, 1, 1.0, float("inf"), None, None, None)

    total = int(actual.size)
    if total == 0:
        return CompareResult(
            name, True, 0, 0, 0.0, 0.0, None, None, None)

    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)

    diff = np.zeros(actual.shape, dtype=np.float32)
    diff[finite] = np.abs(actual[finite] - expected[finite])

    inf_mismatch = (
        (np.isinf(actual) | np.isinf(expected)) & (actual != expected))
    bad = (actual_nan != expected_nan) | inf_mismatch | (finite & (diff > atol))
    bad_count = int(np.count_nonzero(bad))
    bad_ratio = bad_count / total
    max_abs_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0

    first_bad = None
    first_actual = None
    first_expected = None
    if bad_count:
        first_bad = tuple(int(i) for i in np.argwhere(bad)[0])
        first_actual = float(actual[first_bad])
        first_expected = float(expected[first_bad])

    return CompareResult(
        name=name,
        passed=bad_ratio <= max_bad_ratio,
        total=total,
        bad_count=bad_count,
        bad_ratio=bad_ratio,
        max_abs_diff=max_abs_diff,
        first_bad=first_bad,
        first_actual=first_actual,
        first_expected=first_expected,
    )


def compare_outputs(actual, expected, atol, max_bad_ratio):
    names = ["q_out", "q_scale", "k_cache", "v_cache", "k_scale_cache"]
    return [
        compare_array(name, actual[name], expected[name], atol, max_bad_ratio)
        for name in names
    ]


def print_results(case_name, mode, results):
    case_passed = all(result.passed for result in results)
    status = "PASS" if case_passed else "FAIL"
    total = sum(result.total for result in results)
    correct = sum(result.total - result.bad_count for result in results)
    precision_percent = 100.0 if total == 0 else correct / total * 100.0
    print(
        f"[{status}] case={case_name} mode={mode} "
        f"precision={precision_percent:.4f}% correct={correct}/{total}"
    )
    for result in results:
        correct_count = result.total - result.bad_count
        tensor_precision = (
            100.0 if result.total == 0
            else correct_count / result.total * 100.0)
        line = (
            f"    {result.name}: precision={tensor_precision:.4f}% "
            f"correct={correct_count}/{result.total} "
            f"bad={result.bad_count}/{result.total} "
            f"ratio={result.bad_ratio:.6g} "
            f"max_abs_diff={result.max_abs_diff:.6g}"
        )
        if result.first_bad is not None:
            line += (
                f" first_bad={result.first_bad} "
                f"actual={result.first_actual} "
                f"expected={result.first_expected}"
            )
        print(line)
    return case_passed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument(
        "--case", action="append",
        help="case name to run; may be repeated; default: all")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("inplace", "functional", "both"), default="both")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument(
        "--max-bad-ratio", type=float, default=MAX_BAD_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()
    golden_mod = load_golden(args.assets_dir)
    input_cases = get_input_cases()
    case_names = [ic.case_name for ic in input_cases]

    if args.list_cases:
        print("\n".join(case_names))
        return 0

    if (not USE_ZERO_STUB
            and (torch is None
                 or not hasattr(torch, "npu")
                 or not torch.npu.is_available())):
        raise RuntimeError(
            "NPU device is required to run this precision checker")

    selected = set(args.case or case_names)
    unknown = selected - set(case_names)
    if unknown:
        raise ValueError(
            f"unknown case(s): {sorted(unknown)}; available: {case_names}")

    modes = ["inplace", "functional"] if args.mode == "both" else [args.mode]
    all_passed = True
    for ic in input_cases:
        if ic.case_name not in selected:
            continue
        inp = build_case(golden_mod, ic)
        expected = expected_outputs(inp, golden_mod)
        for mode in modes:
            actual = run_npu_op(inp, golden_mod, mode, args.device)
            results = compare_outputs(
                actual, expected, args.atol, args.max_bad_ratio)
            all_passed = (
                print_results(ic.case_name, mode, results) and all_passed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
