import sys

import torch
import torch_npu
import torch_npu.op_plugin.atb._atb_ops  # noqa: F401

import atb_quantize_per_tensor_extension as atb_ext


def build_inputs():
    input_x = torch.tensor(
        [
            -2.0, -1.0, -0.5, 0.0,
            0.5, 1.0, 1.5, 2.0,
            -1.5, -0.25, 0.25, 0.75,
            1.25, 1.75, 2.25, -2.25,
        ],
        dtype=torch.float16,
    ).view(1, 4, 4)
    scale = torch.full((4, 4), 0.5, dtype=torch.float16)
    offset = torch.ones((4, 4), dtype=torch.int8)
    return input_x.npu(), scale.npu(), offset.npu()


def first_mismatch(actual_cpu, expected_cpu):
    diff = actual_cpu.ne(expected_cpu)
    if not diff.any().item():
        return None
    index = diff.nonzero(as_tuple=False)[0].tolist()
    index_tuple = tuple(index)
    return index, actual_cpu[index_tuple].item(), expected_cpu[index_tuple].item()


def main():
    input_x, scale, offset = build_inputs()

    expected = torch.zeros_like(input_x, dtype=torch.int8)
    torch_npu._npu_quantize_per_tensor(input_x, scale, offset, expected)

    actual = atb_ext.run_atb_quantize_per_tensor(input_x, scale, offset)

    expected_cpu = expected.cpu()
    actual_cpu = actual.cpu()

    mismatch = first_mismatch(actual_cpu, expected_cpu)
    if mismatch is None:
        print("PASS: _npu_quantize_per_tensor")
        return 0

    index, actual_value, expected_value = mismatch
    print("FAIL: _npu_quantize_per_tensor")
    print(
        f"mismatch at {index}: actual={actual_value}, expected={expected_value}",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
