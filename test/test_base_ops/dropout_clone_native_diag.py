import torch

import torch_npu


SHAPE = (12, 12)
P = 0.5
SEED = 2
DTYPES = (torch.float32, torch.float16)


def tensor_meta(tensor: torch.Tensor):
    return {
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "data_ptr": hex(tensor.data_ptr()),
    }


def max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs - rhs).abs().max().cpu().item()


def first_mismatch(lhs: torch.Tensor, rhs: torch.Tensor):
    mismatch = lhs.ne(rhs)
    if not mismatch.any().cpu().item():
        return None
    flat_index = mismatch.reshape(-1).nonzero(as_tuple=False)[0].cpu().item()
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    return {
        "index": flat_index,
        "lhs_value": lhs_flat[flat_index].cpu().item(),
        "rhs_value": rhs_flat[flat_index].cpu().item(),
    }


def print_meta(label: str, tensor: torch.Tensor):
    meta = tensor_meta(tensor)
    print(f"  {label}")
    for key, value in meta.items():
        print(f"    {key:14}: {value}")


def print_comparison(label: str, lhs: torch.Tensor, rhs: torch.Tensor):
    equal = torch.equal(lhs, rhs)
    error = max_abs_error(lhs, rhs)
    print(f"  {label}")
    print(f"    equal        : {equal}")
    print(f"    max_abs_error: {error}")
    if not equal:
        mismatch = first_mismatch(lhs, rhs)
        if mismatch is not None:
            print(f"    first_mismatch_index: {mismatch['index']}")
            print(f"    lhs_value           : {mismatch['lhs_value']}")
            print(f"    rhs_value           : {mismatch['rhs_value']}")
    return equal, error


def run_native_dropout(input_tensor: torch.Tensor):
    torch.manual_seed(SEED)
    return torch.ops.aten.native_dropout(input_tensor, P, True)


def run_case(dtype: torch.dtype):
    input_tensor = torch.randn(*SHAPE, dtype=dtype).npu()
    cloned_tensor = input_tensor.clone()

    print(f"dtype: {dtype}")
    print_meta("input", input_tensor)
    print_meta("clone", cloned_tensor)

    result_input, mask_input = run_native_dropout(input_tensor)
    result_clone, mask_clone = run_native_dropout(cloned_tensor)

    eq_result, err_result = print_comparison("native_dropout result: input vs clone", result_input, result_clone)
    eq_mask, err_mask = print_comparison("native_dropout mask  : input vs clone", mask_input, mask_clone)
    print()
    return {
        "dtype": str(dtype),
        "eq_result": eq_result,
        "err_result": err_result,
        "eq_mask": eq_mask,
        "err_mask": err_mask,
    }


def main():
    print("Dropout native clone diagnostic")
    print(f"  shape: {SHAPE}")
    print(f"  p    : {P}")
    print(f"  seed : {SEED}")
    print()

    results = [run_case(dtype) for dtype in DTYPES]
    fp32 = next(result for result in results if result["dtype"] == str(torch.float32))
    fp16 = next(result for result in results if result["dtype"] == str(torch.float16))

    fp32_ok = fp32["eq_result"] and fp32["eq_mask"]
    fp16_bad = (not fp16["eq_result"]) or (not fp16["eq_mask"])

    if fp32_ok and fp16_bad:
        print("Conclusion: fp16 mismatch is introduced during native_dropout computation for clone input, before copy-back.")
    else:
        print("Conclusion: this run did not isolate clone-sensitive native_dropout divergence clearly enough.")


if __name__ == "__main__":
    main()
