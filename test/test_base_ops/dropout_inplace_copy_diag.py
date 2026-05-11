import torch

import torch_npu


SHAPE = (12, 12)
P = 0.5
SEED = 2
DTYPES = (torch.float32, torch.float16)


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


def run_case(dtype: torch.dtype):
    input_tensor = torch.randn(*SHAPE, dtype=dtype).npu()

    torch.manual_seed(SEED)
    output_noninplace = torch.nn.Dropout(p=P, inplace=False)(input_tensor)

    copied = input_tensor.clone()
    copied.copy_(output_noninplace)

    torch.manual_seed(SEED)
    inplace_input = input_tensor.clone()
    torch.nn.Dropout(p=P, inplace=True)(inplace_input)

    print(f"dtype: {dtype}")
    eq_copy, err_copy = print_comparison("noninplace vs copied", output_noninplace, copied)
    eq_inplace, err_inplace = print_comparison("noninplace vs inplace_result", output_noninplace, inplace_input)
    eq_copy_inplace, err_copy_inplace = print_comparison("copied vs inplace_result", copied, inplace_input)
    print()
    return {
        "dtype": str(dtype),
        "eq_copy": eq_copy,
        "err_copy": err_copy,
        "eq_inplace": eq_inplace,
        "err_inplace": err_inplace,
        "eq_copy_inplace": eq_copy_inplace,
        "err_copy_inplace": err_copy_inplace,
    }


def main():
    print("Dropout inplace copy diagnostic")
    print(f"  shape: {SHAPE}")
    print(f"  p    : {P}")
    print(f"  seed : {SEED}")
    print()

    results = [run_case(dtype) for dtype in DTYPES]
    fp32 = next(result for result in results if result["dtype"] == str(torch.float32))
    fp16 = next(result for result in results if result["dtype"] == str(torch.float16))

    fp32_ok = fp32["eq_copy"] and fp32["eq_inplace"] and fp32["eq_copy_inplace"]
    fp16_copy_bad = not fp16["eq_copy"]
    fp16_inplace_bad = not fp16["eq_inplace"]
    fp16_copy_inplace_close = fp16["eq_copy_inplace"] or fp16["err_copy_inplace"] <= fp16["err_inplace"]

    if fp32_ok and fp16_copy_bad and fp16_inplace_bad and fp16_copy_inplace_close:
        print("Conclusion: fp16 mismatch is introduced by copy-back after dropout result generation.")
    else:
        print("Conclusion: this run did not isolate the mismatch cleanly enough to blame copy-back alone.")


if __name__ == "__main__":
    main()
