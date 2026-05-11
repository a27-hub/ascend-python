import torch
import torch.nn.functional as F

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
    noninplace = F.dropout(input_tensor, p=P, training=True, inplace=False)

    torch.manual_seed(SEED)
    manual_target = input_tensor.clone()
    manual_result, _manual_mask = torch.ops.aten.native_dropout(input_tensor, P, True)
    manual_target.copy_(manual_result)

    torch.manual_seed(SEED)
    real_inplace = input_tensor.clone()
    F.dropout(real_inplace, p=P, training=True, inplace=True)

    print(f"dtype: {dtype}")
    eq_nm, err_nm = print_comparison("noninplace vs manual(native_dropout + copy_)", noninplace, manual_target)
    eq_mr, err_mr = print_comparison("manual(native_dropout + copy_) vs real inplace", manual_target, real_inplace)
    eq_nr, err_nr = print_comparison("noninplace vs real inplace", noninplace, real_inplace)
    print()
    return {
        "dtype": str(dtype),
        "eq_nm": eq_nm,
        "err_nm": err_nm,
        "eq_mr": eq_mr,
        "err_mr": err_mr,
        "eq_nr": eq_nr,
        "err_nr": err_nr,
    }


def main():
    print("Dropout inplace path diagnostic")
    print(f"  shape: {SHAPE}")
    print(f"  p    : {P}")
    print(f"  seed : {SEED}")
    print()

    results = [run_case(dtype) for dtype in DTYPES]
    fp32 = next(result for result in results if result["dtype"] == str(torch.float32))
    fp16 = next(result for result in results if result["dtype"] == str(torch.float16))

    fp32_ok = fp32["eq_nm"] and fp32["eq_mr"] and fp32["eq_nr"]
    fp16_manual_ok = fp16["eq_nm"]
    fp16_real_bad = not fp16["eq_mr"] and not fp16["eq_nr"]

    if fp32_ok and fp16_manual_ok and fp16_real_bad:
        print("Conclusion: real fp16 dropout_ path diverges from manual native_dropout + copy_ composition.")
    else:
        print("Conclusion: this run did not isolate a real inplace-path divergence clearly enough.")


if __name__ == "__main__":
    main()
