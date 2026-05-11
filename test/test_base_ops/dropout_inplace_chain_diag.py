import torch
import torch.nn.functional as F

import torch_npu


SHAPE = (12, 12)
P = 0.5
SEED = 2
KEEP_PROB = 1.0 - P
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


def zero_ratio(tensor: torch.Tensor) -> float:
    return tensor.eq(0).float().mean().cpu().item()


def unique_values(tensor: torch.Tensor):
    values = torch.unique(tensor.cpu())
    return [value.item() for value in values[:8]]


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
    base = torch.ones(*SHAPE, dtype=dtype).npu()

    torch.manual_seed(SEED)
    native_result, native_mask = torch.ops.aten.native_dropout(base, P, True)

    torch.manual_seed(SEED)
    noise = torch.empty_like(base)
    noise.bernoulli_(KEEP_PROB)
    noise.div_(KEEP_PROB)
    manual_chain = base.clone()
    manual_chain.mul_(noise)

    torch.manual_seed(SEED)
    real_inplace = base.clone()
    F.dropout(real_inplace, p=P, training=True, inplace=True)

    print(f"dtype: {dtype}")
    print(f"  native zero_ratio : {zero_ratio(native_result)}")
    print(f"  manual zero_ratio : {zero_ratio(manual_chain)}")
    print(f"  inplace zero_ratio: {zero_ratio(real_inplace)}")
    print(f"  native unique     : {unique_values(native_result)}")
    print(f"  manual unique     : {unique_values(manual_chain)}")
    print(f"  inplace unique    : {unique_values(real_inplace)}")
    eq_nm, err_nm = print_comparison("native_dropout vs manual bernoulli/div/mul", native_result, manual_chain)
    eq_mi, err_mi = print_comparison("manual bernoulli/div/mul vs real inplace", manual_chain, real_inplace)
    eq_ni, err_ni = print_comparison("native_dropout vs real inplace", native_result, real_inplace)
    print()
    return {
        "dtype": str(dtype),
        "eq_nm": eq_nm,
        "err_nm": err_nm,
        "eq_mi": eq_mi,
        "err_mi": err_mi,
        "eq_ni": eq_ni,
        "err_ni": err_ni,
    }


def main():
    print("Dropout inplace chain diagnostic")
    print(f"  shape    : {SHAPE}")
    print(f"  p        : {P}")
    print(f"  seed     : {SEED}")
    print(f"  keep_prob: {KEEP_PROB}")
    print()

    results = [run_case(dtype) for dtype in DTYPES]
    fp32 = next(result for result in results if result["dtype"] == str(torch.float32))
    fp16 = next(result for result in results if result["dtype"] == str(torch.float16))

    fp32_ok = fp32["eq_nm"] and fp32["eq_mi"] and fp32["eq_ni"]
    fp16_chain_matches_inplace = fp16["eq_mi"]
    fp16_native_diverges = not fp16["eq_nm"] and not fp16["eq_ni"]

    if fp32_ok and fp16_chain_matches_inplace and fp16_native_diverges:
        print("Conclusion: fp16 real inplace dropout matches the bernoulli/div/mul chain and diverges from native_dropout.")
    else:
        print("Conclusion: this run did not isolate the inplace bernoulli/div/mul chain clearly enough.")


if __name__ == "__main__":
    main()
