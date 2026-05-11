import torch

import torch_npu


SHAPE = (12, 12)
P = 0.5
SEED = 2
KEEP_PROB = 1.0 - P
DTYPES = (torch.float32, torch.float16)


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


def mask_stats(mask: torch.Tensor):
    keep_count = mask.sum().cpu().item()
    numel = mask.numel()
    drop_count = numel - keep_count
    keep_ratio = keep_count / numel
    return {
        "keep_count": keep_count,
        "drop_count": drop_count,
        "keep_ratio": keep_ratio,
    }


def print_mask(label: str, mask: torch.Tensor):
    stats = mask_stats(mask)
    print(f"  {label}")
    print(f"    keep_count: {stats['keep_count']}")
    print(f"    drop_count: {stats['drop_count']}")
    print(f"    keep_ratio: {stats['keep_ratio']}")


def print_comparison(label: str, lhs: torch.Tensor, rhs: torch.Tensor):
    equal = torch.equal(lhs, rhs)
    diff_count = lhs.ne(rhs).sum().cpu().item()
    print(f"  {label}")
    print(f"    equal     : {equal}")
    print(f"    diff_count: {diff_count}")
    if not equal:
        mismatch = first_mismatch(lhs, rhs)
        if mismatch is not None:
            print(f"    first_mismatch_index: {mismatch['index']}")
            print(f"    lhs_value           : {mismatch['lhs_value']}")
            print(f"    rhs_value           : {mismatch['rhs_value']}")
    return equal


def native_keep_mask(dtype: torch.dtype) -> torch.Tensor:
    base = torch.ones(*SHAPE, dtype=dtype).npu()
    torch.manual_seed(SEED)
    result, _ = torch.ops.aten.native_dropout(base, P, True)
    return result.ne(0)


def bernoulli_keep_mask(dtype: torch.dtype) -> torch.Tensor:
    noise = torch.empty(*SHAPE, dtype=dtype).npu()
    torch.manual_seed(SEED)
    noise.bernoulli_(KEEP_PROB)
    return noise.ne(0)


def run_case(dtype: torch.dtype):
    mask_native = native_keep_mask(dtype)
    mask_bernoulli = bernoulli_keep_mask(dtype)

    print(f"dtype: {dtype}")
    print_mask("native keep mask", mask_native)
    print_mask("bernoulli keep mask", mask_bernoulli)
    equal = print_comparison("native keep mask vs bernoulli keep mask", mask_native, mask_bernoulli)
    print()
    return equal


def main():
    print("Dropout bernoulli mask diagnostic")
    print(f"  shape    : {SHAPE}")
    print(f"  p        : {P}")
    print(f"  seed     : {SEED}")
    print(f"  keep_prob: {KEEP_PROB}")
    print()

    results = {str(dtype): run_case(dtype) for dtype in DTYPES}
    fp32_ok = results[str(torch.float32)]
    fp16_ok = results[str(torch.float16)]

    if fp32_ok and not fp16_ok:
        print("Conclusion: fp16 bernoulli_ keep mask diverges from native_dropout keep mask.")
    elif fp32_ok and fp16_ok:
        print("Conclusion: this run did not reproduce a bernoulli_ vs native_dropout mask mismatch.")
    else:
        print("Conclusion: float32 control path also diverged; investigate environment or seed setup first.")


if __name__ == "__main__":
    main()
