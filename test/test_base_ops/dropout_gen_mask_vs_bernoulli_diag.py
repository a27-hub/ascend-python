import numpy as np
import torch
import torch_npu


torch.npu.set_compile_mode(jit_compile=False)

SHAPE = (12, 12)
P = 0.5
SEED = 2
OFFSET = 0


def unpack_keep_mask(mask_tensor: torch.Tensor, numel: int) -> np.ndarray:
    mask_bytes = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    keep_bits = np.unpackbits(mask_bytes, bitorder="little")[:numel]
    return keep_bits.astype(bool)


def bernoulli_keep_mask(dtype: torch.dtype, numel: int) -> np.ndarray:
    torch.manual_seed(SEED)
    keep_prob = 1.0 - P
    mask = torch.ones(numel, dtype=dtype, device="npu")
    mask.bernoulli_(keep_prob)
    return mask.detach().cpu().numpy().reshape(-1) != 0


def print_mask_stats(title: str, keep_mask: np.ndarray) -> None:
    keep_count = int(keep_mask.sum())
    drop_count = int(keep_mask.size - keep_count)
    print(f"  {title}")
    print(f"    keep_count: {keep_count}")
    print(f"    drop_count: {drop_count}")
    print(f"    keep_ratio: {keep_count / keep_mask.size}")


def run_case(dtype: torch.dtype) -> dict:
    x = torch.ones(SHAPE, dtype=dtype, device="npu")
    torch.manual_seed(SEED)
    gen_mask = torch_npu._npu_dropout_gen_mask(x, SHAPE, P, SEED, OFFSET, parallel=False)
    gen_keep_mask = unpack_keep_mask(gen_mask, x.numel())
    bern_keep_mask = bernoulli_keep_mask(dtype, x.numel())

    diff = gen_keep_mask ^ bern_keep_mask
    diff_indices = np.flatnonzero(diff)

    print(f"dtype: {dtype}")
    print_mask_stats("gen_mask keep mask", gen_keep_mask)
    print_mask_stats("bernoulli keep mask", bern_keep_mask)
    print("  gen_mask keep mask vs bernoulli keep mask")
    print(f"    equal     : {bool(np.array_equal(gen_keep_mask, bern_keep_mask))}")
    print(f"    diff_count: {int(diff.sum())}")
    if diff_indices.size:
        idx = int(diff_indices[0])
        print(f"    first_mismatch_index: {idx}")
        print(f"    lhs_value           : {bool(gen_keep_mask[idx])}")
        print(f"    rhs_value           : {bool(bern_keep_mask[idx])}")
    print()

    return {
        "dtype": str(dtype),
        "equal": bool(np.array_equal(gen_keep_mask, bern_keep_mask)),
        "diff_count": int(diff.sum()),
    }


def main() -> None:
    print("Dropout gen_mask vs bernoulli diagnostic")
    print(f"  shape    : {SHAPE}")
    print(f"  p        : {P}")
    print(f"  seed     : {SEED}")
    print(f"  offset   : {OFFSET}")
    print(f"  keep_prob: {1.0 - P}")
    print()

    fp32 = run_case(torch.float32)
    fp16 = run_case(torch.float16)

    if fp32["equal"] and not fp16["equal"]:
        print("Conclusion: fp16 gen_mask keep mask diverges from bernoulli keep mask.")
    else:
        print("Conclusion: this run did not isolate a dtype-specific gen_mask vs bernoulli divergence.")


if __name__ == "__main__":
    main()
