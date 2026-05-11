import torch
import torch.nn.functional as F

import torch_npu


SHAPE = (12, 12)
P = 0.5
SEED = 2
DTYPES = (torch.float32, torch.float16)
KEYWORDS = ("dropout", "copy", "bernoulli", "aclnn", "ViewCopy")


def collect_event_names(fn):
    with torch.autograd.profiler.profile(use_device="npu") as prof:
        fn()
    names = []
    for event in prof.function_events:
        name = event.name
        if any(keyword.lower() in name.lower() for keyword in KEYWORDS):
            names.append(name)
    return names


def unique_in_order(names):
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def print_events(label, names):
    print(f"  {label}")
    if not names:
        print("    <no filtered events>")
        return
    for name in unique_in_order(names):
        print(f"    {name}")


def run_case(dtype):
    base = torch.randn(*SHAPE, dtype=dtype).npu()

    def run_noninplace():
        torch.manual_seed(SEED)
        F.dropout(base, p=P, training=True, inplace=False)

    def run_manual():
        torch.manual_seed(SEED)
        target = base.clone()
        result, _ = torch.ops.aten.native_dropout(base, P, True)
        target.copy_(result)

    def run_real_inplace():
        torch.manual_seed(SEED)
        target = base.clone()
        F.dropout(target, p=P, training=True, inplace=True)

    noninplace_events = collect_event_names(run_noninplace)
    manual_events = collect_event_names(run_manual)
    inplace_events = collect_event_names(run_real_inplace)

    print(f"dtype: {dtype}")
    print_events("noninplace", noninplace_events)
    print_events("manual(native_dropout + copy_)", manual_events)
    print_events("real inplace", inplace_events)
    print()

    return {
        "dtype": str(dtype),
        "noninplace": unique_in_order(noninplace_events),
        "manual": unique_in_order(manual_events),
        "real_inplace": unique_in_order(inplace_events),
    }


def main():
    print("Dropout inplace profiler diagnostic")
    print(f"  shape: {SHAPE}")
    print(f"  p    : {P}")
    print(f"  seed : {SEED}")
    print()

    results = [run_case(dtype) for dtype in DTYPES]
    fp16 = next(result for result in results if result["dtype"] == str(torch.float16))

    if fp16["manual"] != fp16["real_inplace"]:
        print("Conclusion: real fp16 inplace dropout dispatch differs from manual native_dropout + copy_ path.")
    else:
        print("Conclusion: profiler did not show a dispatch difference between manual and real inplace paths.")


if __name__ == "__main__":
    main()
