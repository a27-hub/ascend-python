import torch
import torch_npu
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
import pandas as pd

torch.npu.set_device(0)


class TorchDump(TorchDispatchMode):
    def __init__(self, dump_path=""):
        self._n_ops = 0
        self.dump_path = dump_path
        self.dump_data = []
        self.prefix = dump_path.split("_")[-1].split(".")[0]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        op = f"{func}_{self._n_ops}"
        self._n_ops += 1
        flat_args, _ = pytree.tree_flatten(args)
        flat_out, _ = pytree.tree_flatten(out)
        data = {
            "op_name": op,
            "input": [],
            "kwargs": kwargs,
            "output": [],
        }

        for idx, t in enumerate(flat_args):
            if not torch.is_tensor(t):
                continue
            torch.save(t.cpu(), f"{self.prefix}_{op}_input_{idx}.pt")
            data["input"].append(
                {
                    "mean": t.mean().item(),
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }
            )

        for idx, t in enumerate(flat_out):
            if not torch.is_tensor(t):
                continue
            torch.save(t.cpu(), f"{self.prefix}_{op}_output_{idx}.pt")
            data["output"].append(
                {
                    "mean": t.mean().item(),
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                }
            )

        self.dump_data.append(data)

        return out

    def dump(self):
        df = pd.DataFrame(self.dump_data)
        df.to_csv(self.dump_path, index=False, encoding="utf-8")

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.dump()


if __name__ == "__main__":
    # 1. 加载 ATK --save_data 导出的 tensor（按需改文件名与索引）
    a_raw = torch.load("tensor_input.bin")
    b_raw = torch.load("input.bin")[0]

    device_npu = "npu"
    a_npu = a_raw.to(device_npu).clone().requires_grad_(True)
    b_npu = b_raw.to(device_npu).clone().requires_grad_(True)

    print("Running NPU Dump...")
    with TorchDump(dump_path="remainder_ops_npu.csv"):
        c_npu = a_npu.remainder(b_npu)
        c_npu.backward(torch.load("npu_output_grad_0.pt").to("npu"))

    device_cpu = "cpu"
    a_cpu = a_raw.to(device_cpu).clone().requires_grad_(True)
    b_cpu = b_raw.to(device_cpu).clone().requires_grad_(True)

    print("Running CPU Dump...")
    with TorchDump(dump_path="remainder_ops_cpu.csv"):
        c_cpu = a_cpu.remainder(b_cpu)
        c_cpu.backward(torch.load("cpu_output_grad_0.pt"))

    print("Dump 完成：remainder_ops_npu.csv / remainder_ops_cpu.csv")
