import subprocess
import sys

import torch

import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests


FAILURE_SCRIPT = """
import torch
import torch_npu

value = {value}
tensor = torch.tensor(value, dtype={dtype}, device="npu")
{assert_call}
print("ASSERT_ENQUEUED", flush=True)
torch.npu.synchronize()
print("ASSERT_COMPLETED", flush=True)
"""


class TestAssertAsync(TestCase):
    @SupportedDevices(["Ascend950"])
    def test_assert_async_success(self):
        cases = [
            (True, torch.bool),
            (1, torch.int32),
            (1.0, torch.float32),
            (1.0 + 1.0j, torch.complex64),
        ]
        for value, dtype in cases:
            input_tensor = torch.tensor(value, dtype=dtype, device="npu")
            with self.subTest(dtype=dtype):
                torch._assert_async(input_tensor)
                torch._assert_async(input_tensor, "value is nonzero")
                torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    def test_assert_async_single_element_view(self):
        input_tensor = torch.arange(5, dtype=torch.int32, device="npu")[2:3]
        self.assertEqual(input_tensor.numel(), 1)
        self.assertEqual(input_tensor.storage_offset(), 2)
        torch._assert_async(input_tensor)
        torch.npu.synchronize()

    @SupportedDevices(["Ascend950"])
    def test_assert_async_failure_in_subprocess(self):
        cases = [
            ("False", "torch.bool", "torch._assert_async(tensor)"),
            ("0", "torch.int32", 'torch._assert_async(tensor, "assert_async failure")'),
            ("0.0", "torch.float32", 'torch._assert_async(tensor, "assert_async failure")'),
            ("-0.0", "torch.float32", 'torch._assert_async(tensor, "assert_async failure")'),
        ]
        for value, dtype, assert_call in cases:
            script = FAILURE_SCRIPT.format(value=value, dtype=dtype, assert_call=assert_call)
            with self.subTest(value=value, dtype=dtype):
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertIn("ASSERT_ENQUEUED", result.stdout)
                self.assertNotIn("ASSERT_COMPLETED", result.stdout)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    @SupportedDevices(["Ascend950"])
    def test_assert_async_invalid_numel(self):
        input_tensor = torch.tensor([1, 0], dtype=torch.int32, device="npu")
        with self.assertRaises(RuntimeError):
            torch._assert_async(input_tensor)


if __name__ == "__main__":
    run_tests()
