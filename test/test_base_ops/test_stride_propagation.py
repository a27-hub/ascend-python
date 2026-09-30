# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate torch.Tensor.stride and eq/add/exp output layout on NPU.

The community stride test covers Tensor operands. These regressions also
cover Scalar operands, default floating dtype, and output alias validation.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


class TestStridePropagation(TestCase):
    def _input(self, device, sliced=False):
        tensor = torch.arange(24, dtype=torch.float32, device=device)
        tensor = tensor.reshape(2, 3, 4) / 24
        return tensor[:, :, ::2].transpose(1, 2) if sliced else tensor.transpose(1, 2)

    def _assert_output(self, actual, expected):
        self.assertEqual(actual.device.type, "npu")
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

    def _op(self, name, tensor, scalar=False, out=None):
        kwargs = {} if out is None else {"out": out}
        if name == "exp":
            return torch.exp(tensor, **kwargs)
        other = 0.25 if scalar else tensor
        return getattr(torch, name)(tensor, other, **kwargs)

    def test_tensor_layout(self):
        for name in ("eq", "add", "exp"):
            for sliced in (False, True):
                cpu = self._input("cpu", sliced)
                npu = self._input("npu", sliced)
                with self.subTest(op=name, sliced=sliced):
                    expected = self._op(name, cpu)
                    self._assert_output(self._op(name, npu), expected)
                    cpu_out = torch.empty(0, dtype=expected.dtype)
                    npu_out = torch.empty(0, dtype=expected.dtype, device="npu")
                    self.assertIs(self._op(name, cpu, out=cpu_out), cpu_out)
                    self.assertIs(self._op(name, npu, out=npu_out), npu_out)
                    self._assert_output(npu_out, cpu_out)

    def test_scalar_layout(self):
        for name in ("eq", "add"):
            for sliced in (False, True):
                cpu = self._input("cpu", sliced)
                npu = self._input("npu", sliced)
                with self.subTest(op=name, sliced=sliced):
                    expected = self._op(name, cpu, scalar=True)
                    self._assert_output(self._op(name, npu, scalar=True), expected)
                    if name == "add":
                        self._assert_output(
                            torch.ops.aten.add.Scalar(npu, 0.25),
                            torch.ops.aten.add.Scalar(cpu, 0.25),
                        )
                    cpu_out = torch.empty(0, dtype=expected.dtype)
                    npu_out = torch.empty(0, dtype=expected.dtype, device="npu")
                    self.assertIs(self._op(name, npu, scalar=True, out=npu_out), npu_out)
                    self._op(name, cpu, scalar=True, out=cpu_out)
                    self._assert_output(npu_out, cpu_out)

    def test_out_layout(self):
        for name in ("eq", "add", "exp"):
            for transposed in (False, True):
                cpu = self._input("cpu")
                npu = self._input("npu")
                dtype = torch.bool if name == "eq" else torch.float32
                shape = (2, 3, 4) if transposed else (2, 4, 3)
                cpu_out = torch.empty(shape, dtype=dtype)
                npu_out = torch.empty(shape, dtype=dtype, device="npu")
                if transposed:
                    cpu_out = cpu_out.transpose(1, 2)
                    npu_out = npu_out.transpose(1, 2)
                original_stride = npu_out.stride()
                with self.subTest(op=name, transposed=transposed):
                    self._op(name, cpu, out=cpu_out)
                    self.assertIs(self._op(name, npu, out=npu_out), npu_out)
                    self.assertEqual(npu_out.stride(), original_stride)
                    self._assert_output(npu_out, cpu_out)

    def test_exp_default_dtype(self):
        original_dtype = torch.get_default_dtype()
        try:
            for default_dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(default_dtype)
                for input_dtype in (torch.int32, torch.bool):
                    cpu = torch.arange(6, dtype=torch.int32).reshape(2, 3).t().to(input_dtype)
                    npu = torch.arange(6, dtype=torch.int32, device="npu")
                    npu = npu.reshape(2, 3).t().to(input_dtype)
                    with self.subTest(default_dtype=default_dtype, input_dtype=input_dtype):
                        self._assert_output(torch.exp(npu), torch.exp(cpu))
                        cpu_out = torch.empty(0, dtype=default_dtype)
                        npu_out = torch.empty(0, dtype=default_dtype, device="npu")
                        torch.exp(cpu, out=cpu_out)
                        self.assertIs(torch.exp(npu, out=npu_out), npu_out)
                        self._assert_output(npu_out, cpu_out)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_add_invalid_alpha(self):
        cases = ((torch.int32, 0.5), (torch.float32, True), (torch.float32, 1j))
        for dtype, alpha in cases:
            for device in ("cpu", "npu"):
                tensor = torch.ones(2, 3, dtype=dtype, device=device).t()
                out = torch.empty(0, dtype=dtype, device=device)
                with self.subTest(dtype=dtype, alpha=alpha, device=device):
                    with self.assertRaises(RuntimeError):
                        torch.add(tensor, tensor, alpha=alpha)
                    with self.assertRaises(RuntimeError):
                        torch.add(tensor, tensor, alpha=alpha, out=out)

    def test_exp_out_overlap(self):
        for device in ("cpu", "npu"):
            storage = torch.arange(6, dtype=torch.int32, device=device)
            tensor = storage[:4]
            out = storage[1:5].view(torch.float32)
            with self.subTest(device=device):
                with self.assertRaisesRegex(RuntimeError, "overlap|single memory location"):
                    torch.exp(tensor, out=out)

    def test_stride_arguments(self):
        cpu = self._input("cpu")
        npu = self._input("npu")
        self.assertEqual(npu.stride(), cpu.stride())
        for dim in range(-3, 3):
            self.assertEqual(npu.stride(dim), cpu.stride(dim))
        for tensor in (cpu, npu):
            for dim in (-4, 3):
                with self.assertRaises(IndexError):
                    tensor.stride(dim)


if __name__ == "__main__":
    run_tests()
