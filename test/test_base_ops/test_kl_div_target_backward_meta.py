import torch
import torch_npu  # noqa: F401

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestKlDivTargetBackwardMeta(TestCase):
    @dtypes(torch.float16, torch.float32)
    @parametrize("reduction", [0, 1, 2])
    @parametrize("log_target", [False, True])
    @parametrize("layout", ["scalar", "contiguous", "transpose"])
    @parametrize("backward", ["input", "target"])
    def test_meta_matches_native(self, device, dtype, reduction, log_target, layout, backward):
        target = torch.full(() if layout == "scalar" else (2, 3), 0.5, device=device, dtype=dtype)
        if layout == "transpose":
            target = target.t()
        input = torch.full_like(target, -0.5)
        grad = torch.ones_like(target) if reduction == 0 else target.new_ones(())
        op = torch.ops.npu.kl_div_backward if backward == "input" else torch.ops.npu.kl_div_target_backward
        actual = op(grad, input, target, reduction, log_target=log_target)
        with FakeTensorMode() as mode:
            fake = op(
                mode.from_tensor(grad), mode.from_tensor(input), mode.from_tensor(target),
                reduction, log_target=log_target,
            )
        self.assertEqual(fake.shape, actual.shape)
        self.assertEqual(fake.dtype, actual.dtype)
        self.assertEqual(fake.device, actual.device)
        self.assertEqual(fake.stride(), actual.stride())
        self.assertIsNone(fake._base)


instantiate_device_type_tests(TestKlDivTargetBackwardMeta, globals(), only_for="npu")

if __name__ == "__main__":
    run_tests()
