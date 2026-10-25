import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


class TestClampNarrowInteger(TestCase):
    """Exact integer checks independent of the disabled legacy TestClamp suite."""

    @staticmethod
    def _input(dtype, size):
        limits = torch.iinfo(dtype)
        # Cycle over the whole dtype range, including extrema and bound neighbours.
        values = torch.arange(size, dtype=torch.int32)
        values = values.remainder_(limits.max - limits.min + 1).add_(limits.min).to(dtype)
        low, high = (31, 191) if dtype == torch.uint8 else (-31, 31)
        values[:8] = torch.tensor(
            [limits.min, low - 1, low, low + 1, high - 1, high, high + 1, limits.max], dtype=dtype
        )
        return values

    def _check_clamp(self, inplace):
        for dtype in (torch.int8, torch.uint8, torch.int16):
            low, high = (31, 191) if dtype == torch.uint8 else (-31, 31)
            limits = torch.iinfo(dtype)
            for size in (1024, 1024 * 1024, 8 * 1024 * 1024):
                original = self._input(dtype, size)
                for minimum, maximum in ((low, high), (limits.min, limits.max), (0, 0)):
                    with self.subTest(dtype=dtype, size=size, bounds=(minimum, maximum), inplace=inplace):
                        # Use an independent CPU expression rather than a second clamp call.
                        expected = torch.where(original < minimum, minimum, original)
                        expected = torch.where(expected > maximum, maximum, expected)
                        value = original.npu()
                        pointer = value.data_ptr()
                        result = value.clamp_(minimum, maximum) if inplace else value.clamp(minimum, maximum)
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(result.shape, original.shape)
                        self.assertTrue(torch.equal(result.cpu(), expected))
                        if inplace:
                            self.assertEqual(result.data_ptr(), pointer)
                        else:
                            self.assertTrue(torch.equal(value.cpu(), original))

    def test_clamp(self):
        self._check_clamp(inplace=False)

    def test_clamp_inplace(self):
        self._check_clamp(inplace=True)


if __name__ == '__main__':
    run_tests()
