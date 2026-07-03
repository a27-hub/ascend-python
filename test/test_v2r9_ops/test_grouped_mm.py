import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGroupedMm(TestCase):
    def _manual_grouped_mm_2d_2d(self, mat_a, mat_b, offs):
        outputs = []
        group_start = 0
        for group_end in offs.cpu().tolist():
            outputs.append(mat_a[:, group_start:group_end].mm(mat_b[group_start:group_end, :]))
            group_start = group_end
        return torch.stack(outputs)

    def _padding_indices(self, out):
        indices = []
        last_dim = out.dim() - 1
        storage_numel = out.untyped_storage().nbytes() // out.element_size()
        for prefix in torch.cartesian_prod(*[torch.arange(size) for size in out.shape[:-1]]):
            base = sum(int(prefix[dim]) * out.stride(dim) for dim in range(last_dim))
            row_end = min(base + out.stride(last_dim - 1), storage_numel)
            indices.extend(range(base + out.size(last_dim), row_end))
        return torch.tensor(indices, device=out.device)

    def test_grouped_mm_2d_2d_matches_cpu(self):
        torch.manual_seed(0)
        mat_a_cpu = torch.randn(3, 7, dtype=torch.float32)
        mat_b_cpu = torch.randn(7, 5, dtype=torch.float32)
        offs_cpu = torch.tensor([2, 7], dtype=torch.int32)

        expected = self._manual_grouped_mm_2d_2d(mat_a_cpu, mat_b_cpu, offs_cpu)
        result = torch._grouped_mm(mat_a_cpu.npu(), mat_b_cpu.npu(), offs_cpu.npu()).cpu()

        self.assertRtolEqual(expected.numpy(), result.numpy())

    def test_grouped_mm_deterministic_fills_padding_storage(self):
        torch.manual_seed(0)
        mat_a = torch.randn(3, 7, dtype=torch.float32, device="npu")
        mat_b = torch.randn(7, 5, dtype=torch.float32, device="npu")
        offs = torch.tensor([2, 7], dtype=torch.int32, device="npu")

        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            out = torch._grouped_mm(mat_a, mat_b, offs)
            padding = out.as_strided((out.untyped_storage().nbytes() // out.element_size(),), (1,), 0)[
                self._padding_indices(out)
            ]
            self.assertTrue(torch.isnan(padding).all().cpu().item())
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
            torch.utils.deterministic.fill_uninitialized_memory = old_fill


if __name__ == "__main__":
    run_tests()
