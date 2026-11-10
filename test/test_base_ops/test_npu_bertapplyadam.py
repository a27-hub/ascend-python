import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestNpuBertApplyAdam(TestCase):
    def test_npu_bert_apply_adam_updates_states_in_place(self):
        # beta1=beta2=0.5 keeps the expected moments exact in both dtypes.
        # With grad=[2, -4], m=[0, 0], v=[4, 16], the update is [0.4, -4/9].
        cases = (
            # mode, step_size, weight_decay, max_norm, global_norm, var, m, v
            (0, None, 0.0, -1.0, 0.0, [1.9, -35.0 / 9.0], [1.0, -2.0], [4.0, 16.0]),
            (0, 0.125, 0.0, -1.0, 0.0, [1.9, -35.0 / 9.0], [1.0, -2.0], [4.0, 16.0]),
            (0, None, 0.5, -1.0, 0.0, [1.65, -61.0 / 18.0], [1.0, -2.0], [4.0, 16.0]),
            (0, None, 0.0, 2.0, 2.0,
             [2.0 - 0.125 / (2.5 ** 0.5 + 0.5), -4.0 + 0.25 / (10.0 ** 0.5 + 0.5)],
             [0.5, -1.0], [2.5, 10.0]),
            (1, 0.125, 0.5, 1.0, 2.0, [1.7, -31.0 / 9.0], [1.0, -2.0], [4.0, 16.0]),
            (2, 0.125, 0.0, -1.0, 0.0, [1.95, -71.0 / 18.0], [1.0, -2.0], [4.0, 16.0]),
        )
        for dtype in (torch.float16, torch.float32):
            for mode, step_size, decay, max_norm, global_norm, var_expected, m_expected, v_expected in cases:
                with self.subTest(dtype=dtype, mode=mode, decay=decay, max_norm=max_norm):
                    var = torch.tensor([2.0, -4.0], dtype=dtype, device="npu")
                    m = torch.tensor([0.0, 0.0], dtype=dtype, device="npu")
                    v = torch.tensor([4.0, 16.0], dtype=dtype, device="npu")
                    grad = torch.tensor([2.0, -4.0], dtype=dtype, device="npu")
                    states = (var, m, v)
                    pointers = tuple(state.data_ptr() for state in states)

                    outputs = torch_npu.npu_bert_apply_adam(
                        0.25, 0.5, 0.5, 0.5, grad, max_norm, global_norm, decay,
                        step_size=step_size, adam_mode=mode, out=states)

                    for output, state, pointer, expected in zip(
                            outputs, states, pointers, (var_expected, m_expected, v_expected)):
                        self.assertEqual(output.data_ptr(), pointer)
                        self.assertEqual(state.data_ptr(), pointer)
                        self.assertEqual(output.dtype, dtype)
                        self.assertRtolEqual(state.cpu(), torch.tensor(expected, dtype=dtype))
                    self.assertRtolEqual(grad.cpu(), torch.tensor([2.0, -4.0], dtype=dtype))

    @SupportedDevices(["Ascend950"])
    def test_npu_bert_apply_adam_mbart_default_step_size(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                states = tuple(torch.tensor(values, dtype=dtype, device="npu")
                               for values in ([2.0, -4.0], [0.0, 0.0], [4.0, 16.0]))
                pointers = tuple(state.data_ptr() for state in states)
                grad = torch.tensor([2.0, -4.0], dtype=dtype, device="npu")
                outputs = torch_npu.npu_bert_apply_adam(
                    0.25, 0.5, 0.5, 0.5, grad, 1.0, 2.0, 0.5, adam_mode=1, out=states)
                expected = ([1.65, -61.0 / 18.0], [1.0, -2.0], [4.0, 16.0])
                for output, state, pointer, values in zip(outputs, states, pointers, expected):
                    self.assertEqual(output.data_ptr(), pointer)
                    self.assertEqual(state.data_ptr(), pointer)
                    self.assertRtolEqual(state.cpu(), torch.tensor(values, dtype=dtype))

    def test_npu_bert_apply_adam_requires_out(self):
        grad = torch.tensor([1.0], device="npu")
        with self.assertRaisesRegex(RuntimeError, "npu_bert_apply_adam is not implemented for Tensor"):
            torch_npu.npu_bert_apply_adam(0.25, 0.5, 0.5, 0.5, grad, -1.0, 0.0, 0.0)

    def test_npu_bert_apply_adam(self):
        seed = 3
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        m_in = torch.zeros(321538).npu()
        v_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()

        var_ans = torch.tensor([13.1862, -30.1250, -20.4954])
        m_ans = torch.tensor([0.0014, 0.0018, -0.0021])
        v_ans = torch.tensor([1.8999e-06, 3.2629e-06, 4.4347e-06])

        max_grad_norm = -1.
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.
        lr = 0.
        epsilon = 1e-06
        global_grad_norm = 0.

        var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(
            lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, out=(var_in, m_in, v_in))

        self.assertRtolEqual(var_out[:3].cpu(), var_ans)
        self.assertRtolEqual(m_out[:3].cpu(), m_ans)
        self.assertRtolEqual(v_out[:3].cpu(), v_ans)

    def test_npu_bert_apply_adam_out(self):
        seed = 3
        torch.npu.manual_seed(seed)
        torch.manual_seed(seed)
        torch.npu.manual_seed_all(seed)

        var_in = torch.rand(321538).uniform_(-32., 21.).npu()
        v_in = torch.zeros(321538).npu()
        m_in = torch.zeros(321538).npu()
        grad = torch.rand(321538).uniform_(-0.05, 0.03).npu()

        var_ans = torch.tensor([13.1862, -30.1250, -20.4954])
        m_ans = torch.tensor([0.0014, 0.0018, -0.0021])
        v_ans = torch.tensor([1.8999e-06, 3.2629e-06, 4.4347e-06])

        max_grad_norm = -1.
        beta1 = 0.9
        beta2 = 0.999
        weight_decay = 0.
        lr = 5e-05
        epsilon = 1e-08
        global_grad_norm = 0.
        step_size = 0
        adam_mode = 1

        var_out, m_out, v_out = torch_npu.npu_bert_apply_adam(
            lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm,
            weight_decay, step_size, adam_mode, out=(var_in, m_in, v_in))

        self.assertRtolEqual(var_out[:3].cpu(), var_ans)
        self.assertRtolEqual(m_out[:3].cpu(), m_ans)
        self.assertRtolEqual(v_out[:3].cpu(), v_ans)


if __name__ == "__main__":
    run_tests()
