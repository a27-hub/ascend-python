import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSilentCheck(TestCase):
    def test_silent_check_bf16(self):
        input_tensor = torch.randn(2, 3, dtype=torch.bfloat16, device=torch.device("npu"))
        val = torch.norm(input_tensor)
        pre_val = torch.tensor(0).float().npu()
        max_val = torch.tensor(-10 ** 10).float().npu()
        min_val = torch.tensor(10 ** 10).float().npu()
        step_tensor = torch.tensor(0, dtype=torch.int32).npu()
        min_step = 7
        upper_thresh_1 = 1000000
        upper_thresh_2 = 10000
        sigma_thresh_1 = 100000
        sigma_thresh_2 = 5000

        torch_npu._npu_silent_check(input_tensor, val, pre_val, max_val, min_val, step_tensor, min_step, upper_thresh_1,
                                    sigma_thresh_1, upper_thresh_2, sigma_thresh_2)

    def test_silent_check_exception(self):
        input_tensor = torch.full((2, 3), float('nan'), device=torch.device("npu"))
        val = torch.norm(input_tensor)
        pre_val = torch.tensor(0).float().npu()
        max_val = torch.tensor(-10 ** 10).float().npu()
        min_val = torch.tensor(10 ** 10).float().npu()
        step_tensor = torch.tensor(0, dtype=torch.int32).npu()
        min_step = 7
        upper_thresh_1 = 1000000
        upper_thresh_2 = 10000
        sigma_thresh_1 = 100000
        sigma_thresh_2 = 5000
        
        with self.assertRaises(RuntimeError) as cm:
            torch_npu._npu_silent_check(input_tensor, val, pre_val, max_val, min_val, step_tensor, min_step, upper_thresh_1,
                                        sigma_thresh_1, upper_thresh_2, sigma_thresh_2)
            torch.npu.synchronize()
        exception = cm.exception
        self.assertTrue("errcode:7, msg:convergence sensitivity feature abnormal" in str(exception))


if __name__ == "__main__":
    run_tests()
