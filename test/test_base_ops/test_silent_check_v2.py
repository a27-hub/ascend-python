import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSilentCheck(TestCase):
    def test_silent_check_v2_bf16(self):
        input_tensor = torch.randn(2, 3, dtype=torch.bfloat16, device=torch.device("npu"))
        val = torch.norm(input_tensor)
        check_tensor = torch.zeros(3, dtype=torch.float, device=torch.device("npu"))
        step_tensor = torch.zeros(1, dtype=torch.int64, device=torch.device("npu"))
        min_step = 100
        upper_thresh_1 = 1000000
        upper_thresh_2 = 10000
        sigma_thresh_1 = 100000
        sigma_thresh_2 = 5000
        asd_detect = 2

        torch_npu._npu_silent_check_v2(val, input_tensor, check_tensor, step_tensor, min_step, upper_thresh_1,
                                       sigma_thresh_1, upper_thresh_2, sigma_thresh_2, asd_detect)

    def test_silent_check_v2_fp32(self):
        input_tensor = torch.randn(2, 3, dtype=torch.float, device=torch.device("npu"))
        val = torch.norm(input_tensor)
        check_tensor = torch.zeros(3, dtype=torch.float, device=torch.device("npu"))
        step_tensor = torch.zeros(1, dtype=torch.int64, device=torch.device("npu"))
        min_step = 100
        upper_thresh_1 = 1000000
        upper_thresh_2 = 10000
        sigma_thresh_1 = 100000
        sigma_thresh_2 = 5000
        asd_detect = 2

        torch_npu._npu_silent_check_v2(val, input_tensor, check_tensor, step_tensor, min_step, upper_thresh_1,
                                       sigma_thresh_1, upper_thresh_2, sigma_thresh_2, asd_detect)

    def test_silent_check_v2_exception(self):
        input_tensor = torch.full((2, 3), float('nan'), device=torch.device("npu"))
        val = torch.norm(input_tensor)
        check_tensor = torch.zeros(3, dtype=torch.float, device=torch.device("npu"))
        step_tensor = torch.zeros(1, dtype=torch.int64, device=torch.device("npu"))
        min_step = 100
        upper_thresh_1 = 1000000
        upper_thresh_2 = 10000
        sigma_thresh_1 = 100000
        sigma_thresh_2 = 5000
        asd_detect = 2

        with self.assertRaises(RuntimeError) as cm:
            torch_npu._npu_silent_check_v2(val, input_tensor, check_tensor, step_tensor, min_step, upper_thresh_1,
                                        sigma_thresh_1, upper_thresh_2, sigma_thresh_2, asd_detect)
            torch.npu.synchronize()
        exception = cm.exception
        self.assertTrue("errcode:7, msg:convergence sensitivity feature abnormal" in str(exception))


if __name__ == "__main__":
    run_tests()
