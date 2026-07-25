import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

import torch
from prettytable import PrettyTable
from utils import compute_error_balance, func_log_wrapper


class SingleBenchmarkCompareStandard:
    def __init__(self, high_precision=True):
        self.high_precision = high_precision
        self.small_value = 1.0
        self.error_thd = {torch.float16: [2 ** -11, 2 ** -7],
                          torch.bfloat16: [2 ** -8, 2 ** -6],
                          torch.float32: [2 ** -14, 2 ** -11],
                          torch.float64: [2 ** -14, 2 ** -11]}
        self.eb_thd = {torch.float16: 2 ** -10,
                       torch.bfloat16: 2 ** -7,
                       torch.float32: 2 ** -14,
                       torch.float64: 2 ** -14}

    def get_error_thd(self, dtype):
        if dtype in self.error_thd.keys():
            if dtype == torch.float64:
                logging.warning("The output data of fp64 uses the same standard as fp32.")
            return self.error_thd.get(dtype)[0] if self.high_precision else self.error_thd.get(dtype)[1]
        logging.error(
            "single benchmark compare only supports floating point "
            "in fp16, bf16, fp32."
        )
        return None

    def get_eb_thd(self, dtype):
        if dtype in self.eb_thd.keys():
            return self.eb_thd.get(dtype)
        return None


class SingleBenchmarkAccuracyResult:
    def __init__(
        self,
        result=True,
        error_balance=None,
        max_abs_diff=None,
        max_abs_idx=None,
        max_rel_diff=None,
        max_rel_idx=None,
    ):
        self.result = result
        self.error_balance = error_balance
        self.max_abs_diff = max_abs_diff
        self.max_abs_idx = max_abs_idx
        self.max_rel_diff = max_rel_diff
        self.max_rel_idx = max_rel_idx

    def get_result(self, eb_thd, error_thd):
        if (
            self.error_balance > eb_thd
            or self.max_abs_diff > error_thd
            or self.max_rel_diff > error_thd
        ):
            self.result = False
        else:
            self.result = True


class SingleBenchmarkAccuracyCompare:
    @classmethod
    @func_log_wrapper()
    def check_output_size(
        cls,
        actual_res: torch.Tensor,
        golden_res: torch.Tensor,
    ):
        acc_result = None
        if actual_res.numel() == 0 and golden_res.numel() == 0:
            info = (
                "The npu_output is [],and it is same as bm_output, "
                "the result of data_compare is Pass"
            )
            logging.debug(info)
            acc_result = SingleBenchmarkAccuracyResult(result=True)

        if actual_res.size() != golden_res.size():
            error_info = (
                f"the size of npu output[{actual_res.size()}] and "
                f"benchmark[{golden_res.size()}] is not equal."
            )

            logging.error(error_info)
            acc_result = SingleBenchmarkAccuracyResult(result=False)
        return acc_result

    @classmethod
    @func_log_wrapper()
    def check_actual_invalid_value(
        cls,
        actual_res: torch.Tensor,
    ):
        has_nan = torch.isnan(actual_res).any()
        has_inf = torch.isinf(actual_res).any()
        return has_nan or has_inf

    @classmethod
    @func_log_wrapper()
    def check_golden_invalid_value(
        cls,
        golden_res: torch.Tensor,
    ):
        has_nan = torch.isnan(golden_res).any()
        has_inf = torch.isinf(golden_res).any()
        return has_nan or has_inf

    @classmethod
    @func_log_wrapper()
    def precision_compare_for_case(
        cls,
        actual_res: torch.Tensor,
        golden_res: torch.Tensor,
        benchmark_standard: SingleBenchmarkCompareStandard,
    ):
        error_thd = None
        eb_thd = None
        acc_result = cls.check_output_size(actual_res, golden_res)

        if acc_result:
            failed_info = "比对数据的shape不一致"
            return acc_result, error_thd, eb_thd, failed_info

        if cls.check_golden_invalid_value(golden_res):
            logging.info("The benchmark result contains nan/inf value.")
            failed_info = "标杆结果存在nan值或inf值，依照单标杆标准该用例通过"
            acc_result = SingleBenchmarkAccuracyResult(result=True)
            return acc_result, error_thd, eb_thd, failed_info

        if cls.check_actual_invalid_value(actual_res):
            logging.error("The NPU result contains nan/inf value.")
            failed_info = "NPU结果存在nan值或inf值，依照单标杆标准该用例不通过"
            acc_result = SingleBenchmarkAccuracyResult(result=False)
            return acc_result, error_thd, eb_thd, failed_info

        data_type = actual_res.dtype
        if data_type not in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ]:
            acc_result = cls.compute_binary_diff(actual_res, golden_res)
        else:
            error_thd = benchmark_standard.get_error_thd(data_type)
            eb_thd = benchmark_standard.get_eb_thd(data_type)
            if error_thd is None:
                logging.error(
                    "single benchmark not support the comparison of %s", str(data_type)
                )
                acc_result = SingleBenchmarkAccuracyResult(result=False)
            else:
                if actual_res.dtype in [
                    torch.float16,
                    torch.bfloat16,
                ] and golden_res.dtype in [torch.float32]:
                    actual_res = actual_res.to(torch.float32)
                error_balance = compute_error_balance(
                    actual_res, golden_res, benchmark_standard.small_value
                )
                max_abs_diff, max_abs_idx = cls.compute_abs_diff(
                    actual_res, golden_res, error_thd, benchmark_standard
                )
                max_rel_diff, max_rel_idx = cls.compute_rel_diff(
                    actual_res, golden_res, error_thd, benchmark_standard
                )
                acc_result = SingleBenchmarkAccuracyResult(
                    error_balance=error_balance,
                    max_abs_diff=max_abs_diff,
                    max_abs_idx=max_abs_idx,
                    max_rel_diff=max_rel_diff,
                    max_rel_idx=max_rel_idx,
                )
                acc_result.get_result(eb_thd, error_thd)
        return acc_result, error_thd, eb_thd, None

    @classmethod
    @func_log_wrapper()
    def compute_binary_diff(
        cls,
        actual_res: torch.Tensor,
        golden_res: torch.Tensor,
    ):
        result = torch.equal(actual_res, golden_res)
        if result:
            print("二进制精度比对通过，无需单标杆比对法验证")
        return SingleBenchmarkAccuracyResult(
            result=result, max_abs_diff=0, max_rel_diff=0, error_balance=0
        )

    @classmethod
    @func_log_wrapper()
    def compute_abs_diff(
        cls,
        actual_res: torch.Tensor,
        golden_res: torch.Tensor,
        error_thd: float,
        benchmark_standard: SingleBenchmarkCompareStandard,
    ):
        max_abs_diff = 0
        max_abs_idx = None

        ones = torch.ones_like(actual_res)
        zeros = torch.zeros_like(actual_res)
        diff_value = torch.subtract(actual_res, golden_res)
        diff_abs = torch.abs(diff_value)

        abs_mask_idx = torch.where(
            torch.abs(golden_res) < benchmark_standard.small_value, ones, zeros
        )
        abs_err_idx = torch.where(diff_abs > error_thd, ones, zeros)
        abs_err_idx = abs_err_idx * abs_mask_idx
        abs_err = diff_abs[torch.where(abs_err_idx == 1)]

        if len(abs_err) > 0:
            err_for_max = torch.where(abs_err_idx == 1, diff_abs, zeros)
            logging.debug("err_for_max for abs %s", err_for_max)
            max_abs_idx = torch.argmax(err_for_max)
            max_abs_diff = diff_abs[max_abs_idx]
        elif torch.sum(abs_mask_idx) > 0:
            err_for_max = torch.where(abs_mask_idx == 1, diff_abs, zeros)
            logging.debug("err_for_max for abs %s", err_for_max)
            max_abs_idx = torch.argmax(err_for_max)
            if err_for_max.max() != 0:
                max_abs_diff = diff_abs[max_abs_idx]
        return (
            float(max_abs_diff),
            int(max_abs_idx) if torch.is_tensor(max_abs_idx) else max_abs_idx,
        )

    @classmethod
    @func_log_wrapper()
    def compute_rel_diff(
        cls,
        actual_res: torch.Tensor,
        golden_res: torch.Tensor,
        error_thd: float,
        benchmark_standard: SingleBenchmarkCompareStandard,
    ):
        max_rel_diff = 0
        max_rel_idx = None

        ones = torch.ones_like(actual_res)
        zeros = torch.zeros_like(actual_res)
        diff_value = torch.subtract(actual_res, golden_res)
        diff_abs = torch.abs(diff_value)

        rel_mask_idx = torch.where(
            torch.abs(golden_res) >= benchmark_standard.small_value, ones, zeros
        )
        rel_err = diff_abs / torch.abs(golden_res)
        diff_rel = rel_err
        rel_err_idx = torch.where(rel_err > error_thd, ones, zeros)
        rel_err_idx = rel_err_idx * rel_mask_idx
        rel_err = rel_err[torch.where(rel_err_idx == 1)]
        if len(rel_err) > 0:
            err_for_max = torch.where(rel_err_idx == 1, diff_rel, zeros)
            logging.debug("err_for_max for rel %s", err_for_max)
            max_rel_idx = torch.argmax(err_for_max)
            max_rel_diff = diff_rel[max_rel_idx]
        elif torch.sum(rel_mask_idx) > 0:
            err_for_max = torch.where(rel_mask_idx == 1, diff_rel, zeros)
            logging.debug("err_for_max for rel %s", err_for_max)
            max_rel_idx = torch.argmax(err_for_max)
            if torch.sum(err_for_max) != 0:
                max_rel_diff = diff_rel[max_rel_idx]
        return (
            float(max_rel_diff),
            int(max_rel_idx) if torch.is_tensor(max_rel_idx) else max_rel_idx,
        )


class SingleBenchSummary:
    def __init__(
        self,
        precision_result: SingleBenchmarkAccuracyResult,
        error_thd: float,
        eb_thd: float,
        failed_info: str,
    ):
        self.result = precision_result.result
        self.error_balance = precision_result.error_balance
        self.max_abs_diff = precision_result.max_abs_diff
        self.max_abs_idx = precision_result.max_abs_idx
        self.max_rel_diff = precision_result.max_rel_diff
        self.max_rel_idx = precision_result.max_rel_idx
        self.eb_thd = eb_thd
        self.error_thd = error_thd
        self.failed_info = failed_info

    def get_check_result(self):
        if self.result:
            return "PASS"
        else:
            return "FAILED"

    def get_result_msg(self):
        result_str = ""
        if self.failed_info:
            return self.failed_info

        if self.result:
            result_str += "误差均衡性EB: %s <= 阈值%s\n" % (self.error_balance, self.eb_thd)
            result_str += "最大绝对误差: %s <= 阈值%s\n" % (self.max_abs_diff, self.error_thd)
            result_str += "最大相对误差: %s <= 阈值%s\n" % (self.max_rel_diff, self.error_thd)
        else:
            if self.error_balance > self.eb_thd:
                result_str += "误差均衡性EB超过阈值%s: EB = %s\n" % (
                    self.eb_thd,
                    self.error_balance,
                )
            if self.max_abs_diff > self.error_thd:
                result_str += "小值域最大绝对误差超过阈值%s: idx = %s, 绝对误差 = %s\n" % (
                    self.error_thd,
                    self.max_abs_idx,
                    self.max_abs_diff,
                )
            if self.max_rel_diff > self.error_thd:
                result_str += "大值域最大相对误差超过阈值%s: idx = %s, 相对误差 = %s\n" % (
                    self.error_thd,
                    self.max_rel_idx,
                    self.max_rel_diff,
                )
        return result_str

    def print_detail_table(self):
        table = PrettyTable()
        table.title = "Single Benchmark Metrics Info"
        table.field_names = ["Index", "Result", "Threshold"]
        table.add_row(["error_balance", self.error_balance, self.eb_thd])
        table.add_row(["max_abs_diff", self.max_abs_diff, self.error_thd])
        table.add_row(["max_abs_idx", self.max_abs_idx, "-"])
        table.add_row(["max_rel_diff", self.max_rel_diff, self.error_thd])
        table.add_row(["max_rel_idx", self.max_rel_idx, "-"])

        print(table)


def single_benchmark_compare(actual: torch.Tensor, golden: torch.Tensor, high_precision: bool = True):
    actual_res = actual.flatten()
    golden_res = golden.flatten()
    benchmark_standard = SingleBenchmarkCompareStandard(high_precision=high_precision)
    (
        precision_result,
        error_thd,
        eb_thd,
        failed_info,
    ) = SingleBenchmarkAccuracyCompare.precision_compare_for_case(
        actual_res, golden_res, benchmark_standard
    )

    summary = SingleBenchSummary(precision_result, error_thd, eb_thd, failed_info)
    summary.print_detail_table()
    return summary


if __name__ == "__main__":
    actual = torch.rand(4, 4, dtype=torch.float16)
    golden = torch.rand(4, 4, dtype=torch.float32)

    ret = single_benchmark_compare(actual, golden, high_precision=True)
    print("单标杆精度比对结果: ", ret.get_check_result())
    print(ret.get_result_msg())
