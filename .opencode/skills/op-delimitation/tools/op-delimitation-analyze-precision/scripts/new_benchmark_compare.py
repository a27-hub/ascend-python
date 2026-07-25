import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from functools import reduce
from typing import Union, List

import numpy as np
from prettytable import PrettyTable
from utils import func_log_wrapper


class PrecisionCheckResult:
    SUCCESS = 0
    WARNING = 1
    ERROR = 2

    def __init__(self, precision_check_standard):
        self.precision_check_standard = precision_check_standard
        self.golden_avg = 0
        self.npu_stat_info: Union[None, PrecisionStatInfo] = None
        self.benchmark_stat_info: Union[None, PrecisionStatInfo] = None
        self.total_cnt = 0
        self.inf_nan_cnt = 0
        self.inf_nan_err_cnt = 0
        self.inf_nan_err_ele = []
        self.inf_nan_warn_cnt = 0
        self.inf_nan_warn_ele = []
        self.max_re_ratio = 1
        self.eb_difference = 0
        self.avg_re_ratio = 1
        self.rmse_ratio = 1
        self.max_ulp_ratio = 1
        self.min_ulp_ratio = 1
        self.avg_ulp_ratio = 1
        self.max_abs_ulp_ratio = 1
        self.avg_abs_ulp_ratio = 1
        self.ulp_err_ratio = 1
        self.ae_err_cnt = 0
        self.ae_warn_cnt = 0
        self.ae_err_ele = []
        self.ae_warn_ele = []
        self.ae_err_ele_str = ""
        self.ae_warn_ele_str = ""

    def get_check_result_str(self):
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return "PASS"
        if self.check_result == PrecisionCheckResult.WARNING:
            return "WARNING"
        if self.check_result == PrecisionCheckResult.ERROR:
            return "ERROR"
        return "Unknown"

    def _calc_ratio(self, x: float, y: float, default_value: float = 1.0):
        if y == 0:
            if x == 0:
                return 1.0
            else:
                return default_value
        else:
            return x / y

    def do_summary(self):
        npu_max_re = self.npu_stat_info.re_distribute_info.max_re
        benchmark_max_re = self.benchmark_stat_info.re_distribute_info.max_re
        self.max_re_ratio = self._calc_ratio(
            npu_max_re, benchmark_max_re, 10000.0
        )
        npu_eb = self.npu_stat_info.eb_info.eb
        benchmark_eb = self.benchmark_stat_info.eb_info.eb
        self.eb_difference = npu_eb - benchmark_eb
        npu_avg_re = self.npu_stat_info.re_distribute_info.avg_re
        benchmark_avg_re = self.benchmark_stat_info.re_distribute_info.avg_re
        self.avg_re_ratio = self._calc_ratio(npu_avg_re, benchmark_avg_re)
        self.rmse_ratio = self._calc_ratio(
            self.npu_stat_info.rmse, self.benchmark_stat_info.rmse, 10000.0
        )
        npu_ulp = self.npu_stat_info.ulp_distribute_info
        benchmark_ulp = self.benchmark_stat_info.ulp_distribute_info
        self.max_ulp_ratio = self._calc_ratio(
            npu_ulp.max_ulp, benchmark_ulp.max_ulp, 10000.0
        )
        self.min_ulp_ratio = self._calc_ratio(
            npu_ulp.min_ulp, benchmark_ulp.min_ulp, 10000.0
        )
        self.avg_ulp_ratio = self._calc_ratio(
            npu_ulp.avg_ulp, benchmark_ulp.avg_ulp, 1000.0
        )
        self.max_abs_ulp_ratio = self._calc_ratio(
            max(abs(npu_ulp.max_ulp), abs(npu_ulp.min_ulp)),
            max(abs(benchmark_ulp.max_ulp), abs(benchmark_ulp.min_ulp)),
            10000.0,
        )
        self.avg_abs_ulp_ratio = self._calc_ratio(
            npu_ulp.avg_abs_ulp, benchmark_ulp.avg_abs_ulp, 1000.0
        )
        self.ae_pass_ratio = self._calc_ratio(
            self.total_cnt - self.ae_err_cnt, self.total_cnt
        )
        self.ulp_err_ratio = self._calc_ratio(
            npu_ulp.abs_ulp_err_cnt, benchmark_ulp.abs_ulp_err_cnt
        )

        if (
            self.max_re_ratio > self.precision_check_standard.max_re_rtol
            or self.avg_re_ratio > self.precision_check_standard.avg_re_rtol
            or self.rmse_ratio > self.precision_check_standard.rmse_rtol
        ):
            self.check_result = PrecisionCheckResult.ERROR
        elif (
            self.max_re_ratio > 1
            or self.avg_re_ratio > 1
            or self.rmse_ratio > 1
        ):
            self.check_result = PrecisionCheckResult.WARNING
        else:
            self.check_result = PrecisionCheckResult.SUCCESS

    def print_summary_table(self):
        table = PrettyTable()
        table.title = "Summary Info"
        table.field_names = [
            "Index",
            "Actual / Golden",
            "Error Threshold",
            "Warn Threshold",
        ]

        table.add_row(
            [
                "max re ratio",
                self.max_re_ratio,
                self.precision_check_standard.max_re_rtol,
                1,
            ]
        )
        table.add_row(
            [
                "avg re ratio",
                self.avg_re_ratio,
                self.precision_check_standard.avg_re_rtol,
                1,
            ]
        )
        table.add_row(
            [
                "rmse ratio",
                self.rmse_ratio,
                self.precision_check_standard.rmse_rtol,
                1,
            ]
        )
        table.add_row(["eb difference", self.eb_difference, None, None])
        table.add_row(["max ulp ratio", self.max_ulp_ratio, None, None])
        table.add_row(["min ulp ratio", self.min_ulp_ratio, None, None])
        table.add_row(["avg ulp ratio", self.avg_ulp_ratio, None, None])
        table.add_row(["max abs ulp ratio", self.max_abs_ulp_ratio, None, None])
        table.add_row(["avg abs ulp ratio", self.avg_abs_ulp_ratio, None, None])
        table.add_row(["ulp err ratio", self.ulp_err_ratio, None, None])
        table.add_row(["ae pass ratio", self.ae_pass_ratio, None, None])

        print(table)

    def print_detail_table(self):
        self.npu_stat_info.print_detail_table(self.benchmark_stat_info)

    def get_result_msg(self):
        result_str = ""
        if self.check_result == PrecisionCheckResult.SUCCESS:
            return result_str
        if self.max_re_ratio > self.precision_check_standard.max_re_rtol:
            result_str += "ERROR: 最大相对误差差距超过阈值， %s\n" % self.max_re_ratio
        elif self.max_re_ratio > 1.0:
            result_str += "Warning: 最大相对误差差距超过阈值， %s\n" % self.max_re_ratio
        if self.avg_re_ratio > self.precision_check_standard.avg_re_rtol:
            result_str += "ERROR: 相对误差均值差距超过阈值， %s\n" % self.avg_re_ratio
        elif self.avg_re_ratio > 1.0:
            result_str += "Warning: 相对误差均值差距超过阈值， %s\n" % self.avg_re_ratio
        if self.rmse_ratio > self.precision_check_standard.rmse_rtol:
            result_str += "ERROR: 均方根误差差距超过阈值， %s\n" % self.rmse_ratio
        elif self.rmse_ratio > 1.0:
            result_str += "Warning: 均方根误差差距超过阈值， %s\n" % self.rmse_ratio
        # if self.ae_err_cnt > 0:
        #     result_str += "ERROR: 逐点绝对误差超过阈值， %s\n" % self.ae_err_ele_str
        # elif self.ae_warn_cnt > 0:
        #     result_str += "Warning: 逐点绝对误差超过阈值， %s\n" % self.ae_warn_ele_str
        return result_str


class REHistogramBins:
    # ULP_HISTOGRAM_BIN = [-np.inf, -10, -5, -1, 0, 1, 5, 10, np.inf]
    # ULP_HISTOGRAM_BIN_TITLES = ["小于\n-10", "-10~-5", "-5~-1", "-1~0", "0~1", "1~5", "5~10", "大于\n10"]

    @classmethod
    def get_histogram_bins(cls, dtype: str) -> List[float]:
        if "float16" == dtype:
            return [0, 0.001, 0.002, 0.005, 0.01, np.inf]
        if "float32" == dtype:
            return [0, 0.000001, 0.00001, 0.0001, 0.0005, np.inf]
        return [0, 0.004, 0.008, np.inf]  # for bf16

    @classmethod
    def get_histogram_bin_titles(cls, dtype: str) -> List[str]:
        if np.float16 == dtype or "float16" == dtype:
            return ["千分之\n0~1", "千分之\n1~2", "千分之\n2~5", "千分之\n5~10", "千分之\n>10"]
        if np.float32 == dtype or "float32" == dtype:
            return ["百万分\n0~1", "百万分\n1~10", "十万分\n1~10", "万分\n1~10", "万分\n>5"]
        return ["千分\n0~4", "千分\n4~8", "千分\n>8"]  # for bf16

    @classmethod
    def get_ulp_histogram_bin(cls, dtype: str) -> List[str]:
        if dtype in ["float16", "bfloat16"]:
            return [-np.inf, -10, -5, -1, 0, 1, 5, 10, np.inf]
        return [-np.inf, -128, -64, -32, 0, 32, 64, 128, np.inf]

    @classmethod
    def get_ulp_histogram_bin_titles(cls, dtype: str) -> List[str]:
        if dtype in ["float16", "bfloat16"]:
            return [
                "小于\n-10",
                "-10~-5",
                "-5~-1",
                "-1~0",
                "0~1",
                "1~5",
                "5~10",
                "大于\n10",
            ]
        return [
            "小于\n-128",
            "-128~-64",
            "-64~-32",
            "-32~0",
            "0~32",
            "32~64",
            "64~128",
            "大于\n128",
        ]


class SmallValueInfo:
    def __init__(
        self,
        small_value_thresh: float,
        small_value_count: Union[int, np.ndarray],
        small_value_err_count: Union[int, np.ndarray],
        total_cnt: Union[int, np.ndarray],
    ):
        self.small_value_thresh = small_value_thresh
        self.small_value_count = small_value_count
        self.small_value_err_count = small_value_err_count
        self.total_cnt = total_cnt
        self.small_value_ratio = (
            0 if total_cnt == 0 else small_value_count / total_cnt
        )
        self.small_value_err_ratio = (
            0 if total_cnt == 0 else small_value_err_count / total_cnt
        )

    def print_compare_with_peer(self, other):
        table = PrettyTable()
        table.title = "Small Value Info"
        table.field_names = ["index", "Actual", "Golden"]

        table.add_row(
            [
                "small value count",
                self.small_value_count,
                other.small_value_count,
            ]
        )
        table.add_row(
            [
                "small value err count",
                self.small_value_err_count,
                other.small_value_err_count,
            ]
        )

        print(table)


class REDistributeInfo:
    def __init__(
        self,
        histogram_stat: List[int],
        histogram_bins: List[float],
        max_re: float,
        max_re_value_info: List,
        avg_re: float,
    ):
        self.histogram_stat = histogram_stat
        self.histogram_bins = histogram_bins
        self.total_cnt = reduce(lambda x, y: x + y, histogram_stat)
        if self.total_cnt == 0:
            self.histogram_stat_ratio = [0 for _ in histogram_stat]
            self.histogram_stat_ratio_str = ["0" for _ in histogram_stat]
        else:
            self.histogram_stat_ratio = [
                (x / self.total_cnt) for x in histogram_stat
            ]
            self.histogram_stat_ratio_str = [
                "%.2f" % (x / self.total_cnt * 100) for x in histogram_stat
            ]
        self.max_re = float(max_re)
        self.avg_re = float(avg_re)
        self.max_re_value_info = max_re_value_info  # (npu_value, golden_value)

    def print_compare_with_peer(self, other):
        table = PrettyTable()
        table.title = "RE Distribute Info"
        table.field_names = ["index", "Actual", "Golden"]

        for i in range(len(self.histogram_stat)):
            table.add_row(
                [
                    f"{self.histogram_bins[i]}~{self.histogram_bins[i + 1]}",
                    f"{self.histogram_stat_ratio_str[i]}%({self.histogram_stat[i]})",
                    f"{other.histogram_stat_ratio_str[i]}%({other.histogram_stat[i]})",
                ]
            )

        table.add_row(["max re", self.max_re, other.max_re])
        table.add_row(
            [
                f"value of max re",
                self.max_re_value_info[0],
                other.max_re_value_info[0],
            ]
        )
        table.add_row(
            [
                "golden value of max re",
                self.max_re_value_info[1],
                other.max_re_value_info[1],
            ]
        )
        table.add_row(["avg re", self.avg_re, other.avg_re])

        print(table)


class EBInfo:
    def __init__(
        self,
        equal_cnt: Union[int, np.ndarray],
        larger_cnt: Union[int, np.ndarray],
        smaller_cnt: Union[int, np.ndarray],
        total_cnt: Union[int, np.ndarray],
    ):
        self.equal_cnt = equal_cnt
        self.larger_cnt = larger_cnt
        self.smaller_cnt = smaller_cnt
        self.total_cnt = total_cnt
        self.comparable_cnt = equal_cnt + larger_cnt + smaller_cnt
        self.eb = abs(self.larger_cnt - self.smaller_cnt) / (
            self.comparable_cnt + 0.000001
        )

    def __str__(self):
        return (
            "eb info: total count: %d, larger count: %d, smaller count: %d"
            % (self.total_cnt, self.larger_cnt, self.smaller_cnt)
        )

    def print_compare_with_peer(self, other):
        table = PrettyTable()
        table.title = "EB Info"
        table.field_names = ["index", "Actual", "Golden"]

        table.add_row(["total count", self.total_cnt, other.total_cnt])
        table.add_row(["larger count", self.larger_cnt, other.larger_cnt])
        table.add_row(["smaller count", self.smaller_cnt, other.smaller_cnt])


class ULPDistributeInfo:
    def __init__(
        self,
        histogram_bin,
        histogram_stat,
        total_cnt: Union[int, np.ndarray],
        abs_ulp_err_cnt: Union[int, np.ndarray],
        max_ulp: Union[float, np.ndarray],
        min_ulp: Union[float, np.ndarray],
        avg_ulp: Union[float, np.ndarray],
        avg_abs_ulp: Union[float, np.ndarray],
        max_value_info=None,
        min_val_info=None,
    ):
        self.total_cnt = total_cnt
        self.abs_ulp_err_cnt = abs_ulp_err_cnt
        self.max_ulp = np.floor(max_ulp * 2) / 2
        self.min_ulp = np.floor(min_ulp) / 2
        self.avg_ulp = np.floor(avg_ulp) / 2
        self.avg_abs_ulp = np.floor(avg_abs_ulp) / 2
        self.histogram_bin = histogram_bin
        self.histogram_stat = histogram_stat
        if self.total_cnt == 0:
            self.histogram_stat_ratio = [0 for _ in histogram_stat]
            self.histogram_stat_ratio_str = ["0" for _ in histogram_stat]
        else:
            self.histogram_stat_ratio = [
                (x / self.total_cnt) for x in histogram_stat
            ]
            self.histogram_stat_ratio_str = [
                "%.2f" % (x / self.total_cnt * 100) for x in histogram_stat
            ]
        self.max_val_info = max_value_info  # npu, golden
        self.min_val_info = min_val_info  # npu, golden

    def print_compare_with_peer(self, other):
        table = PrettyTable()
        table.title = "ULP Distribute Info"
        table.field_names = ["index", "Actual", "Golden"]

        for i in range(len(self.histogram_stat)):
            table.add_row(
                [
                    f"{self.histogram_bin[i]}~{self.histogram_bin[i + 1]}",
                    f"{self.histogram_stat_ratio_str[i]}%({self.histogram_stat[i]})",
                    f"{other.histogram_stat_ratio_str[i]}%({other.histogram_stat[i]})",
                ]
            )

        table.add_row(["max ulp", self.max_ulp, other.max_ulp])
        table.add_row(
            [f"value of max ulp", self.max_val_info[0], other.max_val_info[0],]
        )
        table.add_row(
            [
                "golden value of max re",
                self.max_val_info[1],
                other.max_val_info[1],
            ]
        )
        table.add_row(
            [f"value of min ulp", self.min_val_info[0], other.min_val_info[0],]
        )
        table.add_row(
            [
                "golden value of min re",
                self.min_val_info[1],
                other.min_val_info[1],
            ]
        )
        table.add_row(["avg ulp", self.avg_ulp, other.avg_ulp])
        table.add_row(["avg abs ulp", self.avg_abs_ulp, other.avg_abs_ulp])

        print(table)


class PrecisionStatInfo:
    def __init__(
        self,
        re_distribute_info: REDistributeInfo,
        small_value_info: SmallValueInfo,
        eb_info: EBInfo,
        rmse: float,
        ulp_distribute_info: ULPDistributeInfo,
    ):
        self.re_distribute_info = re_distribute_info
        self.small_value_info = small_value_info
        self.eb_info = eb_info
        self.rmse = rmse
        self.ulp_distribute_info = ulp_distribute_info

    def print_detail_table(self, other):
        self.re_distribute_info.print_compare_with_peer(
            other.re_distribute_info
        )
        self.small_value_info.print_compare_with_peer(other.small_value_info)
        self.eb_info.print_compare_with_peer(other.eb_info)
        self.ulp_distribute_info.print_compare_with_peer(
            other.ulp_distribute_info
        )

        table = PrettyTable()
        table.title = "RMSE Info"
        table.field_names = ["index", "Actual", "Golden"]

        table.add_row(["rmse", self.rmse, other.rmse])
        print(table)


class PrecisionStatistician:
    @classmethod
    @func_log_wrapper()
    def _stat_re_distribute(
        cls,
        actual_res: np.ndarray,
        golden_res: np.ndarray,
        re_info: np.ndarray,
        bins: List[float],
        small_value_mask: np.ndarray,
        inf_nan_mask: np.ndarray,
    ) -> REDistributeInfo:
        mask = np.logical_or(small_value_mask, inf_nan_mask)
        masked_re_info = np.where(mask, -1, re_info)
        masked_bins = [-1.0,] + bins
        dist_stat, dist_stat_bin = np.histogram(masked_re_info, masked_bins)
        stat_count = np.sum(dist_stat[1:])
        masked_count = dist_stat[:1][0]
        if stat_count == 0:
            max_re_idx = 0
            avg_re = 0
            max_re = 0
        else:
            max_re_idx = np.argmax(masked_re_info)
            max_re = re_info[max_re_idx]
            total_count = masked_count + stat_count
            # masked_re_avg = np.average(masked_re_info)
            # avg_re = (masked_re_avg + masked_count / total_count) * (total_count / stat_count)
            stat_re_info = masked_re_info[np.logical_not(mask)]
            avg_re = np.average(stat_re_info)
        return REDistributeInfo(
            dist_stat[1:],
            masked_bins[1:],
            max_re,
            [actual_res[max_re_idx], golden_res[max_re_idx]],
            avg_re,
        )

    @classmethod
    @func_log_wrapper()
    def _stat_small_value_info(
        cls,
        abs_ae: np.ndarray,
        small_value_mask: np.ndarray,
        small_value_thresh: float,
        total_cnt: int,
    ) -> SmallValueInfo:
        small_case_cnt = np.sum(small_value_mask)
        npu_small_err_mask = np.logical_not(
            np.less_equal(abs_ae, small_value_thresh)
        )  # use less to fix nan
        npu_small_err_mask = np.logical_and(
            npu_small_err_mask, small_value_mask
        )
        small_case_err_cnt = np.sum(npu_small_err_mask)
        return SmallValueInfo(
            small_value_thresh, small_case_cnt, small_case_err_cnt, total_cnt
        )

    @classmethod
    @func_log_wrapper()
    def _stat_rmse_info(
        cls, abs_ae: np.ndarray, inf_nan_mask: np.ndarray
    ) -> float:
        masked_ae = np.where(inf_nan_mask, 0, abs_ae)
        rmse = np.mean(np.square(masked_ae))
        inf_nan_cnt = np.sum(inf_nan_mask)
        rmse = rmse * (
            abs_ae.size / (abs_ae.size - inf_nan_cnt + 0.0001) + 0.0001
        )
        rmse = np.sqrt(rmse)
        return rmse

    @classmethod
    @func_log_wrapper()
    def _stat_eb_info(cls, actual: np.ndarray, golden: np.ndarray) -> EBInfo:
        golden_round = golden.astype(actual.dtype)
        smaller_cnt = np.sum(np.less(actual, golden_round))
        larger_cnt = np.sum(np.less(actual, golden))
        equal_cnt = np.sum(np.equal(actual, golden))

        return EBInfo(equal_cnt, larger_cnt, smaller_cnt, golden.size)

    @classmethod
    @func_log_wrapper()
    def _get_ulp_params(cls, np_data_type):
        if np_data_type == "float32":
            return -126, 23
        if np_data_type == "bfloat16":
            return -126, 7
        return -14, 10

    @classmethod
    @func_log_wrapper()
    def _stat_ulp_info(
        cls,
        actual: np.ndarray,
        ae: np.ndarray,
        abs_golden: np.ndarray,
        finite_mask: np.ndarray,
        golden: np.ndarray,
        actual_dtype: str,
    ):
        eb = np.floor(np.log2(abs_golden, dtype=np.float64))
        eb = np.where(np.equal(abs_golden, 0), 0, eb)
        min_eb, exponent_num = cls._get_ulp_params(actual_dtype)
        eb = np.maximum(eb, min_eb)
        eb = np.where(finite_mask, eb, 0)

        ulp_err = ae.astype(np.float64) * np.exp2(
            exponent_num - eb, dtype=np.float64
        )
        # set not finite_mask to nan
        ulp_err_fix_to_nan = np.where(finite_mask, ulp_err, np.nan)

        bins = REHistogramBins.get_ulp_histogram_bin(actual_dtype)
        histogram_stat, histogram_bins = np.histogram(ulp_err_fix_to_nan, bins)
        total_cnt = np.sum(histogram_stat)
        if total_cnt > 0:
            ulp_err_fix_to_neg_inf = np.where(finite_mask, ulp_err, -np.inf)
            max_ulp = np.max(ulp_err_fix_to_neg_inf)
            max_idx = np.argmax(ulp_err_fix_to_neg_inf)
            ulp_err_fix_to_pos_inf = np.where(finite_mask, ulp_err, np.inf)
            min_ulp = np.min(ulp_err_fix_to_pos_inf)
            min_idx = np.argmin(ulp_err_fix_to_pos_inf)
            ulp_err_fix_to_0 = np.where(finite_mask, ulp_err, 0)
            ulp_limit = 1 if actual_dtype in ("bfloat16", "float16") else 32
            ulp_err_cnt = np.sum(np.abs(ulp_err_fix_to_0) > ulp_limit)
            avg_ulp = np.average(ulp_err_fix_to_0) * (
                (actual.size / total_cnt) if total_cnt > 0 else 0
            )
            avg_abs_ulp = np.average(np.abs(ulp_err_fix_to_0)) * (
                (actual.size / total_cnt) if total_cnt > 0 else 0
            )
            return ULPDistributeInfo(
                histogram_bins,
                histogram_stat,
                total_cnt,
                ulp_err_cnt,
                max_ulp,
                min_ulp,
                avg_ulp,
                avg_abs_ulp,
                max_value_info=[actual[max_idx], golden[max_idx]],
                min_val_info=[actual[min_idx], golden[min_idx]],
            )
        else:
            return ULPDistributeInfo(
                histogram_bins, histogram_stat, 0, 0, 0, 0, 0, 0, [0, 0], [0, 0]
            )

    @classmethod
    @func_log_wrapper()
    def stat_precision_info(
        cls,
        golden: np.ndarray,
        actual: np.ndarray,
        *,
        abs_golden: np.ndarray = None,
        abs_golden_with_eps: np.ndarray = None,
        small_value_mask: np.ndarray = None,
        small_value_thresh: Union[float, None] = None,
        inf_nan_mask: np.ndarray = None,
        abs_ae: np.ndarray = None,
        re: np.ndarray = None,
        actual_dtype: str = None,
    ) -> PrecisionStatInfo:
        logging.info("[PrecisionStatistician] start stat_precision_info")
        shape_size = golden.size
        # actual_dtype = actual.dtype
        golden = golden.reshape((shape_size,))
        if abs_golden is None:
            abs_golden = np.abs(golden)
        actual = actual.reshape((shape_size,))
        if inf_nan_mask is None:
            golden_inf_mask = np.isinf(golden)
            inf_mask = np.isinf(actual)
            golden_nan_mask = np.isnan(golden)
            nan_mask = np.isnan(actual)
            golden_inf_nan_mask = np.logical_or(
                golden_inf_mask, golden_nan_mask
            )
            inf_nan_mask = np.logical_or(inf_mask, nan_mask)
            inf_nan_mask = np.logical_or(golden_inf_nan_mask, inf_nan_mask)
        # if ae is None:
        #     ae = actual - golden
        if abs_ae is None:
            abs_ae = np.abs(golden - actual)
        # if small_value_thresh is None:
        #     small_value_thresh = SmallValueThresh.get_small_value_thresh(actual_dtype)
        if small_value_mask is None:
            small_value_mask = np.less_equal(
                abs_golden, small_value_thresh
            )  # nan<little ==> false
        if abs_golden_with_eps is None:
            abs_golden_with_eps = abs_golden + np.finfo(golden.dtype).eps
        if re is None:
            re = abs_ae / abs_golden_with_eps
        re_stat_info = cls._stat_re_distribute(
            actual,
            golden,
            re,
            REHistogramBins.get_histogram_bins(actual_dtype),
            small_value_mask,
            inf_nan_mask,
        )
        small_case_info = cls._stat_small_value_info(
            abs_ae, small_value_mask, small_value_thresh, shape_size
        )
        rmse = cls._stat_rmse_info(
            abs_ae, np.logical_or(inf_nan_mask, small_value_mask)
        )
        eb_info = cls._stat_eb_info(actual, golden)
        ae = actual - golden
        ulp_distribute_info = cls._stat_ulp_info(
            actual,
            ae,
            abs_golden,
            finite_mask=np.logical_not(inf_nan_mask),
            golden=golden,
            actual_dtype=actual_dtype,
        )
        res = PrecisionStatInfo(
            re_stat_info, small_case_info, eb_info, rmse, ulp_distribute_info
        )
        logging.info("[PrecisionStatistician] end stat_precision_info")
        return res


class BenchmarkCompareStandard:
    def __init__(self, data_type, **kwargs):
        self.data_type = data_type
        self.max_re_rtol = kwargs.get("max_re_rtol", 10.0)
        self.avg_re_rtol = kwargs.get("avg_re_rtol", 2.0)
        self.rmse_rtol = kwargs.get("rmse_rtol", 2.0)
        self.ulp_rtol = kwargs.get("ulp_rtol", 2.0)
        self.ulp_atol = kwargs.get("ulp_atol")
        self.ae_rtol = kwargs.get("ae_rtol", 2.0)
        self.small_value = kwargs.get("small_value", None)
        if self.small_value is None:
            self.small_value = {
                "float32": 0.000001,
                "float16": 0.001,
                "bfloat16": 0.001,
                "float": 0.000001,
            }[data_type]
        if self.ulp_atol is None:
            self.ulp_atol = {
                "float16": 0.001,
                "bfloat16": 0.001,
                "float32": 0.05,
                "float": 0.05,
            }[data_type]
        self.small_value_atol = kwargs.get("small_value", self.small_value)


def benchmark_compare(
    actual: np.ndarray,
    other: np.ndarray,
    golden: np.ndarray,
    is_bf16=False,
    **kwargs,
):
    actual = actual.flatten()
    golden = golden.flatten()
    other = other.flatten()
    data_type = str(actual.dtype)

    standard = BenchmarkCompareStandard(data_type, **kwargs)
    result = PrecisionCheckResult(standard)
    actual_dtype = actual.dtype
    abs_golden = np.abs(golden)
    eps = np.finfo(golden.dtype).eps if not is_bf16 else 2 ** -8
    abs_golen_with_eps = abs_golden + eps

    golden_with_actual_dtype = golden.astype(actual_dtype)
    npu_finite_mask = np.isfinite(actual)
    gpu_finite_mask = np.isfinite(other)
    golden_finite_mask = np.isfinite(golden_with_actual_dtype)
    all_finite_mask = np.logical_and(npu_finite_mask, gpu_finite_mask)
    all_finite_mask = np.logical_and(golden_finite_mask, all_finite_mask)
    inf_or_nan_mask = np.logical_not(all_finite_mask)

    small_value_thresh = standard.small_value
    small_value_mask = np.less_equal(abs_golden, small_value_thresh)
    small_value_mask = np.logical_and(all_finite_mask, small_value_mask)

    golden_finite = np.where(all_finite_mask, golden, 0)
    finite_cnt = np.sum(all_finite_mask)
    if finite_cnt > 0:
        golden_avg = np.average(golden_finite) * (golden.size / finite_cnt)
    else:
        golden_avg = 0

    npu_ae = actual - golden
    abs_npu_ae = np.abs(npu_ae)
    gpu_ae = other - golden
    abs_gpu_ae = np.abs(gpu_ae)
    npu_re = abs_npu_ae / abs_golen_with_eps
    gpu_re = abs_gpu_ae / abs_golen_with_eps

    npu_stat_info = PrecisionStatistician.stat_precision_info(
        golden,
        actual,
        abs_golden=abs_golden,
        abs_golden_with_eps=abs_golen_with_eps,
        small_value_mask=small_value_mask,
        inf_nan_mask=inf_or_nan_mask,
        # ae=npu_ae,
        small_value_thresh=standard.small_value,
        abs_ae=abs_npu_ae,
        re=npu_re,
        actual_dtype=standard.data_type,
    )

    gpu_stat_info = PrecisionStatistician.stat_precision_info(
        golden,
        other,
        abs_golden=abs_golden,
        abs_golden_with_eps=abs_golen_with_eps,
        small_value_mask=small_value_mask,
        inf_nan_mask=inf_or_nan_mask,
        small_value_thresh=standard.small_value,
        abs_ae=abs_gpu_ae,
        re=gpu_re,
        actual_dtype=standard.data_type,
    )

    result.npu_stat_info = npu_stat_info
    result.golden_avg = golden_avg
    result.benchmark_stat_info = gpu_stat_info
    result.total_cnt = npu_stat_info.re_distribute_info.total_cnt

    result.do_summary()
    result.print_detail_table()
    result.print_summary_table()

    return result


if __name__ == "__main__":
    actual = np.random.rand(4, 4).astype(np.float16)
    other = np.random.rand(4, 4).astype(np.float16)
    golden = np.random.rand(4, 4).astype(np.float32)

    ret = benchmark_compare(actual, other, golden)
    print(ret.get_check_result_str())
    print(ret.get_result_msg())
