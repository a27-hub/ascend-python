# Copyright (c) Huawei Technologies Co., Ltd. 2012-2023. All rights reserved.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abc import ABC, abstractmethod
from prettytable import PrettyTable
from typing import Optional, Tuple, Union, Any, List
from pydantic import BaseModel
from utils import compute_error_balance, NONE_VALUE, ERROR_THDS, EB_THDS

import torch

MAX_RE_RATIO = 10
AVG_RE_RATIO = 2
ROOT_MEAN_SQUARED_RATIO = 2


def compute_relative_error(local_output: torch.Tensor, remote_output: torch.Tensor, small_value=1e-7) -> torch.Tensor:
    diff_value = torch.abs(torch.subtract(local_output, remote_output))
    diff_value_rel = diff_value / (torch.abs(remote_output) + small_value)
    return diff_value_rel


def compute_root_mean_squared_error(local_output: torch.Tensor, remote_output: torch.Tensor) -> torch.Tensor:
    diff_value = torch.subtract(local_output, remote_output)
    root_mean_squared_error = torch.sqrt(torch.sum(diff_value * diff_value) / diff_value.numel()).item()
    return root_mean_squared_error


def check_invalid_value(value: torch.Tensor):
    has_nan = torch.isnan(value).any()
    has_inf = torch.isinf(value).any()
    return has_nan or has_inf


class ResultState:
    SUCCESS = True
    WARNING = True
    ERROR = False


class ResultConfigBase(ABC, BaseModel):
    @abstractmethod
    def update(self, **kwargs):
        pass

class AccuracyConfig(ResultConfigBase):
    result: Optional[bool] = False
    error_info: Optional[str] = None
    new_benchmark_indicate: Optional[dict] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            setattr(self, key, value)


class BaseBenchmarkCompareStandard(ABC):

    def init_by_dtype(self, dtype):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    def check_in_dtypes(self, dtype) -> bool:
        return True


class DoubleBenchmarkCompareStandard(BaseBenchmarkCompareStandard):
    max_re_ratio = MAX_RE_RATIO
    avg_re_ratio = AVG_RE_RATIO
    root_mean_squared_ratio = ROOT_MEAN_SQUARED_RATIO

    def __init__(self):
        self.error_thd = ERROR_THDS[torch.float32]
        self.eb_thd = EB_THDS[torch.float32]

    def init_by_dtype(self, dtype):
        self.error_thd = self.get_thd(dtype, ERROR_THDS) or ERROR_THDS[torch.float32]
        self.eb_thd = self.get_thd(dtype, EB_THDS) or EB_THDS[torch.float32]

    def update(self, **kwargs):
        if "max_re_ratio" in kwargs:
            self.max_re_ratio = kwargs.get("max_re_ratio")
        if "avg_re_ratio" in kwargs:
            self.avg_re_ratio = kwargs.get("avg_re_ratio")
        if "root_mean_squared_ratio" in kwargs:
            self.root_mean_squared_ratio = kwargs.get("root_mean_squared_ratio")

    def in_dtype(self, dtype):
        return dtype in ERROR_THDS

    @staticmethod
    def get_thd(dtype, thds):
        if dtype == torch.float64:
            print("The output data of fp64 uses the same standard as fp32.")
            return thds.get(torch.float32)[dtype]
        if dtype in thds.keys():
            return thds.get(dtype)
        print(
            "double benchmark compare only supports floating point "
            "in fp16, bf16, fp32."
        )
        return NONE_VALUE


class BaseResult(ABC):

    def __init__(self, standard, *args, **kwargs):
        self.standard = standard
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def do_summary(self, golden_result):
        pass

    def get_detail_data(self) -> dict:
        return {}

    def get_compare_data(self) -> dict:
        return {}

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        return ResultState.SUCCESS, ""


class PrecisionStatInfo:

    def __init__(self, results: List[BaseResult]):
        self.results = results

    def do_summary(self, benchmark_stat_info):
        for index, result in enumerate(self.results):
            result.do_summary(benchmark_stat_info.results[index])

    def get_detail_data(self) -> dict:
        ret = dict()
        for result in self.results:
            ret.update(result.get_detail_data())
        return ret

    def get_compare_data(self) -> dict:
        ret = dict()
        for result in self.results:
            ret.update(result.get_compare_data())
        return ret

    def get_result_msg(self, result_msg="") -> Tuple[bool, Union[str, Any]]:
        check_result = ResultState.SUCCESS
        for result in self.results:
            _check_result, check_result_msg = result.get_result()
            if not _check_result:
                result_msg += check_result_msg
                check_result = ResultState.ERROR
        return check_result, result_msg


class BaseBenchmarkSummary:

    def __init__(
        self,
        actual: PrecisionStatInfo,
        benchmark: PrecisionStatInfo,
        *args,
        **kwargs,
    ):
        self.actual = actual
        self.benchmark = benchmark
        self.check_result = None
        self.args = args
        self.kwargs = kwargs

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        self.actual.do_summary(self.benchmark)
        return self.actual.get_result_msg()

    def get_all_data(self) -> dict:
        all_ret = {}
        actual_ret = self.actual.get_detail_data()
        for key, value in actual_ret.items():
            all_ret[f"Actual_{key}"] = value
        benchmark_ret = self.benchmark.get_detail_data()
        for key, value in benchmark_ret.items():
            all_ret[f"Benchmark_{key}"] = value
        compare_ret = self.actual.get_compare_data()
        all_ret.update(compare_ret)
        return all_ret


class ResultConfig(BaseModel):
    value: Optional[float] = None
    compare_value: Optional[float] = None
    name: Optional[str] = "值"
    error_str: Optional[str] = None
    warn_value: Optional[float] = None
    warn_str: Optional[float] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is None:
                continue
            setattr(self, key, value)

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        check_result = ResultState.SUCCESS
        result_str = ""
        if self.value > self.compare_value:
            check_result = check_result and ResultState.ERROR
            result_str += self.error_str or f"ERROR: {self.name}({self.value})超过阈值({self.compare_value})，\n"
        elif self.warn_value is not None and self.value > self.warn_value:
            check_result = check_result and ResultState.WARNING
            result_str += self.warn_str or f"WARNING: {self.name}({self.value})超过阈值({self.warn_value})，\n"
        return check_result, result_str


class EBResult(BaseResult):
    def __init__(self, standard, **kwargs):
        super(EBResult, self).__init__(standard, **kwargs)
        self.error_balance = kwargs.get("error_balance")
        self.eb_difference = None

    def do_summary(self, golden_result):
        self.eb_difference = abs(self.error_balance)

    def get_detail_data(self) -> dict:
        return {
            "误差均衡性": self.error_balance,
        }

    def get_compare_data(self) -> dict:
        return {
            "误差均衡性": self.eb_difference,
        }

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        return ResultConfig(value=self.eb_difference,
                            compare_value=self.standard.eb_thd,
                            name="误差均衡性").get_result()


class MaxReResult(BaseResult):
    def __init__(self, standard, **kwargs):
        super(MaxReResult, self).__init__(standard, **kwargs)
        self.max_relative_error = kwargs.get("max_relative_error")
        self.max_re_ratio = None

    def do_summary(self, golden_result):
        self.max_re_ratio = self.max_relative_error / max(golden_result.max_relative_error, self.standard.error_thd)

    def get_detail_data(self) -> dict:
        return {
            "最大相对误差": self.max_relative_error,
        }

    def get_compare_data(self) -> dict:
        return {
            "最大相对误差比例": self.max_re_ratio,
        }

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        return ResultConfig(value=self.max_re_ratio,
                            compare_value=self.standard.max_re_ratio,
                            name="最大相对误差比例").get_result()


class AvgReResult(BaseResult):
    def __init__(self, standard, **kwargs):
        super(AvgReResult, self).__init__(standard, **kwargs)
        self.avg_relative_error = kwargs.get("avg_relative_error")
        self.avg_re_ratio = None

    def do_summary(self, golden_result):
        self.avg_re_ratio = self.avg_relative_error / max(golden_result.avg_relative_error, self.standard.error_thd)

    def get_detail_data(self) -> dict:
        return {
            "平均相对误差": self.avg_relative_error,
        }

    def get_compare_data(self) -> dict:
        return {
            "平均相对误差比例": self.avg_re_ratio,
        }

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        return ResultConfig(value=self.avg_re_ratio,
                            compare_value=self.standard.avg_re_ratio,
                            name="平均相对误差比例").get_result()


class RmseResult(BaseResult):
    def __init__(self, standard, **kwargs):
        super(RmseResult, self).__init__(standard, **kwargs)
        self.root_mean_squared_error = kwargs.get("root_mean_squared_error")
        self.root_mean_squared_ratio = None

    def do_summary(self, golden_result):
        self.root_mean_squared_ratio = (self.root_mean_squared_error /
                                        max(golden_result.root_mean_squared_error, self.standard.error_thd))

    def get_detail_data(self) -> dict:
        return {
            "均方根误差": self.root_mean_squared_error,
        }

    def get_compare_data(self) -> dict:
        return {
            "均方根误差比例": self.root_mean_squared_ratio,
        }

    def get_result(self) -> Tuple[bool, Union[str, Any]]:
        return ResultConfig(value=self.root_mean_squared_ratio,
                            compare_value=self.standard.root_mean_squared_ratio,
                            name="均方根误差比例").get_result()


class BaseAccuracyCompare(ABC):

    def __init__(self, main_outputs: List[torch.Tensor], compare_outputs: List[torch.Tensor], bm_datas: List[torch.Tensor] = None, **kwargs):
        self.main_outputs = main_outputs
        self.compare_outputs = compare_outputs
        self.bm_datas = bm_datas
        self.kwargs = kwargs

    def run(self) -> List[AccuracyConfig]:
        if len(self.main_outputs) != len(self.compare_outputs):
            raise RuntimeError(f"main_outputs length {len(self.main_outputs)} not equals compare_outputs length {len(self.compare_outputs)}")
        if self.bm_datas and len(self.main_outputs) != len(self.bm_datas):
            raise RuntimeError(
                f"main_outputs length {len(self.main_outputs)} not equals bm_outputs length {len(self.bm_datas)}")
        accuracy_result_list = self.compare_for_data()
        self.print_result(accuracy_result_list)
        return accuracy_result_list

    def print_result(self, accuracy_result_list):
        ret = True
        for index, accuracy_result in enumerate(accuracy_result_list):
            if not accuracy_result.result:
                ret = False
                error_info = accuracy_result.error_info
                print(f"tensor {index} compare result: False, error_info: {error_info} \n")
            else:
                print(f"{index} compare result: True")
        print(f"all compare result: {ret}")

    def compare_for_data(self) -> List[AccuracyConfig]:
        accuracy_result_list = []
        for index in range(len(self.main_outputs)):
            local_output = self.main_outputs[index]
            remote_output = self.compare_outputs[index]
            bm_data = self.bm_datas[index] if self.bm_datas else None
            acc_result = self.check_output_size(
                local_output, remote_output
            )
            if acc_result:
                accuracy_result_list.append(acc_result)
                return accuracy_result_list
            acc_result = self.compute_accuracy_result(local_output, remote_output, bm_data=bm_data)
            accuracy_result_list.append(acc_result)
        return accuracy_result_list

    @abstractmethod
    def compute_accuracy_result(self, local_output, remote_output, bm_data=None) -> AccuracyConfig:
        pass

    @staticmethod
    def check_output_size(local_output, remote_output):
        acc_result = None
        if local_output.dtype != remote_output.dtype:
            error_info = f"The dtypes of local:[{local_output.dtype}] " \
                         f"and remote[{remote_output.dtype}] are different."
            print(error_info)
            acc_result = AccuracyConfig(
                result=False,
                error_info=error_info,
            )
        if local_output.numel() == 0 and remote_output.numel() == 0:
            info = "The npu_output is [],and it is same as bm_output, " \
                   "the result of data_compare is Pass"
            print(info)
            acc_result = AccuracyConfig(
                result=True,
                error_info=info
            )

        if local_output.size() != remote_output.size():
            error_info = f"the size of npu output[{local_output.size()}] and " \
                         f"benchmark[{remote_output.size()}] is not equal."

            print(error_info)
            acc_result = AccuracyConfig(
                result=False,
                error_info=error_info
            )
        return acc_result


class BaseBenchmarkAccuracyCompare(BaseAccuracyCompare):
    DTYPES = []

    def __init__(self, main_outputs: List, compare_outputs: List, bm_datas: List = None, **kwargs):
        super(BaseBenchmarkAccuracyCompare, self).__init__(main_outputs, compare_outputs, bm_datas=bm_datas, **kwargs)
        self.benchmark_standard = None
        self.benchmark_summary_cls = None
        self.init_benchmark(**kwargs)

    def init_benchmark(self, **kwargs):
        self.benchmark_standard = BaseBenchmarkCompareStandard()
        self.benchmark_standard.update(**kwargs)
        self.benchmark_summary_cls = BaseBenchmarkSummary

    def compute_accuracy_result(self, local_output, remote_output, bm_data=None) -> AccuracyConfig:
        if torch.is_complex(local_output):
            real_ret = self._compute_accuracy(
                local_output.real,
                remote_output.real,
                bm_data.real,
            )
            imag_ret = self._compute_accuracy(
                local_output.imag,
                remote_output.imag,
                bm_data.imag,
            )
            info = (
                "[info_real]:"
                + real_ret.error_info
                + "\n[info_imag]:"
                + imag_ret.error_info
            )
            all_data = dict()
            for key, value in real_ret.new_benchmark_indicate.items():
                all_data[f"[real]{key}"] = value
            for key, value in imag_ret.new_benchmark_indicate.items():
                all_data[f"[imag]{key}"] = value
            acc_ret = AccuracyConfig(
                result=real_ret.result and imag_ret.result,
                error_info=info,
                new_benchmark_indicate=all_data,
            )
        elif local_output.dtype in self.DTYPES:
            acc_ret = self._compute_accuracy(
                local_output, remote_output, bm_data
            )
        else:
            raise RuntimeError(
                f"only {self.DTYPES} support new benchmark compare, "
                f"but get {local_output.dtype}"
            )
        return acc_ret

    def _compute_accuracy(
            self, local_output, remote_output, bm_data
    ):
        acc_result = AccuracyConfig(new_benchmark_indicate={})
        if check_invalid_value(remote_output):
            acc_result.update(result=True)
            return acc_result
        if check_invalid_value(local_output):
            error_info = f"local_output contains nan/inf value"
            acc_result.update(result=False, error_info=error_info)
            return acc_result
        dtype = local_output.dtype
        in_dtypes = self.benchmark_standard.check_in_dtypes(dtype)
        if not in_dtypes:
            if torch.equal(local_output, remote_output):
                acc_result.update(result=True)
            else:
                error_info = f"The actual value is different from the benchmark value with equal comparison " \
                             f"in local_output"
                acc_result.update(result=False, error_info=error_info)
            return acc_result
        self.benchmark_standard.init_by_dtype(dtype)
        local_ret = self.compute_with_golden(local_output, bm_data)
        remote_ret = self.compute_with_golden(remote_output, bm_data)
        benchmark_summary = self.benchmark_summary_cls(
            local_ret, remote_ret, self.benchmark_standard, dtype
        )
        ret, error_info = benchmark_summary.get_result()
        all_data = benchmark_summary.get_all_data()
        acc_result.update(result=ret, error_info=error_info, new_benchmark_indicate=all_data)
        return acc_result

    @abstractmethod
    def compute_with_golden(self, local_output: torch.Tensor, bm_data: torch.Tensor) -> PrecisionStatInfo:
        pass

    def print_result(self, accuracy_result_list):
        ret = True
        for index, accuracy_result in enumerate(accuracy_result_list):
            if not accuracy_result.result:
                ret = False
                error_info = accuracy_result.error_info
                print(f"tensor {index} compare result: False, error_info: {error_info} \n")
                self.print_table(accuracy_result.new_benchmark_indicate or {})
            else:
                print(f"{index} compare result: True")
        print(f"all compare result: {ret}")

    @classmethod
    def print_table(cls, kwargs: dict, title="detail"):
        table = PrettyTable()
        table.title = title
        table.field_names = ["key", "value"]
        for key, value in kwargs.items():
            table.add_row([key, value])
        print(table)


class CvFusedDoubleBenchmarkAccuracyCompare(BaseBenchmarkAccuracyCompare):
    DTYPES = [torch.float16, torch.float32, torch.bfloat16]

    def __init__(self, main_outputs: List, compare_outputs: List, bm_datas: List = None, **kwargs):
        self.benchmark_standard = None
        self.benchmark_summary_cls = None
        super(CvFusedDoubleBenchmarkAccuracyCompare, self).__init__(main_outputs, compare_outputs, bm_datas=bm_datas, **kwargs)

    def init_benchmark(self, **kwargs):
        self.benchmark_standard = DoubleBenchmarkCompareStandard()
        self.benchmark_standard.update(**kwargs)
        self.benchmark_summary_cls = BaseBenchmarkSummary

    def compute_with_golden(self, actual, golden):
        print("begin to compute precision info with golden")
        actual.to(golden.dtype)
        res = PrecisionStatInfo(self.get_result(actual, golden))
        print("precision compute finish")
        return res

    def get_result(self, actual, golden):
        max_re_info = self._stat_max_re_info(actual, golden)
        avg_re_info = self._stat_avg_re_info(actual, golden)
        rmse_info = self._stat_rmse_info(actual, golden)
        eb_info = self._stat_eb_info(actual, golden)
        return [max_re_info, avg_re_info, rmse_info, eb_info]

    def _stat_max_re_info(self, actual, golden):
        max_relative_error = torch.max(compute_relative_error(actual, golden)).item()
        return MaxReResult(
            self.benchmark_standard,
            max_relative_error=max_relative_error,
        )

    def _stat_avg_re_info(self, actual, golden):
        avg_relative_error = torch.mean(compute_relative_error(actual, golden)).item()
        return AvgReResult(
            self.benchmark_standard,
            avg_relative_error=avg_relative_error,
        )

    def _stat_rmse_info(self, actual, golden):
        root_mean_squared_error = compute_root_mean_squared_error(actual, golden)
        return RmseResult(
            self.benchmark_standard,
            root_mean_squared_error=root_mean_squared_error,
        )

    def _stat_eb_info(self, actual: torch.Tensor, golden: torch.Tensor):
        eb_difference = compute_error_balance(actual, golden)
        return EBResult(
            self.benchmark_standard,
            error_balance=eb_difference,
        )


def main():
    actual = torch.rand(size=(4, 4)).to(torch.float16)
    other = torch.rand(size=(4, 4)).to(torch.float16)
    golden = torch.rand(size=(4, 4)).to(torch.float32)
    compare = CvFusedDoubleBenchmarkAccuracyCompare([actual], [other], [golden])
    compare.run()


if __name__ == "__main__":
    main()