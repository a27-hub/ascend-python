#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import logging
from functools import wraps

import torch


def compute_error_balance(local_output: torch.Tensor, remote_output: torch.Tensor, small_value=1.0) -> float:
    diff_value = torch.subtract(local_output, remote_output)
    scalar = torch.full_like(remote_output, small_value)
    diff_value_rel = diff_value / torch.max(torch.abs(remote_output), scalar)
    return float(torch.mean(diff_value_rel))


def func_log_wrapper():
    def _out_wrapper(func):
        @wraps(func)
        def _in_wrapper(*kargs, **kwargs):
            logging.debug("start to run: %s" % func.__name__)
            x = func(*kargs, **kwargs)
            logging.debug("end to run: %s" % func.__name__)
            return x

        return _in_wrapper

    return _out_wrapper


NONE_VALUE = None
ERROR_THDS = {torch.float16: 2 ** -11,
              torch.bfloat16: 2 ** -8,
              torch.float32: 2 ** -14,
              torch.float64: 2 ** -14}
EB_THDS = {torch.float16: 2 ** -10,
           torch.bfloat16: 2 ** -7,
           torch.float32: 2 ** -14,
           torch.float64: 2 ** -14}
