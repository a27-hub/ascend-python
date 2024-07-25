// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;


void exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    if (non_blocking) {
        EXEC_NPU_CMD(aclnnForeachCopy, src, dst);
    } else {
        EXEC_NPU_CMD_SYNC(aclnnForeachCopy, src, dst);
    }
}

void split_and_exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    size_t tensor_count = src.size();
    size_t max_tensor_count = 1;
    max_tensor_count = 48;
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
            exec_npu_cmd_copy(dst, src, non_blocking);
            return;
        }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_src(src.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_dst(dst.data() + i * max_tensor_count, max_tensor_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_src(src.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_dst(dst.data() + loop_time * max_tensor_count, remaining_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }
}

void _foreach_copy_(const at::TensorList self, const at::TensorList src, bool non_blocking)
{
    at::native::check_foreach_api_restrictions(self, src, non_blocking);
    if (!at::native::can_use_fast_route(self, src)) {
        return at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    }

    if (self.empty()) {
        return;
    }
    // notice: foreach_copy_, has no real inplace, but here should give true to inplace config.
    split_and_exec_npu_cmd_copy(self, src, non_blocking);
}

} // namespace at_npu
