// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_quant_gmm_alltoallv(const at::Tensor &gmm_x,
                                                         const at::Tensor &gmm_weight,
                                                         const at::Tensor &gmm_x_scale,
                                                         const at::Tensor &gmm_weight_scale,
                                                         c10::string_view hcom,
                                                         int64_t ep_world_size,
                                                         at::IntArrayRef send_counts, 
                                                         at::IntArrayRef recv_counts,
                                                         int64_t gmm_y_dtype,
                                                         const c10::optional<at::Tensor> &send_counts_tensor,
                                                         const c10::optional<at::Tensor> &recv_counts_tensor,
                                                         const c10::optional<at::Tensor> &mm_x,
                                                         const c10::optional<at::Tensor> &mm_weight,
                                                         const c10::optional<at::Tensor> &mm_x_scale,
                                                         const c10::optional<at::Tensor> &mm_weight_scale,
                                                         const c10::optional<at::Tensor> &gmm_x_offset,
                                                         const c10::optional<at::Tensor> &gmm_weight_offset,
                                                         const c10::optional<at::Tensor> &mm_x_offset,
                                                         const c10::optional<at::Tensor> &mm_weight_offset,
                                                         const c10::optional<at::Tensor> &comm_quant_scale,
                                                         c10::optional<int64_t> gmm_x_quant_mode,
                                                         c10::optional<int64_t> gmm_weight_quant_mode,
                                                         c10::optional<int64_t> mm_x_quant_mode,
                                                         c10::optional<int64_t> mm_weight_quant_mode,
                                                         c10::optional<int64_t> comm_quant_mode,
                                                         c10::OptionalIntArrayRef group_size,
                                                         c10::optional<int64_t> gmm_x_dtype,
                                                         c10::optional<int64_t> gmm_weight_dtype,
                                                         c10::optional<int64_t> gmm_x_scale_dtype,
                                                         c10::optional<int64_t> gmm_weight_scale_dtype,
                                                         c10::optional<int64_t> mm_x_dtype,
                                                         c10::optional<int64_t> mm_weight_dtype,
                                                         c10::optional<int64_t> mm_x_scale_dtype,
                                                         c10::optional<int64_t> mm_weight_scale_dtype,
                                                         c10::optional<int64_t> comm_quant_dtype,
                                                         c10::optional<int64_t> mm_y_dtype
                                                         )
    {
        TORCH_CHECK(gmm_x_quant_mode == 1, "gmm_x_quant_mode only support 1 (pertensor), but got ",
                gmm_x_quant_mode, OPS_ERROR(ErrCode::PARAM));
        
        TORCH_CHECK(gmm_weight_quant_mode == 1, "gmm_weight_quant_mode only support 1 (pertensor), but got ",
                gmm_weight_quant_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(mm_x_quant_mode == 1, "mm_x_quant_mode only support 1 (pertensor), but got ",
                mm_x_quant_mode, OPS_ERROR(ErrCode::PARAM));
        
        TORCH_CHECK(mm_weight_quant_mode == 1, "mm_weight_quant_mode only support 1 (pertensor), but got ",
                mm_weight_quant_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(comm_quant_mode == 0, "comm_quant_mode only support 0, but got ",
                comm_quant_mode, OPS_ERROR(ErrCode::PARAM));

        // dim
        at::Tensor mm_y{nullptr};    
        int64_t bsk = 0;            
        for (auto &i : recv_counts) { 
            bsk += i;                 
        }
        bool trans_gmm_weight = false;
        bool trans_mm_weight = false;
        int64_t n1 = gmm_weight.size(1);
        if (mm_x.has_value() && mm_weight.has_value()) {
            const at::Tensor &mm_x_value = mm_x.value(); 
            const at::Tensor &mm_weight_value = mm_weight.value();  
            int64_t bs = mm_x_value.size(0);   // shape (BS， H)
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1); 
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mm_x_value.options()); 
        }

        auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, gmm_x.options()); 
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        char* hcom_ptr = const_cast<char*>(hcom.data());
        TensorWrapper gmm_x_wrapper = {gmm_x, gmm_x_dtype.value_or(ACL_UINT8)};
        TensorWrapper gmm_weight_wrapper = {gmm_weight, gmm_x_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_x_wrapper = {mm_x.value_or(at::Tensor()), mm_x_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_weight_wrapper = {mm_weight.value_or(at::Tensor()), mm_weight_dtype.value_or(ACL_UINT8)};
        TensorWrapper gmm_x_scale_wrapper = {gmm_x_scale, gmm_x_scale_dtype.value_or(ACL_UINT8)};
        TensorWrapper gmm_weight_scale_wrapper = {gmm_weight_scale, gmm_weight_scale_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_x_scale_wrapper ={mm_x_scale.value_or(at::Tensor()), mm_x_scale_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_weight_scale_wrapper = {mm_weight.value_or(at::Tensor()), mm_weight_scale_dtype.value_or(ACL_UINT8)};
        TensorWrapper gmm_x_offset_wrapper = {gmm_x_offset.value_or(at::Tensor()), gmm_x_dtype.value_or(ACL_UINT8)};
        TensorWrapper gmm_weight_offset_wrapper = {gmm_weight_offset.value_or(at::Tensor()), gmm_weight_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_x_offset_wrapper = {mm_x_offset.value_or(at::Tensor()), mm_x_dtype.value_or(ACL_UINT8)};
        TensorWrapper mm_weight_offset_wrapper = {mm_weight_offset.value_or(at::Tensor()), mm_weight_dtype.value_or(ACL_UINT8)};
        TensorWrapper comm_quant_scale_wrapper = {comm_quant_scale.value_or(at::Tensor()), comm_quant_dtype.value_or(ACL_UINT8)};
        int64_t gmm_x_quant_mode_real = gmm_x_quant_mode.value_or(1);
        int64_t gmm_weight_quant_mode_real = gmm_weight_quant_mode.value_or(1);
        int64_t mm_x_quant_mode_real = mm_x_quant_mode.value_or(1);
        int64_t mm_weight_quant_mode_real = mm_weight_quant_mode.value_or(1);
        int64_t comm_quant_mode_real = comm_quant_mode.value_or(0);
        int64_t comm_quant_dtype_real = comm_quant_dtype.value_or(0);
        at::IntArrayRef group_size_real = group_size.value_or(at::IntArrayRef{});
        EXEC_NPU_CMD(aclnnQuantGroupedMatMulAlltoAllv, 
                     gmm_x_wrapper,
                     gmm_weight_wrapper,
                     gmm_x_scale_wrapper,
                     gmm_weight_scale_wrapper,
                     gmm_x_offset_wrapper,
                     gmm_weight_offset_wrapper,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_wrapper,
                     mm_weight_wrapper,
                     mm_x_scale_wrapper,
                     mm_weight_scale_wrapper,
                     mm_x_offset_wrapper,
                     mm_weight_offset_wrapper,
                     comm_quant_scale_wrapper,
                     gmm_x_quant_mode_real,
                     gmm_weight_quant_mode_real,
                     mm_x_quant_mode_real,
                     mm_weight_quant_mode_real,
                     comm_quant_mode_real,
                     comm_quant_dtype_real,
                     group_size_real,
                     hcom_ptr,
                     ep_world_size,
                     send_counts, 
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     gmm_y,
                     mm_y);
        return std::tie(gmm_y, mm_y);  
    }
} // namespace op_api


