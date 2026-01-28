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

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_quant_gmm_alltoallv(const at::Tensor &gmm_x,
                                                         const at::Tensor &gmm_weight,
                                                         c10::string_view hcom,
                                                         int64_t ep_world_size,
                                                         at::IntArrayRef send_counts, // 本rank从通信域内每个rank发送多少token
                                                         at::IntArrayRef recv_counts, // 本rank从通信域内每个rank接收多少token
                                                         const c10::optional<at::Tensor> &gmm_x_scale,
                                                         const c10::optional<at::Tensor> &gmm_weight_scale,
                                                         const c10::optional<at::Tensor> &gmm_x_offset,
                                                         const c10::optional<at::Tensor> &gmm_weight_offset,
                                                         const c10::optional<at::Tensor> &send_counts_tensor,
                                                         const c10::optional<at::Tensor> &recv_counts_tensor,
                                                         const c10::optional<at::Tensor> &mm_x,
                                                         const c10::optional<at::Tensor> &mm_weight,
                                                         const c10::optional<at::Tensor> &mm_x_scale,
                                                         const c10::optional<at::Tensor> &mm_weight_scale,
                                                         const c10::optional<at::Tensor> &mm_x_offset,
                                                         const c10::optional<at::Tensor> &mm_weight_offset,
                                                         const c10::optional<at::Tensor> &comm_quant_scale,
                                                         bool trans_gmm_weight, bool trans_mm_weight,
                                                         int64_t gmm_x_quant_mode,
                                                         int64_t gmm_weight_quant_mode,
                                                         int64_t mm_x_quant_mode,
                                                         int64_t mm_weight_quant_mode,
                                                         int64_t comm_quant_mode,
                                                         int64_t comm_quant_dtype,
                                                         c10::optional<int64_t> gmm_x_dtype,
                                                         c10::optional<int64_t> gmm_weight_dtype,
                                                         c10::optional<int64_t> gmm_x_scale_dtype,
                                                         c10::optional<int64_t> gmm_weight_scale_dtype,
                                                         c10::optional<int64_t> mm_x_dtype,
                                                         c10::optional<int64_t> mm_weight_dtype,
                                                         c10::optional<int64_t> mm_x_scale_dtype,
                                                         c10::optional<int64_t> mm_weight_scale_dtype,
                                                         c10::optional<int64_t> gmm_y_dtype,
                                                         c10::optional<int64_t> mm_y_dtype
                                                         )
    {

        // 声明并初始化 gmm_y 存储矩阵乘法的结果
        // recv_counts 是本 rank 从通信域内每个rank接收多少token

        at::Tensor mm_y{nullptr};    
        int64_t bsk = 0;                // 
        for (auto &i : recv_counts) {  // 累加recv counts
            bsk += i;                  // 本rank在经过AllToAllv之后拿到的Token总数
        }
        // weight是否转置？ 如果转置，最后输出的就是
        int64_t n1 = trans_gmm_weight ? gmm_weight.size(1) : gmm_weight.size(2);
        if (mm_x.has_value() && mm_weight.has_value()) { // 如果使用共享专家，mmx 和 mmweight都传，再计算mmy
            const at::Tensor &mm_x_value = mm_x.value(); // 取 mm_x里面的真实值
            const at::Tensor &mm_weight_value = mm_weight.value();  // 同上
            int64_t bs = mm_x_value.size(0);   // mmx的shape是(BS， H)
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1); // 跟上面一样，在读取输出的shape的第二个值
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mm_x_value.options()); // 创建分配输出一个Tensor
        }

        auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, gmm_x.options()); // 创建分配输出一个Tensor
        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());  // 有传值，无则传空
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        const char *hcom_ptr = (char *)hcom.data();  // 通信域内容 但是ACLNN中好像没有这个东西 group？
        EXEC_NPU_CMD(aclnnQuantGroupedMatMulAlltoAllv,  // 调用ACLNN
                     gmm_x,
                     gmm_weight,
                     gmm_x_scale,
                     gmm_weight_scale,
                     gmm_x_offset,
                     gmm_weight_offset,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_real,
                     mm_weight_real,
                     mm_x_scale,
                     mm_weight_scale,
                     mm_x_offset,
                     mm_weight_offset,
                     comm_quant_scale,
                     gmm_x_quant_mode,
                     gmm_weight_quant_mode,
                     mm_x_quant_mode,
                     mm_weight_quant_mode,
                     comm_quant_mode,
                     comm_quant_dtype,
                     hcom_ptr,
                     ep_world_size,
                     send_counts, 
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     gmm_y,
                     mm_y);
        return std::tie(gmm_y, mm_y);  // 返回结果
    }
} // namespace op_api


