// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/csrc/autograd/custom_function.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const size_t INDEX_ONE = 1;
const size_t INDEX_TWO = 2;
const size_t DIM_TREE = 3;

at::Tensor npu_moe_finalize_routing(const at::Tensor& expandedX,
                                    const at::Tensor& expandedRowIdx,
                                    const c10::optional<at::Tensor>& x1,
                                    const c10::optional<at::Tensor>& x2,
                                    const c10::optional<at::Tensor>& bias,
                                    const c10::optional<at::Tensor>& scales,
                                    const c10::optional<at::Tensor>& expertIdx,
                                    const c10::optional<at::Tensor>& x,
                                    const c10::optional<at::Tensor>& a1,
                                    const c10::optional<at::Tensor>& a2,
                                    const c10::optional<at::Tensor>& v,
                                    const c10::optional<int64_t> drop_pad_mode,
                                    const c10::optional<at::IntArrayRef> zero_expert_range,
                                    const c10::optional<at::IntArrayRef> copy_expert_range,
                                    const c10::optional<at::IntArrayRef> constant_expert_range)
{
    static const bool is_moe_finalize_routing_V2_available = check_aclnn_kernel_available("aclnnMoeFinalizeRoutingV2");
    if (!is_moe_finalize_routing_V2_available) {
        TORCH_CHECK(x1.has_value(), "x1 parameter must have value when there is no aclnnMoeFinalizeRoutingV2",
            OPS_ERROR(ErrCode::PARAM));
        at::Tensor result = npu_preparation::apply_tensor_without_format(x1.value());
        EXEC_NPU_CMD(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, a1, a2, v
            drop_pad_mode, zero_expert_range, copy_expert_range, constant_expert_range, result);
        return result;
    }

    at::Tensor result;
    at::SmallVector<int64_t, op_infer::SIZE> output_size;
    // 如果 x1 存在，result 等于 x1 的形状
    if (x1.has_value()) {
        at::Tensor result = npu_preparation::apply_tensor_without_format(x1.value());
    }
    // 如果 x1 不存在，output_size 的第一维是 bias 或 scales 的第一维，第二维是 bias 的第二维
    else {
        output_size.push_back(dscales.value().size(0)); // 假设 scales 必须存在
        output_size.push_back(bias.value().size(1)); // 假设 bias 的第二维存在
        result = npu_preparation::apply_tensor_without_format(output_size, expandedX.options());
    }
    result = npu_preparation::apply_tensor_without_format(output_size, expandedX.options());
    int64_t mode = c10::value_or_else(drop_pad_mode, [] { return 0; });
    EXEC_NPU_CMD(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, a1, a2, v
            mode, zero_expert_range, copy_expert_range, constant_expert_range, result);
    return result;
}
} // namespace op_api