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

at::Tensor npu_mhc_sinkhorn(const at::Tensor& h_res, int64_t out_flag, float eps, int64_t num_iters)
{
    TORCH_CHECK(h_res.dim() == 3 || h_res.dim() == 4, "h_res's dim should be 3 or 4, but got ",
                h_res.dim(), OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(num_iters > 0 && num_iters <= 100, "num_iters should be within (0, 100], but got ",
                num_iters, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(out_flag == 0, "out_flag should be 0, but got ",
                out_flag, OPS_ERROR(ErrCode::VALUE));

    at::Tensor result = npu_preparation::apply_tensor_without_format(h_res);
    at::Tensor norm_out = npu_preparation::apply_tensor_without_format(h_res);
    at::Tensor sum_out = npu_preparation::apply_tensor_without_format(h_res);
    EXEC_NPU_CMD(aclnnMhcSinkhorn, h_res, out_flag, eps, num_iters, result, norm_out, sum_out);
    return result, norm_out, sum_out;
}
} // namespace op_api