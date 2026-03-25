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
namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t FP4_IN_UINT8_NUM = 2;

#ifdef MOCK_ACLNN
// ============================================================================
// 参数收集 Mock：验证 Python → PTA C++ → aclnn 的参数传递链
//
// 使用方法：
//   1. 编译时添加 -DMOCK_ACLNN
//   2. Python 调用算子
//   3. 读取 /tmp/mock_aclnn_AddRmsNormDynamicMxQuant.json 验证参数
// ============================================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

class JsonBuilder {
public:
    std::ostringstream oss;
    bool first = true;

    void begin() { oss << "{\n"; }
    void end() { oss << "\n}"; }

    void addComma() {
        if (!first) oss << ",\n";
        first = false;
    }

    void add(const char* key, const char* value) {
        addComma();
        oss << "  \"" << key << "\": \"" << escape(value) << "\"";
    }

    void add(const char* key, int64_t value) {
        addComma();
        oss << "  \"" << key << "\": " << value;
    }

    void add(const char* key, double value) {
        addComma();
        oss << "  \"" << key << "\": " << std::setprecision(17) << value;
    }

    void add(const char* key, bool value) {
        addComma();
        oss << "  \"" << key << "\": " << (value ? "true" : "false");
    }

    void addTensor(const char* key, const at::Tensor& tensor) {
        addComma();
        oss << "  \"" << key << "\": {\n";
        oss << "    \"shape\": [";
        for (int64_t i = 0; i < tensor.dim(); i++) {
            if (i > 0) oss << ", ";
            oss << tensor.size(i);
        }
        oss << "],\n";
        oss << "    \"dtype\": \"" << c10::toString(tensor.scalar_type()) << "\",\n";
        oss << "    \"numel\": " << tensor.numel() << ",\n";
        oss << "    \"strides\": [";
        for (int64_t i = 0; i < tensor.dim(); i++) {
            if (i > 0) oss << ", ";
            oss << tensor.stride(i);
        }
        oss << "]\n";
        oss << "  }";
    }

    void addOptionalTensor(const char* key, const c10::optional<at::Tensor>& opt) {
        addComma();
        if (opt.has_value()) {
            const auto& tensor = opt.value();
            oss << "  \"" << key << "\": {\n";
            oss << "    \"has_value\": true,\n";
            oss << "    \"shape\": [";
            for (int64_t i = 0; i < tensor.dim(); i++) {
                if (i > 0) oss << ", ";
                oss << tensor.size(i);
            }
            oss << "],\n";
            oss << "    \"dtype\": \"" << c10::toString(tensor.scalar_type()) << "\"\n";
            oss << "  }";
        } else {
            oss << "  \"" << key << "\": {\"has_value\": false}";
        }
    }

    std::string str() const { return oss.str(); }

private:
    std::string escape(const char* s) {
        std::string result;
        while (*s) {
            if (*s == '"') result += "\\\"";
            else if (*s == '\\') result += "\\\\";
            else if (*s == '\n') result += "\\n";
            else result += *s;
            s++;
        }
        return result;
    }
};

// Mock 函数：参数签名与 EXEC_NPU_CMD 展开后传给 aclnn 的参数一一对应
// 注意：第 8 个参数是 aclDataType y_acltype（= dst_type - 256），不是 dst_type 原值
static void MockAclnnAddRmsNormDynamicMxQuant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    const c10::optional<at::Tensor>& beta,
    double epsilon,
    int64_t scale_alg,
    const char* round_mode,
    aclDataType y_acltype,
    bool output_rstd,
    const TensorWrapper& y_wrapper,
    at::Tensor& x_out,
    const TensorWrapper& mxscale_wrapper,
    at::Tensor& rstd_out)
{
    const at::Tensor& y = y_wrapper.tensor_;
    const at::Tensor& mxscale = mxscale_wrapper.tensor_;

    JsonBuilder json;
    json.begin();

    json.add("aclnn_function", "aclnnAddRmsNormDynamicMxQuant");

    // === 输入 Tensor ===
    json.addTensor("x1", x1);
    json.addTensor("x2", x2);
    json.addTensor("gamma", gamma);
    json.addOptionalTensor("beta", beta);

    // === 标量参数 ===
    json.add("epsilon", epsilon);
    json.add("scale_alg", scale_alg);
    json.add("round_mode", round_mode ? round_mode : "");
    // y_acltype = dst_type - 256，记录实际传给 aclnn 的值
    json.add("y_acltype", static_cast<int64_t>(y_acltype));
    json.add("output_rstd", output_rstd);

    // === 输出 Tensor 元信息（已由主函数分配好） ===
    json.addTensor("y_out", y);
    json.add("y_wrapper_dtype", static_cast<int64_t>(y_wrapper.dtype));
    json.addTensor("x_out", x_out);
    json.addTensor("mxscale_out", mxscale);
    json.add("mxscale_wrapper_dtype", static_cast<int64_t>(mxscale_wrapper.dtype));
    json.addTensor("rstd_out", rstd_out);

    json.end();

    const char* filepath = "/tmp/mock_aclnn_AddRmsNormDynamicMxQuant.json";
    std::ofstream ofs(filepath);
    if (ofs.is_open()) {
        ofs << json.str();
        ofs.close();
        std::cout << "[MOCK-ACLNN] Parameters captured to: " << filepath << std::endl;
    } else {
        std::cerr << "[MOCK-ACLNN] Failed to write " << filepath << std::endl;
        std::cout << json.str() << std::endl;
    }
}

#define EXEC_NPU_CMD_MOCK(cmd, ...) \
    do { MockAclnnAddRmsNormDynamicMxQuant(__VA_ARGS__); } while(0)
#endif // MOCK_ACLNN

}; // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm_dynamic_mx_quant(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    const c10::optional<at::Tensor> &beta,
    double epsilon,
    int64_t scale_alg,
    c10::string_view round_mode,
    int64_t dst_type)
{
    // 1. 推导 y shape（与 x1 一致）
    auto y_shape = op_infer::array_to_small_vector(x1.sizes());

    // 2. 推导 x_out shape（与 x1 一致）
    auto x_out_shape = op_infer::array_to_small_vector(x1.sizes());

    // 3. 推导 mxscale_out shape: rank = rank(x1) + 1
    //    mxscale_out.shape[-2] = (ceil(x1.shape[-1] / 32) + 2 - 1) / 2
    //    mxscale_out.shape[-1] = 2
    auto mxscale_shape = op_infer::array_to_small_vector(x1.sizes());
    int64_t last_dim = mxscale_shape.back();
    int64_t mxscale_last_dim = op_infer::CeilDiv(last_dim, BLOCK_SIZE);
    mxscale_last_dim = (mxscale_last_dim + ALIGN_NUM - 1) / ALIGN_NUM;
    mxscale_shape.back() = mxscale_last_dim;
    mxscale_shape.emplace_back(ALIGN_NUM);

    // 4. 推导 rstd_out shape
    //    output_rstd 由 x1 和 x2 的 requires_grad 决定
    bool output_rstd = x1.requires_grad() && x2.requires_grad();
    c10::SmallVector<int64_t, op_infer::SIZE> rstd_shape;
    if (output_rstd) {
        int64_t rstd_dim = x1.dim() - gamma.dim();
        for (int64_t i = 0; i < x1.dim(); i++) {
            if (i < rstd_dim) {
                rstd_shape.emplace_back(x1.size(i));
            } else {
                rstd_shape.emplace_back(1);
            }
        }
    } else {
        rstd_shape.emplace_back(0);
    }

    // 5. 字符串参数转换
    char *round_mode_ptr = const_cast<char *>(round_mode.data());

    // 6. 分配输出 tensor
    aclDataType y_acltype;
    at::Tensor y;
    bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    if (special_output_type) {
        int64_t y_last_dim_val = y_shape[x1.dim() - 1];
        TORCH_CHECK(y_last_dim_val % FP4_IN_UINT8_NUM == 0,
                    "The last dim of x1 must be divisible by 2 when "
                    "dst_type is float4_e2m1fn_x2 or float4_e1m2fn_x2" + OPS_ERROR(ErrCode::PARAM));
        y_shape[x1.dim() - 1] = y_last_dim_val / FP4_IN_UINT8_NUM;
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
        y_acltype = c10_npu::GetAclDataType(dst_type);
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
    }

    at::Tensor x_out = npu_preparation::apply_tensor_without_format(x_out_shape, x1.options().dtype(x1.scalar_type()));
    at::Tensor mxscale_out = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));
    at::Tensor rstd_out = npu_preparation::apply_tensor_without_format(rstd_shape, c10::dtype(at::ScalarType::Float));

    // 7. TensorWrapper 绑定真实 dtype
    TensorWrapper y_wrapper = {y, y_acltype};
    TensorWrapper mxscale_wrapper = {mxscale_out, aclDataType::ACL_FLOAT8_E8M0};

    // 8. 调用 aclnn 接口（支持 mock 模式）
#ifdef MOCK_ACLNN
    EXEC_NPU_CMD_MOCK(aclnnAddRmsNormDynamicMxQuant, x1, x2, gamma, beta, epsilon, scale_alg,
                      round_mode_ptr, y_acltype, output_rstd, y_wrapper, x_out, mxscale_wrapper, rstd_out);
#else
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicMxQuant, x1, x2, gamma, beta, epsilon, scale_alg,
                 round_mode_ptr, y_acltype, output_rstd, y_wrapper, x_out, mxscale_wrapper, rstd_out);
#endif

    return std::make_tuple(y, x_out, mxscale_out, rstd_out);
}

} // namespace op_api