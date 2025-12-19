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

#include <chrono>
#include <filesystem>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"
#include "third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"

namespace op_api {

namespace {
struct HostFuncArgs {
    at::Tensor tensor;
    std::string path;
};
} // namespace

std::string get_tensor_shape(const at::Tensor& tensor)
{
    auto shape = tensor.sizes();
    std::ostringstream shape_oss;
    shape_oss << "shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_oss << shape[i];
        if (i != shape.size() - 1) shape_oss << ", ";
    }
    shape_oss << "]\n";
    return shape_oss.str();
}

void write_tensor_to_bin(const at::Tensor& tensor, size_t nbytes, const std::string& path)
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        std::ostringstream oss;
        oss << "Failed to open tensor binary file: " << path;
        throw std::runtime_error(oss.str());
    }

    int dtype = static_cast<int>(tensor.scalar_type());
    std::ostringstream dtype_oss;
    dtype_oss << "dtype: " << dtype << '\n';
    std::string dtype_str = dtype_oss.str();
    ofs.write(dtype_str.c_str(), dtype_str.size());
    if (!ofs.good()) {
        std::ostringstream oss;
        oss << "Failed to write tensor dtype to: " << path;
        throw std::runtime_error(oss.str());
    }

    std::string shape = get_tensor_shape(tensor);
    ofs.write(shape.c_str(), shape.size());
    if (!ofs.good()) {
        std::ostringstream oss;
        oss << "Failed to write tensor shape to: " << path;
        throw std::runtime_error(oss.str());
    }

    const void* src = tensor.const_data_ptr();
    ofs.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(nbytes));
    if (!ofs.good()) {
        std::ostringstream oss;
        oss << "Failed to write tensor data to: " << path;
        throw std::runtime_error(oss.str());
    }

    ofs.close();
    if (!ofs) {
        std::ostringstream oss;
        oss << "Failed to close file after write: " << path;
        throw std::runtime_error(oss.str());
    }
}

void save_tensor_callback(void* args)
{
    HostFuncArgs* host_func_args = static_cast<HostFuncArgs*>(args);
    TORCH_CHECK(host_func_args != nullptr, OPS_ERROR(ErrCode::VALUE));

    at::Tensor tensor = host_func_args->tensor;
    std::string path = host_func_args->path;
    at::Tensor tensor_contiguous = tensor.contiguous();
    auto out = at::empty_like(
        tensor_contiguous,
        tensor_contiguous.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);
    const size_t nbytes = static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
    if (nbytes == 0) {
        TORCH_NPU_WARN("nbytes of tensor is 0, skip tensor saving");
        return;
    }
    NPU_CHECK_ERROR(aclrtMemcpy(
        out.data_ptr(),
        nbytes,
        // tensor_contiguous.storage().unsafeGetStorageImpl()->data(),
        tensor_contiguous.data_ptr(),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    write_tensor_to_bin(out, nbytes, path);
}

std::string get_final_save_path(const at::Tensor& self, c10::optional<c10::string_view> save_path, int dev_index)
{
    constexpr int64_t MILLISECOND_WIDTH = 3;
    constexpr int64_t MILLISECONDS_PER_SECOND = 1000;

    std::string final_save_path;
    bool use_default_path = false;

    if (!save_path.has_value() || save_path.value().empty()) {
        std::string base_filename_prefix = "tensor";

        auto now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        std::tm* time_info = std::localtime(&current_time);
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() % MILLISECONDS_PER_SECOND; // in microsecond
        std::ostringstream timestamp_oss;
        timestamp_oss << std::put_time(time_info, "%Y%m%d%H%M%S")
                      << std::setw(MILLISECOND_WIDTH)
                      << std::setfill('0')
                      << millis;

        std::filesystem::path current_path = std::filesystem::current_path();
        std::ostringstream filename_oss;
        filename_oss << base_filename_prefix << "_" << timestamp_oss.str();
        final_save_path = (current_path / filename_oss.str()).string();
    } else {
        std::string path_str(save_path.value().data(), save_path.value().size());
        std::filesystem::path file_path(path_str);
        std::filesystem::path parent_dir = file_path.parent_path();
        TORCH_CHECK(std::filesystem::exists(parent_dir) && file_path.extension() == ".bin",
                    "Invalid save path or save format: ", path_str, OPS_ERROR(ErrCode::PARAM));
        final_save_path = (parent_dir / file_path.stem()).string();
    }

    std::ostringstream device_index_oss;
    device_index_oss << final_save_path;
    if (dev_index != -1) {
        device_index_oss << "_device_" << dev_index;
    }
    device_index_oss << ".bin";
    return device_index_oss.str();
}

void npu_save_tensor(const at::Tensor& self, c10::optional<c10::string_view> save_path)
{
    const auto dev_type = self.device().type();
    int dev_index = -1;
    TORCH_CHECK(dev_type == at::DeviceType::PrivateUse1,
                "Only support npu tensor dump", OPS_ERROR(ErrCode::TYPE));
    dev_index = self.device().index();
    std::string final_save_path = get_final_save_path(self, save_path, dev_index);

    auto acl_stream = c10_npu::getCurrentNPUStream();
    HostFuncArgs* args = new HostFuncArgs();
    args->tensor = self;
    args->path = final_save_path;

    OPS_CHECK_ERROR(c10_npu::acl::AclrtLaunchHostFunc(acl_stream, save_tensor_callback, args));
}
} // namespace op_api
