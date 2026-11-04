// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include <string>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
constexpr size_t ASSERT_ASYNC_MAX_MSG_LEN = 255;

void _assert_async(const at::Tensor& self) {
  op_api::_assert_async(self, c10::string_view{});
}

void _assert_async(const at::Tensor& self, c10::string_view assert_msg) {
  TORCH_CHECK(
      c10_npu::GetSocVersion() == c10_npu::SocVersion::Ascend950,
      "_assert_async is supported only on Ascend950.",
      OPS_ERROR(ErrCode::NOT_SUPPORT));

  TORCH_CHECK(
      self.numel() == 1,
      "_assert_async: Expected one element, but got ",
      self.numel(),
      OPS_ERROR(ErrCode::PARAM));

  TORCH_CHECK(
      assert_msg.size() < ASSERT_ASYNC_MAX_MSG_LEN,
      "_assert_async: assert_msg length must be less than ",
      ASSERT_ASYNC_MAX_MSG_LEN,
      " bytes, but got ",
      assert_msg.size(),
      OPS_ERROR(ErrCode::PARAM));

  std::string msg = assert_msg.empty() ? std::string() : std::string(assert_msg.data(), assert_msg.size());
  char* msg_ptr = const_cast<char*>(msg.c_str());
  EXEC_NPU_CMD(aclnnAssertAsync, self, msg_ptr);
}
} // namespace op_api
