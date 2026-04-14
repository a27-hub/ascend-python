// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DVM_LAZY_FUSION_FLAGS_H
#define DVM_LAZY_FUSION_FLAGS_H

#include <cstdint>
#include <string>
#include <vector>

namespace lazy_fusion {
constexpr unsigned int OptLevel_0 = 0;  // Disable dvm
constexpr unsigned int OptLevel_1 = 1;  // Enable dvm
constexpr unsigned int OpLevel_0 = 0;
constexpr unsigned int OpLevel_1 = 1;

class LazyFusionFlags {
 public:
  LazyFusionFlags();
  ~LazyFusionFlags() = default;

  uint64_t opt_level{OptLevel_0};
  uint64_t op_level{OpLevel_0};
  bool dump_as_text{false};
  std::string dump_dir{"./lazy_fusion_dump"};
  bool synchronize{false};
  bool online_tuning{false};
  std::vector<std::string> disable_ops;
  std::vector<std::string> enable_ops;
  std::vector<std::string> enable_ops_only;
};
}  // namespace lazy_fusion
#endif  // DVM_LAZY_FUSION_FLAGS_H
