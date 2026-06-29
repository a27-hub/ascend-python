// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#pragma once

// PyTorch 2.13 removed named-tensor support (at::Dimname / at::DimnameList /
// at::namedinference / ATen/NamedTensorUtils.h). op-plugin's named-tensor
// operator overloads are gated out of registration for v2.13+, but the dead
// code still references these symbols and must compile.
//
// Forward to torch_npu's single canonical compat header so a translation unit
// that pulls in both this and the core header never sees a duplicate
// definition of the stand-in symbols. op-plugin is always built as part of
// torch_npu, so the path below is on the include search path.
#include "torch_npu/csrc/_compat/NamedTensorCompat.h"
