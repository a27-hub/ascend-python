// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor &_amp_update_scale_(
    at::Tensor &current_scale,
    at::Tensor &growth_tracker,
    const at::Tensor &found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval)
{
    // aclnnAmpUpdateScale is not reliably available on some SoCs (e.g. Ascend950PR,
    // intermittent 561002/561103 at process start). Compose the scale state machine
    // with universally available basic ops, matching the CPU/CUDA reference:
    //   found_inf == 1            : scale *= backoff_factor, tracker = 0
    //   tracker + 1 == interval   : scale *= growth_factor,  tracker = 0
    //   otherwise                 : tracker += 1
    at::Tensor found = found_inf.ne(0);
    at::Tensor tracker_next = growth_tracker.add(1);
    at::Tensor reached = tracker_next.eq(growth_interval);
    at::Tensor scale_new = at::where(found, current_scale.mul(backoff_factor),
        at::where(reached, current_scale.mul(growth_factor), current_scale));
    at::Tensor tracker_new = at::where(found, at::zeros_like(growth_tracker),
        at::where(reached, at::zeros_like(growth_tracker), tracker_next));
    current_scale.copy_(scale_new);
    growth_tracker.copy_(tracker_new);
    return current_scale;
}
#endif

}  // namespace acl_op
