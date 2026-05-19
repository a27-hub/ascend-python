// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include <limits>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {

namespace {

const int64_t RANDOM_DOUBLE_MAX = 1LL << 53;
const int64_t RANDOM_HALF_MAX = 1LL << 11;
const int64_t RANDOM_FLOAT_MAX = 1LL << 24;
const int64_t RANDOM_BFLOAT16_MAX = 1LL << 8;

}  // namespace
std::map<at::ScalarType, int64_t> initialize_dtype_max_value_map()
{
    return {
        {at::kHalf, RANDOM_HALF_MAX + 1},
        {at::kFloat, RANDOM_FLOAT_MAX + 1},
        {at::kDouble, RANDOM_DOUBLE_MAX + 1},
        {at::kInt, std::numeric_limits<int>::max()},
        {at::kShort, std::numeric_limits<int16_t>::max()},
        {at::kChar, std::numeric_limits<int8_t>::max()},
        {at::kByte, std::numeric_limits<uint8_t>::max()},
        {at::kLong, std::numeric_limits<long>::max()},
        {at::kBFloat16, RANDOM_BFLOAT16_MAX + 1},
        {at::kBool, 1}
    };
};

static std::map<at::ScalarType, int64_t> DTYPE_MAX_VALUE_MAP = initialize_dtype_max_value_map();

int64_t get_dtype_max_value(at::ScalarType dtype)
{
    auto iter = DTYPE_MAX_VALUE_MAP.find(dtype);
    TORCH_CHECK(iter != DTYPE_MAX_VALUE_MAP.end(),
                "self scalar_type:", dtype, "is not surpported.", OPS_ERROR(ErrCode::TYPE));
    return iter->second;
}

static const int64_t BLOCK_SIZE = 256;
static const int64_t MAX_THREADS_PER_MULTI_PROCESSOR = 2048;
static const int64_t MAX_PROCESSOR_COUNT = 78;
static const int64_t UNROLL_2 = 2;
static const int64_t UNROLL_4 = 4;
static const int64_t RAND_OFFSET_PER_CALL = 4;
static const int64_t INT32_MAX_VALUE = 2147483647LL;
static const int64_t MAX_DIMS = 8;
static const int64_t RAND_INT64_THRESHOLD = 268435456LL;

struct TensorIterInfo {
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
    int64_t ndim;
    int64_t numel;
    int64_t elementSize;
    int unroll;
    
    TensorIterInfo() : ndim(0), numel(0), elementSize(0), unroll(UNROLL_4) {
        for (int i = 0; i < MAX_DIMS; i++) {
            shape[i] = 1;
            strides[i] = 0;
        }
    }
    
    bool CanUse32BitIndexing() const {
        if (numel > INT32_MAX_VALUE) {
            return false;
        }
        int64_t maxOffset = 1;
        for (int64_t i = 0; i < ndim; i++) {
            if (shape[i] > 1) {
                maxOffset += (shape[i] - 1) * std::abs(strides[i]) * elementSize;
            }
        }
        return maxOffset <= INT32_MAX_VALUE;
    }
    
    int64_t GetDimToSplit() const {
        int64_t maxExtent = -1;
        int64_t splitDim = -1;
        for (int64_t dim = ndim - 1; dim >= 0; dim--) {
            if (shape[dim] >= 2) {
                int64_t extent = (shape[dim] - 1) * std::abs(strides[dim]) * elementSize;
                if (extent > maxExtent) {
                    maxExtent = extent;
                    splitDim = dim;
                }
            }
        }
        return splitDim;
    }
    
    void Narrow(int64_t dim, int64_t start, int64_t size) {
        if (dim < 0 || dim >= ndim || size < 1) {
            return;
        }
        numel = numel / shape[dim] * size;
        shape[dim] = size;
    }
};

inline int64_t CalcCounterOffset(int64_t nelem, int unroll) {
    unsigned int blocksPerSm = MAX_THREADS_PER_MULTI_PROCESSOR / BLOCK_SIZE;
    unsigned int gridX = (nelem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gridX = std::min((unsigned int)MAX_PROCESSOR_COUNT * blocksPerSm, gridX);
    return ((nelem - 1) / (BLOCK_SIZE * gridX * unroll) + 1) * RAND_OFFSET_PER_CALL;
}

inline int64_t CalcSplitCounterOffset(const TensorIterInfo& iter) {
    if (iter.CanUse32BitIndexing()) {
        return CalcCounterOffset(iter.numel, iter.unroll);
    }
    
    TensorIterInfo cur = iter;
    int64_t splitDim = cur.GetDimToSplit();
    if (splitDim < 0) {
        return CalcCounterOffset(iter.numel, iter.unroll);
    }
    
    int64_t leftSize = cur.shape[splitDim] / 2;
    int64_t rightSize = cur.shape[splitDim] - leftSize;
    
    TensorIterInfo left = cur;
    left.Narrow(splitDim, 0, leftSize);
    
    cur.Narrow(splitDim, leftSize, rightSize);
    
    return CalcSplitCounterOffset(left) + CalcSplitCounterOffset(cur);
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
    TORCH_CHECK((var) >= (min) && (var) <= (max), (name), " is out of bounds for ", (dtype)); \

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype) \
    if ((var) < -(1LL << (digits)) || (var) > (1LL << (digits))) { \
        TORCH_WARN((name), " is out of bounds [-(2^", (digits), "), 2^", (digits), "]. ", \
            "Due to precision limitations ", (dtype), " can support discrete uniform distribution only within this range. ", \
            "This warning will become an error in next release, please fix the code in advance"); \
    }

void check_random_bounds(caffe2::TypeMeta dtype, int64_t from, int64_t to)
{
    TORCH_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);

    const auto scalar_type = c10::typeMetaToScalarType(dtype);
    if (at::isFloatingType(scalar_type)) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "check_random_fp_bounds", [&] {
            const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
            const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
            CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
            CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);

            constexpr auto digits = std::numeric_limits<scalar_t>::digits;
            WARN_OUT_OF_BOUNDS(from, "from", digits, dtype);
            WARN_OUT_OF_BOUNDS((to - 1), "to - 1", digits, dtype);
        });
    } else if (scalar_type == at::kUInt64) {
        const auto min = static_cast<int64_t>(std::numeric_limits<uint64_t>::min());
        const auto max = static_cast<int64_t>(std::numeric_limits<uint64_t>::max());
        TORCH_CHECK(static_cast<uint64_t>(from) >= static_cast<uint64_t>(min) &&
                        static_cast<uint64_t>(from) <= static_cast<uint64_t>(max),
                    "from is out of bounds for ", dtype);
        TORCH_CHECK(static_cast<uint64_t>(to - 1) >= static_cast<uint64_t>(min) &&
                        static_cast<uint64_t>(to - 1) <= static_cast<uint64_t>(max),
                    "to - 1 is out of bounds for ", dtype);
    } else if (at::isIntegralType(scalar_type, true)) {
        if (scalar_type == at::kBool) {
            const auto min = 0;
            const auto max = 1;
            CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
            CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);
        } else {
            AT_DISPATCH_ALL_TYPES_AND2(at::kUInt16, at::kUInt32, scalar_type, "check_random_integral_bounds", [&] {
                if constexpr (std::is_integral_v<scalar_t>) {
                    const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
                    const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
                    CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
                    CHECK_OUT_OF_BOUNDS((to - 1), "to - 1", min, max, dtype);
                }
            });
        }
    } else {
        TORCH_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
    }
}

at::Tensor& random_op_api_(at::Tensor& self, int64_t from, int64_t to, c10::optional<at::Generator> generator)
{
    check_random_bounds(self.dtype(), from, to);
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();

    TensorIterInfo iterInfo;
    iterInfo.ndim = self.dim();
    iterInfo.numel = self.numel();
    iterInfo.elementSize = self.itemsize();
    iterInfo.unroll = ((to - from) >= RAND_INT64_THRESHOLD) ? UNROLL_2 : UNROLL_4;
    for (int64_t i = 0; i < iterInfo.ndim && i < MAX_DIMS; i++) {
        iterInfo.shape[i] = self.size(i);
        iterInfo.strides[i] = self.stride(i);
    }
    
    int64_t counterOffset = CalcCounterOffset(iterInfo.numel, iterInfo.unroll);
    if (!iterInfo.CanUse32BitIndexing()) {
        counterOffset += CalcSplitCounterOffset(iterInfo);
    }

    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(counterOffset);
        EXEC_NPU_CMD(aclnnInplaceRandom, self, from, to, pair.first, pair.second);
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(counterOffset);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnInplaceRandomTensor, self, from, to, *seed_ptr, *offset_ptr, offset_intragraph);
#endif
    }
    return self;
}

at::Tensor& random_max_range_op_api_(at::Tensor& self, c10::optional<at::Generator> generator)
{
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();

    TensorIterInfo iterInfo;
    iterInfo.ndim = self.dim();
    iterInfo.numel = self.numel();
    iterInfo.elementSize = self.itemsize();
    iterInfo.unroll = (self.scalar_type() == at::kLong) ? UNROLL_2 : UNROLL_4;
    for (int64_t i = 0; i < iterInfo.ndim && i < MAX_DIMS; i++) {
        iterInfo.shape[i] = self.size(i);
        iterInfo.strides[i] = self.stride(i);
    }
    
    int64_t counterOffset = CalcCounterOffset(iterInfo.numel, iterInfo.unroll);
    if (!iterInfo.CanUse32BitIndexing()) {
        counterOffset += CalcSplitCounterOffset(iterInfo);
    }

    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(counterOffset);
        EXEC_NPU_CMD(aclnnInplaceRandomMaxRange, self, pair.first, pair.second);
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(counterOffset);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnInplaceRandomMaxRangeTensor, self, *seed_ptr, *offset_ptr, offset_intragraph);
#endif
    }
    return self;
}

at::Tensor& random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                    c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, from, to, generator));
    int64_t to_ = to.value_or(get_dtype_max_value(self.scalar_type()));
    random_op_api_(self, from, to_, generator);
    return self;
}

at::Tensor& random_(at::Tensor& self, int64_t to, c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, to, generator));
    int64_t from = 0;
    random_op_api_(self, from, to, generator);
    return self;
}

at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> generator)
{
    if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950 && self.scalar_type() != at::kDouble) {
        DO_COMPATIBILITY(aclnnInplaceRandomMaxRange, acl_op::random_(self, generator));
        random_max_range_op_api_(self, generator);
    } else {
        DO_COMPATIBILITY(aclnnInplaceRandom, acl_op::random_(self, generator));
        int64_t from = 0;
        int64_t to = get_dtype_max_value(self.scalar_type());
        random_op_api_(self, from, to, generator);
    }
    return self;
}

}
