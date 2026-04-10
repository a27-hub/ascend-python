// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/OpInterface.h"

namespace op_api {
const int DIMENSION_2D = 2;
std::tuple<at::Tensor, at::Tensor> _pack_padded_sequence(const at::Tensor &input, const at::Tensor &lengths,
                                                         bool batch_first)
{
    TORCH_CHECK(input.dim() >= DIMENSION_2D, "Input must have two dims.", input.dim(), OPS_ERROR(ErrCode::PARAM));
    // get the size of B and T, the input size is [T, B, *] or [B, T, *]
    auto batchsize = batch_first ? input.size(0) : input.size(1);

    TORCH_CHECK(input.numel() > 0, "Cannot pack empty tensors." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.numel() < std::numeric_limits<int64_t>::max(),
                "Input tensor contain more than the max number of int64." + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(lengths.size(0) == batchsize, "Expected 'len(lengths)' to be equal to batch_size, but got ",
                lengths.size(0), " (batch_size=", batchsize, ")" + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(lengths.device().type() == at::kCPU,
                "'lengths' argument should be a CPU tensor, but got ",
                lengths.device().str(), " tensor" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(lengths.dim() == 1 && lengths.scalar_type() == at::kLong,
                "'lengths' argument should be a 1D CPU int64 tensor, but got ",
                lengths.dim(), "D ", lengths.device().str(), " ", lengths.scalar_type(), " tensor",
                OPS_ERROR(ErrCode::PARAM));

    auto lengths_contiguous = lengths.contiguous();
    auto lengths_vec = lengths_contiguous.data_ptr<int64_t>();
    TORCH_CHECK(lengths_vec != nullptr && lengths_vec[batchsize - 1] > 0,
                "Length of all samples has to be greater than 0, but found an element "
                "in 'lengths' that is <= 0" + OPS_ERROR(ErrCode::PARAM));

    for (int64_t i = 0; i < batchsize - 1; ++i) {
        TORCH_CHECK(lengths_vec[batchsize - 1 - i] <= lengths_vec[batchsize - 2 - i],
                    "`lengths` array must be sorted in decreasing order when "
                    "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
                    "to pack_padded_sequence and/or pack_sequence to sidestep this "
                    "requirement if you do not need ONNX exportability.",
                    OPS_ERROR(ErrCode::PARAM));
    }

    auto input_t = batch_first ? input.transpose(0, 1) : input;
    std::vector<at::Tensor> steps;
    steps.reserve(batchsize);

    at::Tensor batchsizes = at::empty({lengths_vec[0]}, lengths.options());
    auto batchsize_vec = batchsizes.data_ptr<int64_t>();
    TORCH_CHECK(batchsize_vec != nullptr, "batchsizes is null" + OPS_ERROR(ErrCode::PARAM));

    std::vector<int64_t> step_shape;
    step_shape.reserve(input_t.dim() - 1);
    step_shape.emplace_back(-1);
    for (int64_t i = 2; i < input_t.dim(); ++i) {
        step_shape.emplace_back(input_t.size(i));
    }

    int64_t prev_len = 0;
    for (int64_t i = 0; i < batchsize; ++i) {
        int64_t cur_len = lengths_vec[batchsize - 1 - i];
        if (cur_len > prev_len) {
            auto current_batch_size = batchsize - i;
            steps.push_back(
                input_t.slice(0, prev_len, cur_len)
                    .slice(1, 0, current_batch_size)
                    .contiguous()
                    .view(step_shape));
            for (int64_t j = 0; j < (cur_len - prev_len); ++j) {
                (*batchsize_vec++) = current_batch_size;
            }
            prev_len = cur_len;
        }
    }

    auto output = at::cat(steps, 0);
    return std::tie(output, batchsizes);
}
} // namespace op_api
