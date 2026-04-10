// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> _pad_packed_sequence(const at::Tensor &input, const at::Tensor &batch_sizes,
                                                        bool batch_first, const at::Scalar &padding_value,
                                                        int64_t total_length)
{
    auto batch_sizes_t = batch_sizes.contiguous();
    TORCH_CHECK(batch_sizes_t.dim() == 1 && batch_sizes_t.device().type() == at::kCPU &&
                    batch_sizes_t.scalar_type() == at::kLong,
                "'batch_sizes' should be a 1D CPU int64 tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(batch_sizes_t.numel() > 0, "batch_sizes can not be empty", OPS_ERROR(ErrCode::PARAM));

    int64_t *batch_size_vec = batch_sizes_t.data_ptr<int64_t>();
    TORCH_CHECK(batch_size_vec != nullptr, "batch_sizes is null" + OPS_ERROR(ErrCode::PARAM));

    int64_t max_batch_size = batch_size_vec[0];
    int64_t max_real_seq_length = batch_sizes_t.size(0);
    int64_t max_seq_length = max_real_seq_length;
    if (total_length > 0) {
        TORCH_CHECK(total_length >= max_seq_length,
                    "Expected total_length to be at least the length of the longest "
                    "sequence in input, but got total_length=",
                    total_length,
                    " and max sequence length being ",
                    max_seq_length,
                    OPS_ERROR(ErrCode::PARAM));
        max_seq_length = total_length;
    }

    std::vector<int64_t> output_size;
    output_size.reserve(input.dim() + 1);
    output_size.push_back(max_seq_length);
    output_size.push_back(max_batch_size);
    auto trailing_dims = input.sizes().slice(1);
    output_size.insert(output_size.end(), trailing_dims.begin(), trailing_dims.end());

    auto output = at::full(output_size, padding_value, input.options());

    std::vector<int64_t> tmp_view_size(output_size.begin(), output_size.end());

    auto lengths_t = at::empty(max_batch_size, batch_sizes_t.options());
    int64_t *lengths = lengths_t.mutable_data_ptr<int64_t>() + max_batch_size - 1;

    int64_t data_offset = 0;
    int64_t prev_batch_size = max_batch_size;
    int64_t prev_i = 0;
    for (int64_t i = 0; i <= max_real_seq_length; ++i) {
        int64_t cur_batch_size = i != max_real_seq_length ? batch_size_vec[i] : 0;
        if (cur_batch_size != prev_batch_size) {
            int64_t l = prev_batch_size * (i - prev_i);
            auto tmp = input.slice(0, data_offset, data_offset + l);
            tmp_view_size[0] = i - prev_i;
            tmp_view_size[1] = prev_batch_size;
            output.slice(0, prev_i, i).slice(1, 0, prev_batch_size).copy_(tmp.view(tmp_view_size));
            data_offset += l;
            prev_i = i;
        }

        int64_t dec = prev_batch_size - cur_batch_size;
        if (dec > 0) {
            for (int64_t j = 0; j < dec; ++j) {
                (*lengths--) = i;
            }
        }
        prev_batch_size = cur_batch_size;
    }

    if (batch_first) {
        output = output.transpose(0, 1);
    }

    return std::tie(output, lengths_t);
}
} // namespace op_api
