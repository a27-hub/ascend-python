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
    TORCH_CHECK(batch_sizes.dim() >= 1, "Input batch_sizes dim number must larger than one."
    + OPS_ERROR(ErrCode::PARAM));

    // Move batch_sizes to CPU and get pointer
    auto batch_sizes_cpu = batch_sizes.to("cpu").contiguous();
    int64_t *batch_sizes_ptr = batch_sizes_cpu.data_ptr<int64_t>();
    TORCH_CHECK(batch_sizes_ptr != nullptr, "batch_sizes is null" + OPS_ERROR(ErrCode::PARAM));

    int64_t max_batch_size = batch_sizes_ptr[0];
    int64_t max_real_seq_length = batch_sizes_cpu.size(0);
    int64_t max_seq_length = max_real_seq_length;

    if (total_length > 0) {
        TORCH_CHECK(total_length >= max_seq_length,
            "Expected total_length to be at least the length of the longest "
            "sequence in input, but got total_length=",
            total_length,
            " and "
            "max sequence length being ",
            max_seq_length, OPS_ERROR(ErrCode::PARAM));
        max_seq_length = total_length;
    }

    // Create output tensor with correct size: [max_seq_length, max_batch_size, *input.sizes()[1:]]
    // Use at::full to create a tensor filled with padding_value
    std::vector<int64_t> output_size;
    output_size.reserve(input.dim() + 1);
    output_size.push_back(max_seq_length);
    output_size.push_back(max_batch_size);
    auto input_sizes = input.sizes();
    for (int64_t i = 1; i < input.dim(); i++) {
        output_size.push_back(input_sizes[i]);
    }
    auto output = at::full(output_size, padding_value, input.options());

    // Prepare view size for temporary tensors (will be modified during iteration)
    std::vector<int64_t> tmp_view_size = output_size;

    // Create lengths tensor
    at::Tensor lengths_t = at::empty(max_batch_size, batch_sizes_cpu.options());
    int64_t *lengths_ptr = lengths_t.mutable_data_ptr<int64_t>() + max_batch_size - 1;

    // Iterate through batch_sizes and fill output tensor
    int64_t data_offset = 0;
    int64_t prev_batch_size = max_batch_size;
    int64_t prev_i = 0;

    for (int64_t i = 0; i <= max_real_seq_length; ++i) {
        int64_t cur_batch_size = (i != max_real_seq_length) ? batch_sizes_ptr[i] : 0;
        // When batch_size changes, copy data from input to output
        if (cur_batch_size != prev_batch_size) {
            int64_t l = prev_batch_size * (i - prev_i);
            auto tmp = input.slice(0, data_offset, data_offset + l);
            tmp_view_size[0] = i - prev_i;
            tmp_view_size[1] = prev_batch_size;
            output.slice(0, prev_i, i).slice(1, 0, prev_batch_size).copy_(tmp.view(tmp_view_size));
            data_offset += l;
            prev_i = i;
        }
        // Record lengths for each sequence that ends at this step
        int64_t dec = prev_batch_size - cur_batch_size;
        if (dec > 0) {
            for (int64_t j = 0; j < dec; j++) {
                (*lengths_ptr--) = i;
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
