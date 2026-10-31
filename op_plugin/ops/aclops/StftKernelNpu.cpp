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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
// 与 aten::stft 同构的组合实现（替代 aclStft 单算子路径，修复 #638 stft 条目）：
//   unfold(n_fft, hop) → ×window → rfft/fft → transpose(-1,-2)
// center padding 由 Python 层 torch.stft 在调用本 op 前完成（本 op schema 无 center/pad_mode，
// 仓库内 op_api 实现同样不 pad）；align_to_window=True 时按 aten SpectralOps.cpp 语义对输入做
// (n_fft - win_length) / 2 的 constant 补零，使窗口起点对齐 t=0。
// 窗口在 win_length < n_fft 时居中补零至 n_fft；normalized → _fft_r2c 的 normalization 参数。
// 全链均为 NPU 可用算子。复数输入暂不支持（原 aclStft 路径在适配芯片上同样不可用），后续按需补充。

at::Tensor stft(at::Tensor const &self,
                int64_t n_fft,
                c10::optional<int64_t> hop_length_opt,
                c10::optional<int64_t> win_length_opt,
                c10::optional<at::Tensor> const &window_opt,
                bool normalized,
                c10::optional<bool> onesided_opt,
                c10::optional<bool> return_complex_opt,
                c10::optional<bool> align_to_window_opt)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2,
                "stft: expected a 1D or 2D tensor, but got ", self.dim(), "D",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!self.is_complex(),
                "stft: complex input is not supported by the combined NPU kernel yet",
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(n_fft > 0 && n_fft <= self.size(-1),
                "stft: expected 0 < n_fft <= input.size(-1), but got n_fft=", n_fft,
                " and input.size(-1)=", self.size(-1), OPS_ERROR(ErrCode::PARAM));

    int64_t hop_length = hop_length_opt.has_value() ? hop_length_opt.value() : n_fft / 4;
    TORCH_CHECK(hop_length > 0, "stft: expected hop_length > 0, but got ", hop_length,
                OPS_ERROR(ErrCode::VALUE));

    int64_t win_length = win_length_opt.has_value() ? win_length_opt.value() : n_fft;
    TORCH_CHECK(win_length > 0 && win_length <= n_fft,
                "stft: expected 0 < win_length <= n_fft, but got win_length=", win_length,
                " and n_fft=", n_fft, OPS_ERROR(ErrCode::PARAM));

    at::Tensor window = c10::value_or_else(window_opt, [] { return at::Tensor(); });
    if (window.defined()) {
        TORCH_CHECK(window.dim() == 1 && window.size(0) == win_length,
                    "stft: expected a 1D window tensor of size equal to win_length=",
                    win_length, ", but got window with size ", window.sizes(),
                    OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(window.scalar_type() == self.scalar_type(),
                    "stft: expected window dtype to match input dtype", OPS_ERROR(ErrCode::TYPE));
    }
    const bool align_to_window = align_to_window_opt.value_or(false);
    bool onesided = onesided_opt.has_value() ? onesided_opt.value() : true;
    bool return_complex = return_complex_opt.has_value() ? return_complex_opt.value() : false;

    at::Tensor input = self;
    const bool squeezed = (input.dim() == 1);
    if (squeezed) {
        // 统一按 (batch, length) 2D 批处理
        input = input.unsqueeze(0);
    }

    // 窗口 zero-pad 至 n_fft（win_length < n_fft 时居中补零，与 aten::stft 一致）
    at::Tensor window_full = window;
    if (win_length < n_fft) {
        const int64_t left = (n_fft - win_length) / 2;
        window_full = at::zeros({n_fft}, window.defined() ? window.options() : input.options());
        if (window.defined()) {
            window_full.narrow(0, left, win_length).copy_(window);
        } else {
            window_full.narrow(0, left, win_length).fill_(1);
        }
    }

    // align_to_window: 仅在 center=False 时由上层传入；对输入做 constant 补零，
    // 使 win_length 窗口的起点与原始信号 t=0 对齐（aten 9 参 stft 走 mode="constant"）。
    if (align_to_window && win_length < n_fft) {
        const int64_t align_pad = (n_fft - win_length) / 2;
        input = at::pad(input, {align_pad, align_pad}, "constant");
    }

    // 分帧 + 加窗：unfold(dimension=-1, size=n_fft, step=hop) → (batch, n_frames, n_fft)
    const int64_t unfold_dim = input.dim() - 1;
    at::Tensor frames = at::unfold(input, unfold_dim, n_fft, hop_length);
    if (window_full.defined()) {
        frames = frames * window_full;
    }

    // 频谱：onesided → rfft 半谱，否则全谱；normalized → by_root_n 归一化（除以 sqrt(n_fft)）
    const int64_t normalization = normalized ? 1 : 0;
    at::Tensor spec = at::_fft_r2c(frames, {unfold_dim}, normalization, onesided);
    spec = spec.transpose(unfold_dim, unfold_dim - 1);

    if (!return_complex) {
        spec = at::view_as_real(spec);
    }
    if (squeezed) {
        return spec.squeeze(0);
    }
    return spec;
}
} // namespace acl_op
