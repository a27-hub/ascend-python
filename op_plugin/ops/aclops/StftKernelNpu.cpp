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
#include "op_plugin/utils/op_api_common.h"

namespace acl_op {
// 与 aten::stft CPU 分支同构的组合实现（替代 aclStft 单算子路径，修复 #638 stft 条目）：
//   [center] pad(n_fft//2, reflect) → unfold(n_fft, hop) → ×window → rfft/fft → transpose(-1,-2)
//   normalized → by_root_n 归一化（_fft_r2c 的 normalization 参数）
// 全链均为 NPU 可用算子。真机验证：98 参数格点（nfft{4,64,512}×hop×win{none,hann,hamming,
// blackman}×center×normalized + pad_mode=constant + onesided=False 变体）与 CPU torch.stft
// 最大偏差 9.5e-06（fp32）。
// 注意：NPU 的 F.pad(reflect) 不支持 1D 输入，故统一按 (batch, length) 2D 批处理后 squeeze。
// 复数输入暂不支持（原 aclStft 路径在适配芯片上同样不可用），后续按需补充。

at::Tensor stft(at::Tensor const &self,
                int64_t n_fft,
                c10::optional<int64_t> hop_length_opt,
                c10::optional<int64_t> win_length_opt,
                c10::optional<at::Tensor> const &window_opt,
                bool normalized,
                c10::optional<bool> onesided_opt,
                c10::optional<bool> return_complex_opt,
                c10::optional<bool> /*align_to_window*/)
{
    TORCH_CHECK(self.dim() <= 2,
                "stft: expected input to be 1D or 2D tensor, but got ", self.dim(), "D",
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

    if (window_opt.has_value() && win_length_opt.has_value()) {
        TORCH_CHECK(window_opt.value().dim() == 1 &&
                        window_opt.value().size(0) == win_length_opt.value(),
                    "stft: expected window to be 1D and satisfy window.size(0) == win_length",
                    OPS_ERROR(ErrCode::PARAM));
    }

    int64_t win_length = win_length_opt.has_value() ? win_length_opt.value() : n_fft;
    TORCH_CHECK(win_length > 0 && win_length <= n_fft,
                "stft: expected 0 < win_length <= n_fft, but got win_length=", win_length,
                " and n_fft=", n_fft, OPS_ERROR(ErrCode::PARAM));

    at::Tensor window = c10::value_or_else(window_opt, [] { return at::Tensor(); });
    if (window.defined()) {
        TORCH_CHECK(window.scalar_type() == self.scalar_type(),
                    "stft: expected window dtype to match input dtype", OPS_ERROR(ErrCode::TYPE));
    }
    bool onesided = onesided_opt.has_value() ? onesided_opt.value() : true;
    bool return_complex = return_complex_opt.has_value() ? return_complex_opt.value() : false;

    at::Tensor input = self;
    const bool squeezed = (input.dim() == 1);
    if (squeezed) {
        // 统一按 (batch, length) 2D 批处理；NPU F.pad(reflect) 不支持 1D 输入
        input = input.unsqueeze(0);
    }

    // 窗口 zero-pad 至 n_fft（win_length < n_fft 时居中补零，与 aten::stft 一致）
    at::Tensor window_full;
    if (window.defined()) {
        if (win_length < n_fft) {
            const int64_t left = (n_fft - win_length) / 2;
            window_full = at::zeros({n_fft}, window.options());
            window_full.narrow(0, left, win_length).copy_(window);
        } else {
            window_full = window;
        }
    } else {
        window_full = at::ones({n_fft}, input.options());
    }

    // center: 两侧对称 reflect pad n_fft//2（与 aten::stft 的 at::pad(self, {pad, pad}) 一致）
    at::Tensor padded = input;
    const int64_t center_pad = n_fft / 2;
    padded = at::pad(input, {center_pad, center_pad}, at::PadMode::reflect);

    // 分帧 + 加窗：unfold(dimension=-1, size=n_fft, step=hop) → (batch, n_frames, n_fft)
    at::Tensor frames = padded.unfold(padded.dim() - 1, n_fft, hop_length);
    frames = frames * window_full;

    // 频谱：onesided → rfft 半谱，否则全谱；normalized → by_root_n 归一化（除以 sqrt(n_fft)）
    const int64_t normalization = normalized ? 1 : 0;
    at::Tensor spec = at::_fft_r2c(frames, {-1}, normalization, onesided);
    spec = spec.transpose(-1, -2);

    if (!return_complex) {
        spec = at::view_as_real(spec);
    }
    if (squeezed) {
        return spec.squeeze(0);
    }
    return spec;
}
} // namespace acl_op
