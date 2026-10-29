// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Copyright (c) 2026 Ascend Community.
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "ATen/core/grad_mode.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// conj(z) = real(z) + i * (-imag(z))。
//
// CANN 无 aclnnConj，采用方案D（mask 广播乘法）组合实现：
//   view_as_real(t) 是与 complex 共享 storage 的 (...,2) 实视图（连续 t 下连续）；
//   mul(view_as_real(t), [1,-1], out=view_as_real(t)) 就地翻转虚部符号，
//   aclnnMul 广播乘在连续视图上原位 0 额外分配（实测，CANN 9.2.0 / 910B4）。
//
// 对比方案C（copy_ + imag().neg_()，已弃用）：neg_/mul_(-1) 对 stride-2 虚部视图
// 内部搬运连续，产生 1× 全张量工作区（见 Spec §8.3）；方案D 在连续 (...,2) 视图上
// 无额外分配，内存与 GPU 持平：functional=1×（仅输出）、in-place=0、.out=0。
//
// 非连续（转置/切片）输入：先搬一次连续临时（1×），原位翻转，再拷回原 storage，
// 避免 aclnnMul 对非连续 out 的内部双份搬运（实测直 mul 达 2×）。
//
// is_conj 语义（Spec §4.5-3 实测结论）：
//   - functional  is_conj：物化底层原值 P（self.conj().clone()，与 GPU _conj_physical 一致）；
//   - .out       is_conj：copy_ 感知 conj 位读出逻辑值 conj(P)，翻转虚部 → P；
//   - in-place   is_conj：经物理别名 self.conj() 翻转物理 storage → storage=conj(P)、
//                 保留 conj 位、逻辑值=P（与 GPU/CPU 官方 conj_physical_ 一致）。

namespace {

at::Tensor imag_flip_mask(const at::Tensor& t) {
    // mask dtype 跟随 view_as_real 的实类型（complex64→float32、complex128→float64）
    at::Tensor sr = at::view_as_real(t);
    return at::tensor({1.0, -1.0},
                      at::TensorOptions().dtype(sr.scalar_type()).device(t.device()));
}

// 就地翻转 complex t 的虚部符号（写回共享 storage，保留 conj 位）。
// physical 必须是 t 存储的非-conj 别名：t 本身（非 is_conj）或 t.conj()（is_conj）。
void conj_flip_inplace(at::Tensor& physical) {
    at::Tensor pr = at::view_as_real(physical);
    at::Tensor mask = imag_flip_mask(physical);
    if (pr.is_contiguous()) {
        // aclnnMul 连续 (...,2) 视图原位：0 额外分配
        at::mul_out(pr, pr, mask);
    } else {
        // 非连续：搬一次连续临时（1×），原位翻转，再拷回原 storage
        at::Tensor pc = pr.contiguous();
        at::mul_out(pc, pc, mask);
        pr.copy_(pc);
    }
}
}  // namespace

at::Tensor& conj_physical_out(const at::Tensor& self, at::Tensor& result) {
    // 官方 conj_physical 无 autograd 公式（不可微）。no_grad 守卫保证内部
    // mul/copy 不因 self 带 requires_grad 触发 "out=... AD" 报错，且结果不带
    // grad_fn —— 与 GPU/CPU 一致：错误推迟到结果上的 backward() 才抛。
    at::NoGradGuard guard;
    // 实数 .out：对齐 GPU —— 拷贝而非原样返回
    if (!self.is_complex()) {
        if (result.sizes() != self.sizes()) {
            result.resize_as_(self);
        }
        result.copy_(self);
        return result;
    }
    npu_preparation::check_tensor({self}, result, self.scalar_type(), self.sizes());
    if (self.is_conj()) {
        // .out 直接以 is_conj 输入调用：copy_ 感知 conj 位，读出逻辑值 conj(P)，
        // 再翻转虚部 → 底层原值 P（对齐 GPU/CPU 官方语义）。
        result.copy_(self);
        conj_flip_inplace(result);
    } else {
        // 非 is_conj：mul(view_as_real(self), mask, out=view_as_real(result)) 直达，
        // 跳过 copy_；self 非连续时先搬连续。
        at::Tensor sr = at::view_as_real(self);
        at::Tensor rr = at::view_as_real(result);
        at::Tensor mask = imag_flip_mask(self);
        if (sr.is_contiguous()) {
            at::mul_out(rr, sr, mask);
        } else {
            at::Tensor sc = sr.contiguous();
            at::mul_out(rr, sc, mask);
        }
    }
    return result;
}

at::Tensor conj_physical(const at::Tensor& self) {
    // 官方 conj_physical 不可微：no_grad 守卫（同 conj_physical_out 注释）。
    // 实数输入功能式仍为别名返回（对齐官方 composite conj_physical）。
    at::NoGradGuard guard;
    // 功能式实数：别名返回（对齐官方 composite conj_physical）
    if (!self.is_complex()) {
        return self;
    }
    // is_conj：物化底层原值（对齐官方 _conj_physical，不落设备 kernel）
    if (self.is_conj()) {
        return self.conj().clone();
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options());
    return op_api::conj_physical_out(self, result);
}

at::Tensor& conj_physical_(at::Tensor& self) {
    // 原地实数：no-op（对齐官方 conj_physical_）
    if (!self.is_complex()) {
        return self;
    }
    // is_conj：对物理 storage 翻转虚部（self.conj() 为物理别名、清 conj 位），
    // 写回后 storage=conj(P)、conj 位保留、逻辑值=P，与 GPU/CPU 官方一致。
    at::Tensor physical = self.is_conj() ? self.conj() : self;
    conj_flip_inplace(physical);
    return self;
}

} // namespace op_api
