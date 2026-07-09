// op_plugin/ops/opapi/AutogradErrorFallback.cpp
#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <tuple>

// ============ 1. npu_apply_rotary_pos_emb 错误函数 ============
// 签名匹配: (Tensor query, Tensor key, Tensor cos, Tensor sin, str layout='BSND', str rotary_mode='half') -> (Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor> npu_apply_rotary_pos_emb_autograd_error(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::string& layout,
    const std::string& rotary_mode) {
    
    TORCH_CHECK(false, 
        "\n========== ERROR ==========\n"
        "npu_apply_rotary_pos_emb does NOT support backward propagation!\n"
        "Please ensure you are not calling .backward() on tensors\n"
        "that depend on this operator.\n"
        "============================\n");
    
    return std::make_tuple(query, key);
}

// ============ 2. npu_interleave_rope 错误函数 ============
// 签名匹配: (Tensor x, Tensor cos, Tensor sin) -> Tensor
at::Tensor npu_interleave_rope_autograd_error(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin) {
    
    TORCH_CHECK(false, 
        "\n========== ERROR ==========\n"
        "npu_interleave_rope does NOT support backward propagation!\n"
        "Please ensure you are not calling .backward() on tensors\n"
        "that depend on this operator.\n"
        "============================\n");
    
    return x;
}

// ============ 3. npu_moe_gating_top_k_softmax 错误函数 ============
// 签名匹配: (Tensor x, Tensor? finished=None, int k=1) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_gating_top_k_softmax_autograd_error(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& finished,
    int64_t k) {
    
    TORCH_CHECK(false, 
        "\n========== ERROR ==========\n"
        "npu_moe_gating_top_k_softmax does NOT support backward propagation!\n"
        "Please ensure you are not calling .backward() on tensors\n"
        "that depend on this operator.\n"
        "============================\n");
    
    return std::make_tuple(x, x, x);
}