
 15 
 16 #include <torch/library.h>
 17 #include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
 18 
 
 36 TORCH_LIBRARY_IMPL(npu, AutogradPrivateUse1, m)
 37 {
 38     m.impl("npu_add_rms_norm", torch::autograd::autogradNotImplementedFallback());
 39     m.impl("npu_apply_rotary_pos_emb", torch::autograd::autogradNotImplementedFallback());
 40     m.impl("npu_interleave_rope", torch::autograd::autogradNotImplementedFallback());
 41     m.impl("npu_moe_gating_top_k_softmax", torch::autograd::autogradNotImplementedFallback());
 42 }
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
~                                                                                                                  