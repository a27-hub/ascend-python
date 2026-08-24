#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

#include <ATen/ops/_fused_adam.h>

#include <vector>

namespace op_api {
namespace {
void fused_adam_cpu_fallback(
	at::TensorList self,
	at::TensorList grads,
	at::TensorList exp_avgs,
	at::TensorList exp_avg_sqs,
	at::TensorList max_exp_avg_sqs,
	at::TensorList state_steps,
	const double lr,
	const double beta1,
	const double beta2,
	const double weight_decay,
	const double eps,
	const bool amsgrad,
	const bool maximize,
	const c10::optional<at::Tensor>& grad_scale,
	const c10::optional<at::Tensor>& found_inf)
{
	std::vector<at::Tensor> self_cpu;
	std::vector<at::Tensor> grads_cpu;
	std::vector<at::Tensor> exp_avgs_cpu;
	std::vector<at::Tensor> exp_avg_sqs_cpu;
	std::vector<at::Tensor> max_exp_avg_sqs_cpu;
	std::vector<at::Tensor> state_steps_cpu;
	self_cpu.reserve(self.size());
	grads_cpu.reserve(grads.size());
	exp_avgs_cpu.reserve(exp_avgs.size());
	exp_avg_sqs_cpu.reserve(exp_avg_sqs.size());
	max_exp_avg_sqs_cpu.reserve(max_exp_avg_sqs.size());
	state_steps_cpu.reserve(state_steps.size());

	for (const auto& tensor : self) {
		self_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : grads) {
		grads_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : exp_avgs) {
		exp_avgs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : exp_avg_sqs) {
		exp_avg_sqs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : max_exp_avg_sqs) {
		max_exp_avg_sqs_cpu.emplace_back(tensor.cpu());
	}
	for (const auto& tensor : state_steps) {
		state_steps_cpu.emplace_back(tensor.cpu());
	}

	c10::optional<at::Tensor> grad_scale_cpu =
		grad_scale.has_value() ? c10::optional<at::Tensor>(grad_scale->cpu()) : c10::nullopt;
	c10::optional<at::Tensor> found_inf_cpu =
		found_inf.has_value() ? c10::optional<at::Tensor>(found_inf->cpu()) : c10::nullopt;

	at::_fused_adam_(
		self_cpu,
		grads_cpu,
		exp_avgs_cpu,
		exp_avg_sqs_cpu,
		max_exp_avg_sqs_cpu,
		state_steps_cpu,
		lr,
		beta1,
		beta2,
		weight_decay,
		eps,
		amsgrad,
		maximize,
		grad_scale_cpu,
		found_inf_cpu);

	for (size_t i = 0; i < self.size(); ++i) {
		self[i].copy_(self_cpu[i]);
		exp_avgs[i].copy_(exp_avgs_cpu[i]);
		exp_avg_sqs[i].copy_(exp_avg_sqs_cpu[i]);
		if (amsgrad) {
			max_exp_avg_sqs[i].copy_(max_exp_avg_sqs_cpu[i]);
		}
	}
}
} // namespace

void _fused_adam_(
	at::TensorList self,
	at::TensorList grads,
	at::TensorList exp_avgs,
	at::TensorList exp_avg_sqs,
	at::TensorList max_exp_avg_sqs,
	at::TensorList state_steps,
	const double lr,
	const double beta1,
	const double beta2,
	const double weight_decay,
	const double eps,
	const bool amsgrad,
	const bool maximize,
	const c10::optional<at::Tensor>& grad_scale,
	const c10::optional<at::Tensor>& found_inf)
{
	if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950) {
		fused_adam_cpu_fallback(
			self,
			grads,
			exp_avgs,
			exp_avg_sqs,
			max_exp_avg_sqs,
			state_steps,
			lr,
			beta1,
			beta2,
			weight_decay,
			eps,
			amsgrad,
			maximize,
			grad_scale,
			found_inf);
		return;
	}

	bool is_same_size = (self.size() == grads.size() &&
					   self.size() == exp_avgs.size() &&
					   self.size() == exp_avg_sqs.size() &&
					   self.size() == state_steps.size() &&
					   (max_exp_avg_sqs.size() == 0 ||
					   self.size() == max_exp_avg_sqs.size()));
	if (!is_same_size) {
		TORCH_CHECK(false, "the size of tensor list should be same.");
	}

	// 融合Adam优化器：一次性调用aclnnFusedAdam算子（TensorList输入），
	// 替代逐参数多次调用aclnnApplyAdamV2，与aclnnFusedSgd的调用方式保持一致。
	// 注意：CANN侧FusedAdam核内会对state_steps做+1后再计算bias correction
	// （见fused_adam_regbase.h中 stepCount_ = state_steps + 1），而PyTorch在调用
	// _fused_adam_前已通过_foreach_add_将state_steps自增为当前步数t，因此这里先
	// 对每个state_step减1，保证bias correction使用的步数语义与PyTorch一致。
	std::vector<at::Tensor> state_steps_adjusted;
	state_steps_adjusted.reserve(state_steps.size());
	for (size_t i = 0; i < state_steps.size(); i++) {
		state_steps_adjusted.emplace_back(state_steps[i].sub(1));
	}
	at::TensorList state_steps_list(state_steps_adjusted);

	EXEC_NPU_CMD(
		aclnnFusedAdam,
		self,
		grads,
		exp_avgs,
		exp_avg_sqs,
		max_exp_avg_sqs,
		state_steps_list,
		grad_scale.value_or(at::Tensor()),
		found_inf.value_or(at::Tensor()),
		lr,
		beta1,
		beta2,
		weight_decay,
		eps,
		amsgrad,
		maximize);
}
}  // namespace op_api
