/**
 * Minimal ATB C++ runtime example for wrapper verification.
 *
 * This example is intended to run in a Linux + Ascend NPU + torch_npu
 * compatible environment. It validates that the public wrapper
 * `at_npu::native::atb::_npu_quantize_per_tensor` can be called and that
 * its output matches a fixed expected result.
 *
 * Build and run with the helper script in the same directory:
 *
 *   ./build_and_run.sh
 *
 * Runtime still depends on the full torch_npu-compatible closure, including
 * but not limited to:
 *   - libtorch_npu.so
 *   - libtorch_python.so
 *   - libatb.so
 *   - CANN / NNAL runtime libraries
 */

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <iostream>
#include <stdexcept>

#include "op_plugin/include/atb_ops.h"

namespace {

constexpr int64_t kBatch = 1;
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 4;
constexpr int64_t kNumel = kBatch * kRows * kCols;

at::Tensor make_input_x_cpu()
{
    return at::tensor(
               {-2.0, -1.0, -0.5, 0.0,
                0.5,  1.0,  1.5,  2.0,
                -1.5, -0.25, 0.25, 0.75,
                1.25, 1.75,  2.25, -2.25},
               at::TensorOptions().dtype(at::kFloat))
        .reshape({kBatch, kRows, kCols})
        .to(at::kHalf);
}

at::Tensor make_scale_cpu()
{
    return at::full({kRows, kCols}, 0.5, at::TensorOptions().dtype(at::kFloat)).to(at::kHalf);
}

at::Tensor make_offset_cpu()
{
    return at::ones({kRows, kCols}, at::TensorOptions().dtype(at::kChar));
}

at::Tensor calculate_expected(
    const at::Tensor& input_x_cpu,
    const at::Tensor& scale_cpu,
    const at::Tensor& offset_cpu)
{
    at::Tensor expected = at::round(input_x_cpu.to(at::kFloat) / scale_cpu.to(at::kFloat));
    expected = expected + offset_cpu.to(at::kFloat);
    expected = at::clamp(expected, -128.0, 127.0).to(at::kChar);
    return expected.contiguous();
}

bool compare_and_report(const at::Tensor& actual_cpu, const at::Tensor& expected_cpu)
{
    const at::Tensor actual = actual_cpu.contiguous();
    const int8_t* actual_ptr = actual.data_ptr<int8_t>();
    const int8_t* expected_ptr = expected_cpu.data_ptr<int8_t>();

    for (int64_t idx = 0; idx < kNumel; ++idx) {
        if (actual_ptr[idx] != expected_ptr[idx]) {
            const int64_t batch = idx / (kRows * kCols);
            const int64_t rem = idx % (kRows * kCols);
            const int64_t row = rem / kCols;
            const int64_t col = rem % kCols;
            std::cout << "FAIL: _npu_quantize_per_tensor" << std::endl;
            std::cout << "mismatch at [" << batch << "," << row << "," << col
                      << "]: actual=" << static_cast<int>(actual_ptr[idx])
                      << ", expected=" << static_cast<int>(expected_ptr[idx]) << std::endl;
            return false;
        }
    }

    std::cout << "PASS: _npu_quantize_per_tensor" << std::endl;
    return true;
}

} // namespace

int main()
{
    try {
        const c10::Device npu_device(c10::DeviceType::PrivateUse1, 0);

        at::Tensor input_x_cpu = make_input_x_cpu();
        at::Tensor scale_cpu = make_scale_cpu();
        at::Tensor offset_cpu = make_offset_cpu();
        at::Tensor expected_cpu = calculate_expected(input_x_cpu, scale_cpu, offset_cpu);

        at::Tensor input_x_npu = input_x_cpu.to(npu_device);
        at::Tensor scale_npu = scale_cpu.to(npu_device);
        at::Tensor offset_npu = offset_cpu.to(npu_device);
        at::Tensor y_npu = at::zeros({kBatch, kRows, kCols},
            at::TensorOptions().dtype(at::kChar).device(npu_device));

        at_npu::native::atb::_npu_quantize_per_tensor(input_x_npu, scale_npu, offset_npu, y_npu);

        at::Tensor actual_cpu = y_npu.to(at::kCPU);
        return compare_and_report(actual_cpu, expected_cpu) ? 0 : 1;
    } catch (const c10::Error& e) {
        std::cout << "FAIL: _npu_quantize_per_tensor" << std::endl;
        std::cout << "c10::Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "FAIL: _npu_quantize_per_tensor" << std::endl;
        std::cout << "std::exception: " << e.what() << std::endl;
        return 1;
    }
}
