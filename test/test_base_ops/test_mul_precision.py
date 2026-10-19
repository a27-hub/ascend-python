import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


def bits16(out):
    # 位级比较：修复前后仅差 1 ulp，rtol 断言不可分辨
    return out.cpu().contiguous().view(torch.uint16).flatten().tolist()


def fp64_expected(scalar_value, tensor_value, out_dtype):
    # 期望值由 fp64 数学现算（非硬编码）：标量保持 fp32 参与计算时，
    # 乘积 = fp32(S) × fp32(V)，其结果与 fp64 精确乘积经验证无双舍入差异
    # （bf16 对 delta=3.3e-9 vs bf16 半格 9.8e-4；fp16 对距中点余量 2e-5 vs fp32 误差 1e-8）
    return (torch.tensor(scalar_value, dtype=torch.float64)
            * torch.tensor(tensor_value, dtype=torch.float64)).to(out_dtype)


class TestMulPrecision(TestCase):
    def test_mul_cpu0dim_fp32_scalar_bf16(self):
        # issue #5392 实验三：CPU 0-dim fp32 标量 × 卡上 bf16 张量
        # 修复前标量被强转 bf16，输出 0xBF59；修复后标量保 fp32，输出 0xBF5A（与 GPU 对齐）
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_cpu0dim_fp64_scalar_bf16_opmath(self):
        # fp64 标量场景：原生对 fp64 标量同样按 opmath(float) 读入后计算，
        # 修复后以 fp32 上卡（fp64→float 单次转换）；若被改成 double/bf16 上卡本用例失败
        a = torch.tensor(0.22401839494705, dtype=torch.float64)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_operand_order_consistency(self):
        # 同一乘法的三条路径位级一致：标量在 self 侧 / other 侧 / 原地写回
        # 修复前 a*b(0xBF59) != b*a(0xBF5A)，存在用户可见的不对称
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        self.assertEqual(bits16(a * b), bits16(b * a))
        inplace = b.clone()
        inplace.mul_(a)
        self.assertEqual(bits16(a * b), bits16(inplace))

    def test_mul_out_variant(self):
        # out= 形态走 mul_out 入口（self_tensor_to_device 的第二个调用点），同样保精度
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([-3.7969], dtype=torch.bfloat16).npu()
        out = torch.empty(1, dtype=torch.bfloat16, device="npu")
        torch.mul(a, b, out=out)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, -3.7969, torch.bfloat16)))

    def test_mul_cpu0dim_fp32_scalar_fp16(self):
        # 检视意见补充：result_type=fp16 场景看护
        # 判别数对经搜索选定：fp32 标量路径乘积 0.22883129… 与标量预舍 fp16 路径
        # 乘积 0.22881150… 分居 fp16 舍入中点两侧（输出 0x3353 vs 0x3352）
        a = torch.tensor(0.22401839494705)
        b = torch.tensor([1.021484375], dtype=torch.float16).npu()
        out = a * b
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(bits16(out), bits16(fp64_expected(0.22401839494705, 1.021484375, torch.float16)))

    def test_mul_fp32_tensor_regression(self):
        # 爆炸半径守门：fp32 输出场景不命中修复分支，行为必须与修改前完全一致
        a = torch.tensor(0.22401839494705)
        b_cpu = torch.tensor([-3.7969], dtype=torch.float32)
        b_npu = torch.tensor([-3.7969], dtype=torch.float32).npu()
        self.assertEqual((a * b_npu).cpu(), a * b_cpu)


if __name__ == "__main__":
    run_tests()
