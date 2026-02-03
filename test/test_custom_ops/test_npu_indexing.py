import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestIndexing(TestCase):
    def setUp(self):
        super().setUp()
        tensor = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
        self._cpu_input = tensor.numpy()
        self._npu_input = tensor.npu()

    def npu_op_exec(
        self,
        input_tensor,
        begin,
        end,
        strides,
        begin_mask=0,
        end_mask=0,
        ellipsis_mask=0,
        new_axis_mask=0,
        shrink_axis_mask=0,
    ):
        output = torch_npu.npu_indexing(
            input_tensor,
            begin,
            end,
            strides,
            begin_mask,
            end_mask,
            ellipsis_mask,
            new_axis_mask,
            shrink_axis_mask,
        )
        return output.cpu().numpy()

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_1_basic_slicing_all_dimensions(self):
        """基础切片测试，覆盖所有维度"""
        # 测试每个维度的基础切片
        cpu_output = self._cpu_input[0:2, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 使用负索引
        cpu_output = self._cpu_input[0:-1, 1:-1, -3:-1, -2:]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, -3, -2], [-1, -1, -1, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_2_strides_all_directions(self):
        """测试各种步长组合"""
        # 正向步长
        cpu_output = self._cpu_input[::2, ::1, ::2, ::3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [2, 1, 2, 3])
        self.assertRtolEqual(cpu_output, npu_output)

        # 负向步长
        cpu_output = self._cpu_input[1::-1, 2::-2, 3::-1, 4::-3]
        npu_output = self.npu_op_exec(self._npu_input, [1, 2, 3, 4], [0, 0, 0, 0], [-1, -2, -1, -3])
        self.assertRtolEqual(cpu_output, npu_output)

        # 混合方向步长
        cpu_output = self._cpu_input[::1, ::-1, ::2, ::-2]
        npu_output = self.npu_op_exec(self._npu_input, [0, 3, 0, 5], [2, 0, 4, 0], [1, -1, 2, -2])
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_3_begin_mask_all_combinations(self):
        """测试所有begin_mask组合"""
        # mask第一维
        cpu_output = self._cpu_input[:, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask第二维
        cpu_output = self._cpu_input[0:2, :, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 2, 3], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b0010)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask所有维度
        cpu_output = self._cpu_input[:, :, :, :]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], begin_mask=0b1111)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask部分维度
        cpu_output = self._cpu_input[:, 1:, :3, 2:]
        npu_output = self.npu_op_exec(
            self._npu_input,
            [0, 1, 0, 2],
            [2, 3, 3, 5],
            [1, 1, 1, 1],
            begin_mask=0b1001,
            end_mask=0b0110,
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_4_end_mask_all_combinations(self):
        """测试所有end_mask组合"""
        # mask第一维
        cpu_output = self._cpu_input[0:, 1:3, 2:4, 3:5]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 3, 4, 5], [1, 1, 1, 1], end_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # mask多个维度
        cpu_output = self._cpu_input[0:, 1:, 2:, 3:]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 0, 0, 0], [1, 1, 1, 1], end_mask=0b1111)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_5_shrink_axis_mask_all_combinations(self):
        """测试所有shrink_axis_mask组合"""
        # 缩减单个维度
        cpu_output = self._cpu_input[0, :, :, :]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 0, 0, 0], [1, 3, 4, 5], [1, 1, 1, 1], shrink_axis_mask=0b0001
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 缩减两个维度
        cpu_output = self._cpu_input[0, 1, :, :]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 0, 0], [1, 2, 4, 5], [1, 1, 1, 1], shrink_axis_mask=0b0011
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 缩减所有维度
        cpu_output = self._cpu_input[0, 1, 2, 3]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 1, 1], shrink_axis_mask=0b1111
        )
        self.assertEqual(cpu_output.item(), npu_output.item())

        # 缩减中间维度
        cpu_output = self._cpu_input[:, 1, :, 2]
        npu_output = self.npu_op_exec(
            self._npu_input, [0, 1, 0, 2], [2, 2, 4, 3], [1, 1, 1, 1], shrink_axis_mask=0b1010
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_6_new_axis_mask_all_combinations(self):
        """测试所有new_axis_mask组合"""
        # 在开头新增维度
        cpu_output = self._cpu_input[None, :, :, :, :]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], new_axis_mask=0b0001)
        self.assertRtolEqual(cpu_output, npu_output)

        # 在中间新增维度
        cpu_output = self._cpu_input[:, None, :, None, :]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], new_axis_mask=0b1010)
        self.assertRtolEqual(cpu_output, npu_output)

        # 在所有位置新增维度
        cpu_output = self._cpu_input[None, None, None, None, :, :, :, :]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 0], [2, 3, 4, 5], [1, 1, 1, 1], new_axis_mask=0b1111)
        # 注意：这里new_axis_mask的位数应该等于原始维度数，但会插入多个新维度
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_7_ellipsis_mask_various_positions(self):
        """测试省略号在不同位置"""
        # 省略号在中间
        cpu_output = self._cpu_input[0, ..., 2]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 0, 2], [1, 3, 4, 3], [1, 1, 1, 1], ellipsis_mask=0b0100)
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号在开头
        cpu_output = self._cpu_input[..., 1, 2]
        npu_output = self.npu_op_exec(self._npu_input, [0, 0, 1, 2], [2, 3, 2, 3], [1, 1, 1, 1], ellipsis_mask=0b1000)
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号在结尾
        cpu_output = self._cpu_input[0, 1, ...]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 0, 0], [1, 2, 4, 5], [1, 1, 1, 1], ellipsis_mask=0b0010)
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_8_complex_combinations_all_masks(self):
        """测试复杂的mask组合"""
        # 组合begin_mask, end_mask, strides
        cpu_output = self._cpu_input[:, 1:3:2, ::-1, 2:]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 1, 4, 2],  # 注意第三维因为[::-1]所以从4开始
            end=[2, 3, -1, 5],
            strides=[1, 2, -1, 1],
            begin_mask=0b0001,  # mask第一维
            end_mask=0b0100,  # mask第三维
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 组合shrink_axis_mask和new_axis_mask
        cpu_output = self._cpu_input[0, None, 1, :, None, 2]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 1, 0, 2],
            end=[1, 2, 4, 3],
            strides=[1, 1, 1, 1],
            shrink_axis_mask=0b1001,  # 缩减第一维和第四维
            new_axis_mask=0b0110,  # 在第二维和第四维位置新增维度
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_9_edge_cases_with_all_masks(self):
        """边界情况测试"""
        # 空切片
        cpu_output = self._cpu_input[0:0, 1:1, 2:2, 3:3]
        npu_output = self.npu_op_exec(self._npu_input, [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 全部使用负索引
        cpu_output = self._cpu_input[-2:, -3:, -4:, -5:]
        npu_output = self.npu_op_exec(self._npu_input, [-2, -3, -4, -5], [2, 3, 4, 5], [1, 1, 1, 1])
        self.assertRtolEqual(cpu_output, npu_output)

        # 混合mask和负索引
        cpu_output = self._cpu_input[:, -2:, :, -1:]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, -2, 0, -1],
            end=[2, 3, 4, 5],
            begin_mask=0b0101,
            end_mask=0b0101,
        )
        self.assertRtolEqual(cpu_output, npu_output)

    @SupportedDevices(["Ascend950"])
    def test_npu_indexing_10_ellipsis_with_other_masks(self):
        """测试省略号与其他mask的组合"""
        # 省略号与begin/end mask组合
        cpu_output = self._cpu_input[..., 1:, :2]
        # 这相当于 self.tensor[:, :, 1:, :2]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 0, 1, 0],
            end=[2, 3, 3, 2],
            strides=[1, 1, 1, 1],
            ellipsis_mask=0b1100,  # 在前两维使用省略号
            begin_mask=0b0010,  # mask第三维的begin
            end_mask=0b0001,  # mask第四维的end
        )
        self.assertRtolEqual(cpu_output, npu_output)

        # 省略号与负步长
        cpu_output = self._cpu_input[..., ::-1, :]
        npu_output = self.npu_op_exec(
            self._npu_input,
            begin=[0, 0, 4, 0],
            end=[2, 3, 0, 5],
            strides=[1, 1, -1, 1],
            ellipsis_mask=0b1000,
            end_mask=0b0100,
        )
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
