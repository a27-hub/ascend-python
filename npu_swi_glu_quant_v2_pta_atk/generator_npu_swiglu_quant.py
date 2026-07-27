import random
from typing import Union, List
import itertools
import numpy as np

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig , CaseConfig


def adjust_input_shape(input_shape, max_product=128 * 1024):
    """
    调整输入形状，以确保除最后一维之外的维度乘积不超过最大值。

    参数:
    input_shape (tuple): 输入张量的形状。
    max_product (int): 除最后一维之外的维度乘积的最大允许值，默认为128K。

    返回:
    tuple: 调整后的输入张量形状。
    """

    # 分离最后一维和其他维度
    other_dims = input_shape[:-1]
    first_dim = input_shape[0]
    last_dim = input_shape[-1]
    # 偶数处理
    if last_dim % 2 != 0:
        input_shape[-1] = input_shape[-1] + 1

    if first_dim % 2 != 0:
        input_shape[0] = input_shape[0] + 1

    # 计算其他维度的乘积
    product_of_other_dims = np.prod(other_dims)

    # 如果乘积已经满足条件，直接返回原形状
    if product_of_other_dims <= max_product:
        return input_shape

    # 如果不满足条件，调整其他维度
    adjusted_shape = list(input_shape)

    # 找到可以缩小的维度，并尽可能保持原始比例
    scaling_factor = np.power(max_product / product_of_other_dims, 1.0 / len(other_dims))

    for i in range(len(other_dims)):
        # 将每个维度按比例缩小
        adjusted_shape[i] = max(1, int(adjusted_shape[i] * scaling_factor))

    # 确保调整后的乘积确实不超过最大值
    adjusted_product = np.prod(adjusted_shape[:-1])
    if adjusted_product > max_product:
        # 如果仍然超过最大值，强制设置为最接近最大值的尺寸
        adjusted_shape[:-1] = [max_product] + [1] * (len(other_dims) - 1)

    adjusted_shape[-1] = (adjusted_shape[-1] + 1) // 2 * 2
    adjusted_shape[0] = (adjusted_shape[0] + 1) // 2 * 2

    return adjusted_shape

@GENERATOR_REGISTRY.register("ascend_generator_npu_swiglu_quant")
class SwigluquantGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.x_shape = []
        self.dtype = int

    def after_input_config(
            self,
            index: int,
            input_case: Union[InputCaseConfig, List[InputCaseConfig]]
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        x = case_config.inputs[0]
        smooth_scales = case_config.inputs[1]
        offsets_t = case_config.inputs[2]
        group_index_t = case_config.inputs[3]
        activate_left = case_config.inputs[4]
        quantModeOptional = case_config.inputs[5]
        staticMode = case_config.inputs[6]
        dstType = case_config.inputs[8]
        x.shape[-1] = min(8192 , x.shape[-1])
        x.shape = adjust_input_shape(x.shape)
        if dstType.range_values == 29 and x.shape[-1] % 4 != 0:
            x.shape[-1] = x.shape[-1] * 2

        mul1 = int(np.prod(x.shape[:-1]))

        N = x.shape[-1] // 2
        group_index_t.shape[0] = min(group_index_t.shape[0] , int(mul1 / 2))
        G = group_index_t.shape[0]

        group_index_t.range_values = [0 , mul1]

        
        smooth_scales.shape[0] = G
        if len(smooth_scales.shape) == 2:
            smooth_scales.shape[-1] = N
        if quantModeOptional.range_values == "static" :
            if staticMode.range_values == "per_channel":
                offsets_t.shape = [G,N]
            elif staticMode.range_values == "per_tensor":
                offsets_t.shape = offsets_t.shape[:1]
                offsets_t.shape[0] = G
                smooth_scales.shape = offsets_t.shape

        return case_config