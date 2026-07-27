from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


@GENERATOR_REGISTRY.register("gdd_geluandmul_fatreluandmul")
class aclnnGeluAndMulGenerator(CaseGenerator):
    """GELU and Mul 基础用例生成器"""
    
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # 1. 先检查inputs是否为空，避免访问case_config.inputs[0]时报错
        if not case_config.inputs:
            return case_config

        # 2. 获取第一个输入的shape，并检查是否为空列表
        input_shape = case_config.inputs[0].shape
        if not input_shape:
            return case_config

        # 3. 检查最后一维是否为奇数，仅在维度数≥1时操作
        if input_shape[-1] % 2 != 0:
            # 修改最后一维为偶数（+1）
            case_config.inputs[0].shape[-1] += 1

        return case_config
