# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig
import random

H_CHOICES = [512, 768, 1024, 1536, 1792, 2048, 2560, 4096]
G_CHOICES = [192, 256, 384]
MAX_S = 4 * 1024


@GENERATOR_REGISTRY.register("generate_swiglu_group_quant_grad")
class GddGenerateAclnnSwigluGroupQuantGradGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.type = None
        self.dtype = None
        self.tuple_numbers = 0
        self.tensors_shape = []

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        B = random.randint(1, 31)
        S = random.randint(1, MAX_S)
        T = B * S

        H = random.choice(H_CHOICES)
        H2 = H * 2

        use_3d = len(case_config.inputs[0].shape) == 3

        if use_3d:
            case_config.inputs[0].shape = [B, S, H]
            case_config.inputs[1].shape = [B, S, H2]
            case_config.inputs[2].shape = [B, S, 1]
            case_config.inputs[3].shape = [B, S, H]
        else:
            case_config.inputs[0].shape = [T, H]
            case_config.inputs[1].shape = [T, H2]
            case_config.inputs[2].shape = [T, 1]
            case_config.inputs[3].shape = [T, H]


        case_config.inputs[1].dtype = case_config.inputs[0].dtype
        case_config.inputs[3].dtype = case_config.inputs[0].dtype

        G = random.choice(G_CHOICES)
        case_config.inputs[4].shape = [G]

        clamp_limit = random.choice([0.0, 0.5, 1.0, 5.0, 10.0])
        case_config.inputs[5].range_values = clamp_limit



        print(f"参数类型打印======================================================================")
        print(f"case_config[0] type:{case_config.inputs[0]}")
        print(f"case_config[1] type:{case_config.inputs[1]}")
        print(f"case_config[2] type:{case_config.inputs[2]}")
        print(f"case_config[3] type:{case_config.inputs[3]}")
        print(f"case_config[4] type:{case_config.inputs[4]}")
        print(f"case_config[5] type:{case_config.inputs[5]}")

        return case_config
