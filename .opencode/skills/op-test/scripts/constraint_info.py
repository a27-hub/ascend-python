"""约束信息数据结构 — 脚本间接口契约。
scaffold_init 产出 → scaffold_coverage 消费，经 `.constraint_info.json` 序列化。
步骤4 (LLM) 不碰此模块——它读 CHECKLIST「约束」列人读摘要。
"""

from dataclasses import dataclass, field


@dataclass
class ConstraintInfo:
    """单个参数的约束信息。"""
    param: str
    optional: bool = False
    supports_empty: bool = False
    dtype_link: str | None = None          # "dtype要和x一致" → "x"
    shape_link: dict | None = None         # {"ref":"x", "rule":"last_dim"|"same"|"same_except_last"}
    rank: int | None = None                # 秩约束, 来自 md 维度列单值 "1" / "2"
    dim_range: tuple | None = None         # 维度范围, 来自 md 维度列 "2-8"
    valid_dtypes: list = field(default_factory=list)  # 910b 全集, 来自阶段1 multi-config
    mutex: list = field(default_factory=list)         # 互斥对 [(param, condition)], md 多抽不到
