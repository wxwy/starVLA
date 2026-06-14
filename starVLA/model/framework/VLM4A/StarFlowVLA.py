# Copyright 2026 StarFlow-VLA contributors. All rights reserved.
"""StarFlow-VLA framework facade。

本文件只提供 StarFlowVLA 的 StarVLA registry 入口。P0 阶段不复制
QwenPI_v3 的主体构建、forward 或 predict_action 逻辑，而是继承
Qwen_PI_v3，并通过 mapping 元数据记录 StarVLA-native 实现边界。
"""

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.starflow_vla.mapping import build_starflow_mapping
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("StarFlowVLA")
class StarFlowVLA(Qwen_PI_v3):
    """默认复用 QwenPI_v3 与 LayerwiseFM 的 StarFlow-VLA facade。"""

    framework_name = "StarFlowVLA"
    implementation_mode = "starvla_native"
    base_framework = "QwenPI_v3"

    def describe_starflow_mapping(self) -> dict:
        """返回设计抽象到 StarVLA-native 实现的可序列化映射。"""
        return build_starflow_mapping(self.config)
