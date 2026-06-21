# Copyright 2026 StarFlow-VLA contributors. All rights reserved.
"""StarFlow-VLA framework facade。

本文件只提供 StarFlowVLA 的 StarVLA registry 入口。P0 阶段不复制
QwenPI_v3 的主体构建、forward 或 predict_action 逻辑，而是继承
Qwen_PI_v3，并通过 mapping 元数据记录 StarVLA-native 实现边界。
"""

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.starflow_vla.mapping import build_starflow_mapping
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _get_config_value(config, path, default=None):
    value = config
    for key in path:
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
    return value


@FRAMEWORK_REGISTRY.register("StarFlowVLA")
class StarFlowVLA(Qwen_PI_v3):
    """默认复用 QwenPI_v3 与 LayerwiseFM 的 StarFlow-VLA facade。"""

    framework_name = "StarFlowVLA"
    implementation_mode = "starvla_native"
    base_framework = "QwenPI_v3"

    def describe_starflow_mapping(self) -> dict:
        """返回设计抽象到 StarVLA-native 实现的可序列化映射。"""
        return build_starflow_mapping(self.config)

    def _prepare_state_condition(self, instructions, state):
        """按 `state_mode` 选择状态进入 instruction 或 action head。"""
        state_mode = _get_config_value(
            self.config,
            ("framework", "state_mode"),
            "discretized_instruction",
        )
        if state is None:
            return instructions, None
        if state_mode == "discretized_instruction":
            return super()._prepare_state_condition(instructions, state)
        if state_mode == "continuous_head":
            return instructions, state
        if state_mode == "none":
            return instructions, None
        raise ValueError(f"Unsupported StarFlowVLA state_mode: {state_mode}")
