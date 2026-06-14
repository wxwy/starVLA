# Copyright 2026 StarFlow-VLA contributors. All rights reserved.
"""StarFlow-VLA framework facade.

本文件只提供 StarFlowVLA 的 StarVLA registry 入口。P0 阶段不复制
QwenPI_v3 的主体构建、forward 或 predict_action 逻辑，而是继承
Qwen_PI_v3，并通过 mapping 元数据记录 StarVLA-native 实现边界。
"""

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("StarFlowVLA")
class StarFlowVLA(Qwen_PI_v3):
    """StarFlow-VLA facade that reuses QwenPI_v3 and LayerwiseFM by default."""

    framework_name = "StarFlowVLA"
    implementation_mode = "starvla_native"
    base_framework = "QwenPI_v3"

    def describe_starflow_mapping(self) -> dict:
        """Return a serializable mapping from design abstractions to StarVLA-native modules."""
        action_cfg = getattr(getattr(self.config, "framework", None), "action_model", None)
        return {
            "framework_name": self.framework_name,
            "implementation_mode": self.implementation_mode,
            "base_framework": self.base_framework,
            "action_head": "LayerwiseFM",
            "state_mode": "discretized_instruction",
            "adapter_mode": "future_token_cross_dit",
            "flow_condition_runtime": False,
            "perceiver_enabled": False,
            "num_target_vision_tokens": getattr(action_cfg, "num_target_vision_tokens", 32),
            "solver": "euler",
            "num_inference_timesteps": getattr(action_cfg, "num_inference_timesteps", None),
        }
