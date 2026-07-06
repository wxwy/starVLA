"""MoWA WAM-to-action bridge interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .full_heads import MoWAFutureFeatures


@dataclass(frozen=True)
class MoWAActionBridgeConfig:
    wam_feature_dim: int = 32
    action_hidden_dim: int = 2048
    num_action_layers: int = 36
    num_bridge_tokens: int = 4


@dataclass(frozen=True)
class MoWAActionBridgeOutput:
    layerwise_condition_features: tuple[Any, ...]
    attention_mask: Any
    active_heads: tuple[str, ...]
    masked_heads: tuple[str, ...]


class MoWAActionBridge:
    """将 MoWA future features 映射为 LayerwiseFM 可消费的条件 token。

    该接口只定义 bridge 边界，不改写 LayerwiseFM action head 内部逻辑。
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        import torch.nn as nn

        class _TorchMoWAActionBridge(nn.Module):
            def __init__(self, config: MoWAActionBridgeConfig | None = None):
                super().__init__()
                self.config = config or MoWAActionBridgeConfig()
                self.token_projector = nn.Linear(
                    self.config.wam_feature_dim,
                    self.config.num_bridge_tokens * self.config.action_hidden_dim,
                )
                self.layer_embedding = nn.Parameter(
                    torch.zeros(
                        self.config.num_action_layers,
                        self.config.num_bridge_tokens,
                        self.config.action_hidden_dim,
                    )
                )
                nn.init.normal_(self.layer_embedding, mean=0.0, std=0.02)

            def forward(self, future_features: MoWAFutureFeatures) -> MoWAActionBridgeOutput:
                hidden = future_features.hidden_features
                if hidden.dim() != 2:
                    raise ValueError("MoWAActionBridge expects hidden_features with shape [B, D].")
                if hidden.shape[-1] != self.config.wam_feature_dim:
                    raise ValueError(
                        "MoWAActionBridge wam_feature_dim mismatch: "
                        f"expected {self.config.wam_feature_dim}, got {hidden.shape[-1]}."
                    )

                batch_size = hidden.shape[0]
                tokens = self.token_projector(hidden).reshape(
                    batch_size,
                    self.config.num_bridge_tokens,
                    self.config.action_hidden_dim,
                )
                layerwise = tuple(
                    tokens + self.layer_embedding[layer_idx].unsqueeze(0)
                    for layer_idx in range(self.config.num_action_layers)
                )
                attention_mask = torch.ones(
                    batch_size,
                    self.config.num_bridge_tokens,
                    dtype=torch.bool,
                    device=hidden.device,
                )
                return MoWAActionBridgeOutput(
                    layerwise_condition_features=layerwise,
                    attention_mask=attention_mask,
                    active_heads=future_features.active_heads,
                    masked_heads=future_features.masked_heads,
                )

        return _TorchMoWAActionBridge(*args, **kwargs)
