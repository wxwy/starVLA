"""MoWA future latent prior interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoWAFutureLatentPriorConfig:
    current_latent_dim: int = 1024
    text_hidden_dim: int = 2048
    hidden_dim: int = 2048
    future_latent_dim: int = 1024


@dataclass(frozen=True)
class MoWAFutureLatentPriorOutput:
    predicted_future_latent: Any
    history_latent_used: bool
    future_latent_target_required: bool


class MoWAFutureLatentPrior(nn.Module):
    """Predict future latent from current latent + text only.

    明确不接收 robot history latent；future latent 只作为 target。
    """

    def __init__(self, config: MoWAFutureLatentPriorConfig | None = None):
        super().__init__()
        self.config = config or MoWAFutureLatentPriorConfig()
        self.trunk = nn.Sequential(
            nn.Linear(
                self.config.current_latent_dim + self.config.text_hidden_dim,
                self.config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.future_latent_dim),
        )

    def forward(
        self,
        current_latent: Any,
        text_hidden: Any,
        *,
        history_latent: Any | None = None,
    ) -> MoWAFutureLatentPriorOutput:
        if history_latent is not None:
            raise ValueError("MoWA future latent prior does not accept history_latent input.")
        if current_latent.dim() != 2:
            raise ValueError("current_latent must have shape [B, D].")
        if text_hidden.dim() != 2:
            raise ValueError("text_hidden must have shape [B, D].")
        if current_latent.shape[0] != text_hidden.shape[0]:
            raise ValueError("current_latent and text_hidden batch size must match.")
        if current_latent.shape[-1] != self.config.current_latent_dim:
            raise ValueError(
                "current_latent dim mismatch: "
                f"expected {self.config.current_latent_dim}, got {current_latent.shape[-1]}."
            )
        if text_hidden.shape[-1] != self.config.text_hidden_dim:
            raise ValueError(
                "text_hidden dim mismatch: "
                f"expected {self.config.text_hidden_dim}, got {text_hidden.shape[-1]}."
            )
        fused = torch.cat((current_latent, text_hidden), dim=-1)
        predicted_future_latent = self.trunk(fused)
        return MoWAFutureLatentPriorOutput(
            predicted_future_latent=predicted_future_latent,
            history_latent_used=False,
            future_latent_target_required=True,
        )

    def compute_loss(
        self,
        current_latent: Any,
        text_hidden: Any,
        future_latent_target: Any,
        *,
        history_latent: Any | None = None,
    ) -> tuple[Any, dict[str, Any], MoWAFutureLatentPriorOutput]:
        output = self.forward(
            current_latent,
            text_hidden,
            history_latent=history_latent,
        )
        target = future_latent_target
        if target.dim() != 2:
            raise ValueError("future_latent_target must have shape [B, D].")
        if target.shape != output.predicted_future_latent.shape:
            raise ValueError(
                "future_latent_target shape mismatch: "
                f"expected {tuple(output.predicted_future_latent.shape)}, got {tuple(target.shape)}."
            )
        loss = F.mse_loss(output.predicted_future_latent, target)
        return (
            loss,
            {
                "future_latent_mse": loss,
            },
            output,
        )
