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
    future_steps: int = 1
    history_latent_dim: int = 0
    history_steps: int = 0


@dataclass(frozen=True)
class MoWAFutureLatentPriorOutput:
    predicted_future_latent: Any
    done_logits: Any
    history_latent_used: bool
    future_latent_target_required: bool


class MoWAFutureLatentPrior(nn.Module):
    """Predict future latent from current latent + text only.

    明确不接收 robot history latent；future latent 只作为 target。
    """

    def __init__(self, config: MoWAFutureLatentPriorConfig | None = None):
        super().__init__()
        self.config = config or MoWAFutureLatentPriorConfig()
        self.history_enabled = self.config.history_latent_dim > 0 and self.config.history_steps > 0
        input_dim = self.config.current_latent_dim + self.config.text_hidden_dim
        self.history_projector = None
        if self.history_enabled:
            self.history_projector = nn.Linear(self.config.history_latent_dim, self.config.hidden_dim)
            input_dim += self.config.hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(
                input_dim,
                self.config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.future_steps * self.config.future_latent_dim),
        )
        self.done_head = nn.Linear(self.config.hidden_dim, self.config.future_steps)

    def forward(
        self,
        current_latent: Any,
        text_hidden: Any,
        *,
        history_latent: Any | None = None,
        history_valid_mask: Any | None = None,
    ) -> MoWAFutureLatentPriorOutput:
        if history_latent is not None and not self.history_enabled:
            raise ValueError("MoWA future latent prior is not configured for history_latent input.")
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
        features = [current_latent, text_hidden]
        if self.history_enabled:
            if history_latent is None or history_latent.dim() != 3:
                raise ValueError("history_latent must have shape [B, H, D] when history conditioning is enabled.")
            if history_latent.shape[1:] != (self.config.history_steps, self.config.history_latent_dim):
                raise ValueError("history_latent shape does not match configured history steps/dimension.")
            if history_valid_mask is None:
                history_valid_mask = torch.ones(history_latent.shape[:2], device=history_latent.device, dtype=torch.bool)
            mask = history_valid_mask.unsqueeze(-1).to(history_latent.dtype)
            history_summary = (history_latent * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            features.append(self.history_projector(history_summary))
        fused = torch.cat(features, dim=-1)
        hidden = self.trunk[:-1](fused)
        predicted_future_latent = self.trunk[-1](hidden).view(-1, self.config.future_steps, self.config.future_latent_dim)
        if self.config.future_steps == 1:
            predicted_future_latent = predicted_future_latent.squeeze(1)
        return MoWAFutureLatentPriorOutput(
            predicted_future_latent=predicted_future_latent,
            done_logits=self.done_head(hidden),
            history_latent_used=False,
            future_latent_target_required=True,
        )

    def compute_loss(
        self,
        current_latent: Any,
        text_hidden: Any,
        future_latent_target: Any,
        future_valid_mask: Any | None = None,
        done_target: Any | None = None,
        *,
        history_latent: Any | None = None,
        history_valid_mask: Any | None = None,
    ) -> tuple[Any, dict[str, Any], MoWAFutureLatentPriorOutput]:
        output = self.forward(
            current_latent,
            text_hidden,
            history_latent=history_latent,
            history_valid_mask=history_valid_mask,
        )
        target = future_latent_target
        if target.dim() == 2 and self.config.future_steps == 1:
            target = target.unsqueeze(1)
        if target.dim() != 3:
            raise ValueError("future_latent_target must have shape [B, F, D].")
        prediction = output.predicted_future_latent.unsqueeze(1) if output.predicted_future_latent.dim() == 2 else output.predicted_future_latent
        if target.shape != prediction.shape:
            raise ValueError(
                "future_latent_target shape mismatch: "
                f"expected {tuple(output.predicted_future_latent.shape)}, got {tuple(target.shape)}."
            )
        per_step_loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        if future_valid_mask is None:
            future_valid_mask = torch.ones_like(per_step_loss, dtype=torch.bool)
        valid_count = future_valid_mask.sum()
        latent_loss = (per_step_loss * future_valid_mask.to(per_step_loss.dtype)).sum() / valid_count.clamp(min=1)
        if done_target is None:
            done_target = (~future_valid_mask).to(dtype=output.done_logits.dtype)
        done_loss = F.binary_cross_entropy_with_logits(output.done_logits, done_target.to(output.done_logits.dtype))
        loss = latent_loss + done_loss
        return (
            loss,
            {
                "future_latent_mse": latent_loss,
                "done_loss": done_loss,
                "future_valid_count": valid_count,
            },
            output,
        )
