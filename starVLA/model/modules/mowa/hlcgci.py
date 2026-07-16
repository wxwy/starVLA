"""MoWA HLC-GCI interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MoWAHLCGCIConfig:
    history_latent_dim: int = 1024
    condition_hidden_dim: int = 1024
    history_steps: int = 10
    compressed_history_dim: int = 512
    gate_hidden_dim: int = 256


@dataclass(frozen=True)
class MoWAHLCGCIOutput:
    compressed_history: Any
    gate_values: Any
    gated_condition_tokens: Any
    history_tokens: Any


class MoWAHLCGCI(nn.Module):
    """Encode history latent into condition-side history tokens."""

    def __init__(self, config: MoWAHLCGCIConfig | None = None):
        super().__init__()
        self.config = config or MoWAHLCGCIConfig()
        self.history_projector = nn.Linear(
            self.config.history_latent_dim,
            self.config.compressed_history_dim,
        )
        self.temporal_encoder = nn.GRU(
            input_size=self.config.history_latent_dim,
            hidden_size=self.config.compressed_history_dim,
            batch_first=True,
        )
        self.history_token_projector = nn.Linear(
            self.config.compressed_history_dim,
            self.config.condition_hidden_dim,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.config.compressed_history_dim, self.config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.gate_hidden_dim, self.config.condition_hidden_dim),
            nn.Sigmoid(),
        )
        self.condition_projector = nn.Linear(
            self.config.condition_hidden_dim,
            self.config.condition_hidden_dim,
        )

    def encode_history_tokens(self, history_latent: Any) -> Any:
        temporal_history, _ = self.temporal_encoder(history_latent)
        return self.history_token_projector(temporal_history)

    def forward(
        self,
        history_latent: Any,
        condition_tokens: Any,
    ) -> MoWAHLCGCIOutput:
        if history_latent.dim() != 3:
            raise ValueError("history_latent must have shape [B, T, D].")
        if condition_tokens.dim() != 3:
            raise ValueError("condition_tokens must have shape [B, N, D].")
        if history_latent.shape[1] != self.config.history_steps:
            raise ValueError(
                "history_latent time steps mismatch: "
                f"expected {self.config.history_steps}, got {history_latent.shape[1]}."
            )
        if history_latent.shape[-1] != self.config.history_latent_dim:
            raise ValueError(
                "history_latent dim mismatch: "
                f"expected {self.config.history_latent_dim}, got {history_latent.shape[-1]}."
            )
        if condition_tokens.shape[-1] != self.config.condition_hidden_dim:
            raise ValueError(
                "condition_tokens dim mismatch: "
                f"expected {self.config.condition_hidden_dim}, got {condition_tokens.shape[-1]}."
            )
        compressed_per_step = self.history_projector(history_latent)
        compressed_history = compressed_per_step.mean(dim=1)
        history_tokens = self.encode_history_tokens(history_latent)
        gate_values = self.gate_mlp(compressed_history)
        gated_condition_tokens = self.condition_projector(condition_tokens)
        gated_condition_tokens = gated_condition_tokens * gate_values.unsqueeze(1)
        return MoWAHLCGCIOutput(
            compressed_history=compressed_history,
            gate_values=gate_values,
            gated_condition_tokens=gated_condition_tokens,
            history_tokens=history_tokens,
        )
