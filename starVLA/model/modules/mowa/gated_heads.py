"""MoWA Future-GatedHeads — optional per-head gate wrapper (M2-003).

This module wraps :class:`MoWAFutureFeatureHeads` and adds a learnable
per-head scalar gate in [0, 1] after a sigmoid.  The configured initial
gate value uses the same [0, 1] semantics as the observed gate value so
that reports can be interpreted directly.

Design contract (from ``02_detailed_design.md`` §5.5, §8.4):
  * Only one comparison against FullHeads is planned (E-002).
  * No per-head / leave-one-out / selected-head sweep is implemented here.
  * Configured gate values must stay in [0, 1].
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from starVLA.model.modules.mowa.p0_heads import (
    MoWAFutureFeatureHeads,
    MoWAFutureFullHeadsConfig,
    MoWAFutureFeatures,
    _compute_mowa_head_loss,
)
from starVLA.mowa_constants import MOWA_FUTURE_FULL_HEADS


class MoWAGatedHeadsConfig:
    """Thin dataclass-alike config for :class:`MoWAGatedHeads`."""

    def __init__(
        self,
        heads_config: MoWAFutureFullHeadsConfig,
        init_gate_value: float = 0.5,
        *,
        comparison_scope: str = "single_fullheads_control_only",
        allow_per_head_sweep: bool = False,
    ):
        self.heads_config = heads_config
        self.init_gate_value = float(init_gate_value)
        self.comparison_scope = str(comparison_scope)
        self.allow_per_head_sweep = bool(allow_per_head_sweep)
        if not 0.0 <= self.init_gate_value <= 1.0:
            raise ValueError("GatedHeads init_gate_value must be in [0, 1].")
        if self.comparison_scope != "single_fullheads_control_only":
            raise ValueError(
                "GatedHeads comparison_scope must stay 'single_fullheads_control_only'."
            )
        if self.allow_per_head_sweep:
            raise ValueError("GatedHeads must not enable per-head sweep.")


class MoWAGatedHeads(nn.Module):
    """Wraps FullHeads with learnable per-head scalar gates."""

    def __init__(self, config: MoWAGatedHeadsConfig):
        super().__init__()
        self.config = config
        self._full_heads = MoWAFutureFeatureHeads(config.heads_config)
        self._head_names = MOWA_FUTURE_FULL_HEADS
        init_gate_tensor = torch.full(
            (len(self._head_names),),
            config.init_gate_value,
            dtype=torch.float32,
        )
        init_gate_tensor = init_gate_tensor.clamp(min=1e-6, max=1.0 - 1e-6)
        raw_gates = torch.logit(init_gate_tensor)
        self._raw_gates = nn.Parameter(raw_gates)
        gated = torch.sigmoid(self._raw_gates)
        if not ((gated >= 0.0) & (gated <= 1.0)).all():
            raise ValueError("GatedHeads gate values must be in [0, 1].")

    @property
    def head_names(self) -> tuple[str, ...]:
        return self._head_names

    def gate_values(self) -> dict[str, float]:
        """Return the current per-head gate values (logged per step)."""
        gated = torch.sigmoid(self._raw_gates)
        return {name: float(gated[i].item()) for i, name in enumerate(self._head_names)}

    def gate_summary(self, *, step: int | None = None) -> dict[str, object]:
        """Return a structured summary for logging/auditing."""

        return {
            "comparison_scope": self.config.comparison_scope,
            "allow_per_head_sweep": self.config.allow_per_head_sweep,
            "step": step,
            "gate_values": self.gate_values(),
        }

    def future_features(
        self,
        hidden_features: torch.Tensor,
        masks: dict[str, bool],
    ) -> MoWAFutureFeatures:
        return self(hidden_features, masks)

    def forward(self, hidden_features: torch.Tensor, masks: dict[str, bool]) -> MoWAFutureFeatures:
        raw = self._full_heads.future_features(hidden_features, masks)
        gated = torch.sigmoid(self._raw_gates)
        gated_outputs = {}
        for i, name in enumerate(self._head_names):
            if name in raw.head_outputs:
                gated_outputs[name] = raw.head_outputs[name] * gated[i]
        return MoWAFutureFeatures(
            hidden_features=raw.hidden_features,
            head_outputs=gated_outputs,
            active_heads=raw.active_heads,
            masked_heads=raw.masked_heads,
        )

    def compute_loss(
        self,
        hidden_features: torch.Tensor,
        targets: Mapping[str, Any],
        masks: Mapping[str, Any],
    ):
        import torch

        outputs = self(hidden_features, masks)
        losses = {}
        active_losses = []
        for head in MOWA_FUTURE_FULL_HEADS:
            if not bool(masks.get(head, False)):
                continue
            if head not in targets:
                raise KeyError(f"MoWA GatedHeads active head missing target: {head}")
            loss = _compute_mowa_head_loss(
                head,
                outputs.head_outputs[head],
                targets[head],
                action_outcome_loss_type=self.config.heads_config.action_outcome_loss_type,
            )
            losses[head] = loss
            active_losses.append(loss)
        if not active_losses:
            raise ValueError("MoWA GatedHeads loss requires at least one active mask.")
        total = torch.stack(active_losses).sum()
        losses["total"] = total
        return total, losses, outputs.head_outputs


# Preferred semantic aliases and backward-compatible P0 aliases.
MoWAFutureGatedHeadsConfig = MoWAGatedHeadsConfig
MoWAFutureGatedHeads = MoWAGatedHeads
MoWAP0GatedHeadsConfig = MoWAGatedHeadsConfig
MoWAP0GatedHeads = MoWAGatedHeads
