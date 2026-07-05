"""MoWA Future-GatedHeads — optional per-head gate wrapper (M2-003).

This module wraps :class:`MoWAFutureFeatureHeads` and adds a learnable
per-head scalar gate in [0, 1] after a sigmoid.  The gate value is logged
per step so that the contribution of each head can be audited without
running leave-one-out sweeps.

Design contract (from ``02_detailed_design.md`` §5.5, §8.4):
  * Only one comparison against FullHeads is planned (E-002).
  * No per-head / leave-one-out / selected-head sweep is implemented here.
  * If any gate is outside [0, 1] the module raises ``ValueError`` at
    initialisation time (sanity check).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.mowa.p0_heads import MoWAFutureFeatureHeads, MoWAFutureFullHeadsConfig, MoWAFutureFeatures
from starVLA.mowa_constants import MOWA_FUTURE_FULL_HEADS


class MoWAGatedHeadsConfig:
    """Thin dataclass-alike config for :class:`MoWAGatedHeads`."""

    def __init__(self, heads_config: MoWAFutureFullHeadsConfig, init_gate: float = 0.5):
        self.heads_config = heads_config
        self.init_gate = float(init_gate)


class MoWAGatedHeads(nn.Module):
    """Wraps FullHeads with learnable per-head scalar gates."""

    def __init__(self, config: MoWAGatedHeadsConfig):
        super().__init__()
        self._full_heads = MoWAFutureFeatureHeads(config.heads_config)
        self._head_names = MOWA_FUTURE_FULL_HEADS
        raw_gates = torch.full((len(self._head_names),), config.init_gate)
        self._raw_gates = nn.Parameter(raw_gates)
        # sanity: sigmoid(init_gate) must be in [0, 1]
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
