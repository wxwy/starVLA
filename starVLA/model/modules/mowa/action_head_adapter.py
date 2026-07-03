"""Action-head adapter boundary for MoWA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoWAActionHeadBinding:
    action_head_type: str
    condition_kind: str
    injection_mode: str
    implemented: bool
    notes: tuple[str, ...]


MOWA_ACTION_HEAD_BINDINGS: dict[str, MoWAActionHeadBinding] = {
    "LayerwiseFM": MoWAActionHeadBinding(
        action_head_type="LayerwiseFM",
        condition_kind="layerwise_condition_features",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=True,
        notes=(
            "LayerwiseFM consumes per-layer condition sequences.",
            "MoWA bridge tokens can be appended to each vl_embs_list layer with an expanded attention mask.",
        ),
    ),
    "MLP": MoWAActionHeadBinding(
        action_head_type="MLP",
        condition_kind="action_hidden_features",
        injection_mode="hidden_feature_fusion",
        implemented=False,
        notes=(
            "MLP heads do not consume condition tokens.",
            "MoWA must fuse bridge features into action hidden states before predict_action.",
        ),
    ),
    "DiT-B": MoWAActionHeadBinding(
        action_head_type="DiT-B",
        condition_kind="single_condition_sequence",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=False,
        notes=(
            "GR00T-style DiT heads consume a single condition sequence rather than layerwise features.",
            "MoWA needs a separate single-sequence adapter before enabling this head.",
        ),
    ),
    "DiT-L": MoWAActionHeadBinding(
        action_head_type="DiT-L",
        condition_kind="single_condition_sequence",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=False,
        notes=(
            "GR00T-style DiT heads consume a single condition sequence rather than layerwise features.",
            "MoWA needs a separate single-sequence adapter before enabling this head.",
        ),
    ),
}


def resolve_mowa_action_head_binding(action_head_type: str) -> MoWAActionHeadBinding:
    """Resolve how MoWA should bind to a concrete action head."""

    if action_head_type not in MOWA_ACTION_HEAD_BINDINGS:
        supported = ", ".join(sorted(MOWA_ACTION_HEAD_BINDINGS))
        raise ValueError(f"Unsupported MoWA action head binding: {action_head_type}. Supported: {supported}")
    return MOWA_ACTION_HEAD_BINDINGS[action_head_type]


def append_layerwise_bridge_tokens(
    vl_embs_list: list[Any],
    encoder_attention_mask: Any,
    bridge_output: Any,
) -> tuple[list[Any], Any]:
    """Append MoWA bridge tokens to LayerwiseFM condition-side features.

    This helper is action-head-adapter code, not LayerwiseFM internals.
    """

    import torch

    layerwise_bridge = bridge_output.layerwise_condition_features
    if len(vl_embs_list) != len(layerwise_bridge):
        raise ValueError(
            "MoWA bridge layer count mismatch: "
            f"vl_layers={len(vl_embs_list)}, bridge_layers={len(layerwise_bridge)}."
        )

    adapted = []
    for layer_idx, (vl_embs, bridge_tokens) in enumerate(zip(vl_embs_list, layerwise_bridge)):
        if vl_embs.dim() != 3 or bridge_tokens.dim() != 3:
            raise ValueError("MoWA LayerwiseFM adapter expects [B, T, D] tensors.")
        if vl_embs.shape[0] != bridge_tokens.shape[0] or vl_embs.shape[-1] != bridge_tokens.shape[-1]:
            raise ValueError(
                "MoWA LayerwiseFM adapter shape mismatch at layer "
                f"{layer_idx}: vl={tuple(vl_embs.shape)}, bridge={tuple(bridge_tokens.shape)}."
            )
        adapted.append(torch.cat((vl_embs, bridge_tokens.to(dtype=vl_embs.dtype)), dim=1))

    if encoder_attention_mask is None:
        return adapted, None

    bridge_mask = bridge_output.attention_mask
    if bridge_mask.shape[0] != encoder_attention_mask.shape[0]:
        raise ValueError(
            "MoWA bridge attention mask batch mismatch: "
            f"encoder={tuple(encoder_attention_mask.shape)}, bridge={tuple(bridge_mask.shape)}."
        )
    adapted_mask = torch.cat((encoder_attention_mask.to(dtype=torch.bool), bridge_mask), dim=1)
    return adapted, adapted_mask
