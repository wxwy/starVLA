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
        injection_mode="add_bridge_summary_to_action_hidden_state",
        implemented=True,
        notes=(
            "MLP/OFT heads consume action hidden states rather than condition tokens.",
            "MoWA adapter can add a bridge-token summary to action hidden states before predict_action.",
        ),
    ),
    "VLA_Adapter": MoWAActionHeadBinding(
        action_head_type="VLA_Adapter",
        condition_kind="layerwise_task_action_hidden_features",
        injection_mode="insert_bridge_tokens_before_action_queries",
        implemented=True,
        notes=(
            "VLA_Adapter consumes [B, Layers, Total_Len, D] hidden states and keeps action queries at the tail.",
            "MoWA adapter can insert bridge tokens before the action-query suffix without modifying VLA_Adapter internals.",
        ),
    ),
    "DiT-S": MoWAActionHeadBinding(
        action_head_type="DiT-S",
        condition_kind="single_condition_sequence",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=True,
        notes=(
            "DiT heads consume a single condition sequence.",
            "MoWA adapter can append one selected bridge layer to the condition sequence with an expanded attention mask.",
        ),
    ),
    "DiT-B": MoWAActionHeadBinding(
        action_head_type="DiT-B",
        condition_kind="single_condition_sequence",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=True,
        notes=(
            "GR00T-style DiT heads consume a single condition sequence rather than layerwise features.",
            "MoWA adapter can append one selected bridge layer to the condition sequence with an expanded attention mask.",
        ),
    ),
    "DiT-L": MoWAActionHeadBinding(
        action_head_type="DiT-L",
        condition_kind="single_condition_sequence",
        injection_mode="append_bridge_tokens_to_condition_side",
        implemented=True,
        notes=(
            "GR00T-style DiT heads consume a single condition sequence rather than layerwise features.",
            "MoWA adapter can append one selected bridge layer to the condition sequence with an expanded attention mask.",
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


def append_single_sequence_bridge_tokens(
    condition_features: Any,
    encoder_attention_mask: Any,
    bridge_output: Any,
    *,
    layer_index: int = -1,
) -> tuple[Any, Any]:
    """Append MoWA bridge tokens to a single-sequence condition action head."""

    import torch

    bridge_tokens = bridge_output.layerwise_condition_features[layer_index]
    if condition_features.dim() != 3 or bridge_tokens.dim() != 3:
        raise ValueError("MoWA single-sequence adapter expects [B, T, D] tensors.")
    if condition_features.shape[0] != bridge_tokens.shape[0] or condition_features.shape[-1] != bridge_tokens.shape[-1]:
        raise ValueError(
            "MoWA single-sequence adapter shape mismatch: "
            f"condition={tuple(condition_features.shape)}, bridge={tuple(bridge_tokens.shape)}."
        )
    adapted = torch.cat((condition_features, bridge_tokens.to(dtype=condition_features.dtype)), dim=1)
    if encoder_attention_mask is None:
        return adapted, None
    adapted_mask = torch.cat(
        (
            encoder_attention_mask.to(dtype=torch.bool),
            bridge_output.attention_mask,
        ),
        dim=1,
    )
    return adapted, adapted_mask


def fuse_mlp_bridge_features(
    action_hidden_features: Any,
    bridge_output: Any,
    *,
    layer_index: int = -1,
    scale: float = 1.0,
) -> Any:
    """Fuse MoWA bridge tokens into MLP/OFT action hidden states without changing head internals."""

    bridge_tokens = bridge_output.layerwise_condition_features[layer_index]
    if action_hidden_features.dim() != 3 or bridge_tokens.dim() != 3:
        raise ValueError("MoWA MLP adapter expects [B, T, D] tensors.")
    if action_hidden_features.shape[0] != bridge_tokens.shape[0] or action_hidden_features.shape[-1] != bridge_tokens.shape[-1]:
        raise ValueError(
            "MoWA MLP adapter shape mismatch: "
            f"hidden={tuple(action_hidden_features.shape)}, bridge={tuple(bridge_tokens.shape)}."
        )
    bridge_summary = bridge_tokens.to(dtype=action_hidden_features.dtype).mean(dim=1, keepdim=True)
    return action_hidden_features + bridge_summary * scale


def append_vla_adapter_bridge_tokens(
    actions_hidden_states: Any,
    bridge_output: Any,
    *,
    action_query_num: int,
) -> Any:
    """Insert MoWA bridge tokens before VLA_Adapter action-query tokens.

    VLA_Adapter expects action queries at the tail, so bridge tokens must be
    inserted between task/vision tokens and the action-query suffix.
    """

    import torch

    if actions_hidden_states.dim() != 4:
        raise ValueError("MoWA VLA_Adapter adapter expects hidden states shaped [B, Layers, T, D].")
    if action_query_num <= 0:
        raise ValueError("MoWA VLA_Adapter adapter expects action_query_num > 0.")
    layerwise_bridge = bridge_output.layerwise_condition_features
    if actions_hidden_states.shape[1] != len(layerwise_bridge):
        raise ValueError(
            "MoWA VLA_Adapter bridge layer count mismatch: "
            f"hidden_layers={actions_hidden_states.shape[1]}, bridge_layers={len(layerwise_bridge)}."
        )
    if action_query_num >= actions_hidden_states.shape[2]:
        raise ValueError(
            "MoWA VLA_Adapter adapter requires task tokens before action queries: "
            f"total_len={actions_hidden_states.shape[2]}, action_query_num={action_query_num}."
        )

    bridge = torch.stack(
        [tokens.to(dtype=actions_hidden_states.dtype) for tokens in layerwise_bridge],
        dim=1,
    )
    if actions_hidden_states.shape[0] != bridge.shape[0] or actions_hidden_states.shape[-1] != bridge.shape[-1]:
        raise ValueError(
            "MoWA VLA_Adapter adapter shape mismatch: "
            f"hidden={tuple(actions_hidden_states.shape)}, bridge={tuple(bridge.shape)}."
        )
    task_tokens = actions_hidden_states[:, :, :-action_query_num, :]
    action_queries = actions_hidden_states[:, :, -action_query_num:, :]
    return torch.cat((task_tokens, bridge, action_queries), dim=2)


def describe_mowa_action_head_bindings() -> tuple[dict[str, Any], ...]:
    """Return JSON-ready MoWA action-head adapter capabilities."""

    return tuple(
        {
            "action_head_type": binding.action_head_type,
            "condition_kind": binding.condition_kind,
            "injection_mode": binding.injection_mode,
            "adapter_helper_implemented": binding.implemented,
            "notes": list(binding.notes),
        }
        for binding in MOWA_ACTION_HEAD_BINDINGS.values()
    )
