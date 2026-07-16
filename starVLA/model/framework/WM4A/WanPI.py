# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanPI Framework — Wan2.2-TI2V World Model + Layer-wise Cross-DiT Flow-Matching.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone with a layer-wise cross-DiT flow-matching action head, inspired by π₀.

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D (30 blocks)
    → Multi-layer hidden_states [30 × (B, N, 3072)]
    → LayerwiseFlowmatchingActionHead (cross-DiT)
    → action predictions [B, chunk_len, action_dim]

Key differences from WanGR00T:
  - Action head: Layer-wise cross-DiT (not single-layer flow-matching)
  - Uses ALL transformer layers (not just last hidden state)
  - More expressive multi-scale feature fusion
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.mowa import (
    CrossViewAttentionAdapter,
    MultiViewDoneHead,
    MoWAFutureLatentPrior,
    MoWAFutureLatentPriorConfig,
    MoWAHLCGCI,
    MoWAHLCGCIConfig,
    MultiViewFutureFusion,
    MultiViewPatchGrid,
    flatten_view_batch,
    masked_future_flow_loss,
    resolve_cross_view_layer_indices,
    unflatten_view_batch,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _prepare_action_state(
    examples: List[dict],
    *,
    required: bool,
    expected_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """统一校验并组装训练/推理使用的当前机器人状态。"""

    present = ["state" in example and example["state"] is not None for example in examples]
    if not all(present):
        if required:
            missing = [index for index, value in enumerate(present) if not value]
            raise ValueError(
                "MoWA continuous state conditioning requires `state` for every sample; "
                f"missing batch indices={missing}."
            )
        if any(present):
            raise ValueError("Inconsistent `state` presence within one batch.")
        return None

    state = torch.as_tensor(np.asarray([example["state"] for example in examples]), device=device, dtype=dtype)
    if state.ndim != 3 or state.shape[1] != 1:
        raise ValueError(
            "MoWA action state must have shape [B,1,state_dim], "
            f"got {tuple(state.shape)}."
        )
    if state.shape[2] != expected_dim:
        raise ValueError(
            f"MoWA action state_dim mismatch: expected {expected_dim}, got {state.shape[2]}."
        )
    if not torch.isfinite(state).all():
        raise ValueError("MoWA action state contains NaN or Inf.")
    return state


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"MoWA contract `{name}` must be a tensor, got {type(value).__name__}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"MoWA contract `{name}` contains NaN or Inf.")


@dataclass
class WanPIDefaultConfig:
    """WanPI default parameters."""

    name: str = "WanPI"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
        }
    )

    # LayerwiseFM reads qwenvl.vl_hidden_dim and qwenvl.num_vl_layers
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "vl_hidden_dim": 1024,
            "num_vl_layers": 30,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 15,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "cross_attention_dim": 1024,
            },
        }
    )

    # === MoWA interface (optional) ===
    interface: dict = field(default_factory=dict)
    mowa: dict = field(default_factory=dict)


@FRAMEWORK_REGISTRY.register("WanPI")
class Wan_PI(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V + Layer-wise cross-DiT.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Layer-wise cross-DiT flow-matching head fed by all transformer layers
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanPIDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        # Auto-detect hidden size and num layers from world model
        wm_hidden = self.backbone.model.config.hidden_size
        # Cosmos uses transformer_blocks, Wan uses blocks
        if hasattr(self.backbone.transformer, "transformer_blocks"):
            num_blocks = len(self.backbone.transformer.transformer_blocks)
        else:
            num_blocks = len(self.backbone.transformer.blocks)

        # Project world model features to action model's cross-attention dim.
        # The world model runs under bfloat16 AMP, so hidden states are bfloat16;
        # we cast them to float32 before the projector so downstream modules
        # (action head, MoWA auxiliary heads) stay in a consistent dtype.
        cross_attn_dim = self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_attn_dim)

        # Sync vl_hidden_dim so LayerwiseFM action head stays consistent
        self.config.framework.qwenvl.vl_hidden_dim = cross_attn_dim
        self.config.framework.qwenvl.num_vl_layers = num_blocks

        # Resolve DiT shape from config (default to cross_attn_dim).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", cross_attn_dim)
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_blocks,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        self._setup_mowa_multiview(
            num_blocks=num_blocks,
            wm_hidden=wm_hidden,
            cross_attn_dim=cross_attn_dim,
            wm_dtype=next(self.backbone.transformer.parameters()).dtype,
        )
        self._mowa_state_debug_logged = False
        self._mowa_expected_num_blocks = num_blocks
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        self.mowa_validate_data_flow = bool(getattr(mowa_cfg, "validate_data_flow", False))
        self.mowa_validation_steps = int(getattr(mowa_cfg, "validation_steps", 2))
        if self.mowa_validation_steps <= 0:
            raise ValueError("framework.mowa.validation_steps must be positive.")
        self._mowa_train_validation_count = 0
        self._mowa_predict_validation_count = 0

        # Register hooks for ALL transformer blocks
        self._all_hidden_states = []
        self._all_hooks = []
        self._register_all_hooks()

        # MoWA modules (future latent prior + HLC-GCI)
        self._setup_mowa_future_latent_prior_loss()
        self._setup_mowa_hlc_gci_conditioning()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """允许单视角旧 checkpoint 为新增多视角模块使用明确的默认初始化。"""

        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        multiview_prefixes = (
            "cross_view_adapters.",
            "mowa_view_embeddings.",
            "mowa_action_view_embeddings.",
            "mowa_future_fusion.",
        )
        non_multiview_missing = [
            key for key in result.missing_keys if not key.startswith(multiview_prefixes)
        ]
        if result.missing_keys and not non_multiview_missing:
            logger.info(
                "[E003 multi-view] checkpoint lacks %d multi-view parameters; using configured initialization.",
                len(result.missing_keys),
            )
        if strict and (non_multiview_missing or result.unexpected_keys):
            raise RuntimeError(
                f"Error(s) in loading state_dict: missing={non_multiview_missing}, "
                f"unexpected={result.unexpected_keys}."
            )
        return result

    def _register_all_hooks(self):
        """Register forward hooks on ALL transformer blocks for layerwise features."""
        for hook in self._all_hooks:
            hook.remove()
        self._all_hooks.clear()

        if hasattr(self.backbone.transformer, "transformer_blocks"):
            blocks = self.backbone.transformer.transformer_blocks
        else:
            blocks = self.backbone.transformer.blocks
        if self.mowa_multiview_enabled:
            self._all_hooks.append(blocks[0].register_forward_pre_hook(self._add_view_embedding_pre_hook))
        for layer_index, block in enumerate(blocks):
            hook = block.register_forward_hook(self._make_capture_hook(layer_index))
            self._all_hooks.append(hook)

    def _make_capture_hook(self, layer_index: int):
        def _hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if (
                self._mowa_multiview_grid is not None
                and self.mowa_multiview_debug_mode != "disable_cross_view"
                and str(layer_index) in self.cross_view_adapters
            ):
                hidden = self.cross_view_adapters[str(layer_index)](hidden, self._mowa_multiview_grid)
            self._all_hidden_states.append(hidden)
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden

        return _hook

    def _add_view_embedding_pre_hook(self, module, inputs):
        if self._mowa_multiview_grid is None:
            return None
        hidden = inputs[0]
        grid = self._mowa_multiview_grid
        view_ids = torch.arange(grid.num_views, device=hidden.device).repeat(grid.batch_size)
        view_embedding = self.mowa_view_embeddings(view_ids).to(dtype=hidden.dtype).unsqueeze(1)
        return (hidden + view_embedding, *inputs[1:])

    def _setup_mowa_multiview(
        self,
        *,
        num_blocks: int,
        wm_hidden: int,
        cross_attn_dim: int,
        wm_dtype: torch.dtype,
    ) -> None:
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        multi_view_cfg = getattr(mowa_cfg, "multi_view", None)
        self.mowa_multiview_enabled = bool(getattr(multi_view_cfg, "enabled", False))
        self.mowa_multiview_debug_mode = str(getattr(multi_view_cfg, "debug_mode", "normal"))
        if self.mowa_multiview_debug_mode not in {
            "normal", "zero_wrist", "shuffle_wrist", "disable_cross_view"
        }:
            raise ValueError(f"Unsupported multi-view debug mode: {self.mowa_multiview_debug_mode}.")
        self._mowa_multiview_grid: MultiViewPatchGrid | None = None
        self._mowa_multiview_future_steps = 0
        self._mowa_multiview_debug_logged = False
        self.cross_view_adapters = torch.nn.ModuleDict()
        if not self.mowa_multiview_enabled:
            return

        num_views = len(tuple(getattr(multi_view_cfg, "views", ("main", "wrist"))))
        if num_views != 2:
            raise ValueError(f"E003 multi-view requires exactly two views, got {num_views}.")
        cross_view_cfg = getattr(mowa_cfg, "cross_view", None)
        configured_num_layers = getattr(cross_view_cfg, "num_layers", 10)
        layer_indices = resolve_cross_view_layer_indices(
            num_blocks,
            num_layers=None if configured_num_layers is None else int(configured_num_layers),
            start_layer_ratio=float(getattr(cross_view_cfg, "start_layer_ratio", 2.0 / 3.0)),
        )
        attention_heads = int(getattr(cross_view_cfg, "num_heads", 8))
        bottleneck_dim = int(getattr(cross_view_cfg, "bottleneck_dim", 512))
        if bool(getattr(cross_view_cfg, "enabled", True)):
            for layer_index in layer_indices:
                self.cross_view_adapters[str(layer_index)] = CrossViewAttentionAdapter(
                    wm_hidden,
                    attention_heads,
                    bottleneck_dim=bottleneck_dim,
                    gate_init=float(getattr(cross_view_cfg, "gate_init", 0.0)),
                    zero_init_output=bool(getattr(cross_view_cfg, "zero_init_output", True)),
                ).to(dtype=wm_dtype)
        self.mowa_view_embeddings = torch.nn.Embedding(num_views, wm_hidden).to(dtype=wm_dtype)
        torch.nn.init.normal_(self.mowa_view_embeddings.weight, std=1e-3)
        self.mowa_action_view_embeddings = torch.nn.Embedding(num_views, cross_attn_dim)
        torch.nn.init.normal_(self.mowa_action_view_embeddings.weight, std=1e-3)
        action_fusion_cfg = getattr(mowa_cfg, "action_fusion", None)
        self.mowa_future_fusion = MultiViewFutureFusion(
            cross_attn_dim,
            int(getattr(action_fusion_cfg, "num_heads", 8)),
            int(getattr(action_fusion_cfg, "num_queries", 8)),
        )
        done_cfg = getattr(mowa_cfg, "done_head", None)
        self.mowa_done_head = MultiViewDoneHead(
            wm_hidden,
            int(getattr(done_cfg, "hidden_dim", 512)),
        )
        self.mowa_done_loss_weight = float(getattr(done_cfg, "loss_weight", 1.0))
        future_loss_cfg = getattr(mowa_cfg, "future_loss", None)
        self.mowa_future_main_weight = float(getattr(future_loss_cfg, "main_weight", 1.0))
        self.mowa_future_wrist_weight = float(getattr(future_loss_cfg, "wrist_weight", 1.0))

    def _setup_mowa_future_latent_prior_loss(self) -> None:
        interface_cfg = getattr(self.config, "interface", None)
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        role_enabled = getattr(self.config, "experiment_role", None) == "mowa_future_latent_prior"
        self.mowa_future_latent_prior_loss_enabled = bool(
            getattr(mowa_cfg, "enable_future_latent_prior_loss", role_enabled)
        )
        self.mowa_future_latent_prior = None
        if self.mowa_multiview_enabled:
            # 双视角路径直接监督 Wan 的 future flow 输出，不再实例化旧 pooled MLP prior。
            return
        if not self.mowa_future_latent_prior_loss_enabled or interface_cfg is None:
            return
        self.mowa_future_latent_prior = MoWAFutureLatentPrior(
            MoWAFutureLatentPriorConfig(
                current_latent_dim=int(getattr(interface_cfg, "current_latent_dim", 1024)),
                text_hidden_dim=int(getattr(interface_cfg, "text_hidden_dim", 4096)),
                hidden_dim=int(getattr(interface_cfg, "hidden_dim", 2048)),
                future_latent_dim=int(getattr(interface_cfg, "future_latent_dim", 1024)),
                future_steps=int(getattr(interface_cfg, "future_steps", 1)),
                history_latent_dim=int(getattr(interface_cfg, "history_latent_dim", 0)),
                history_steps=int(getattr(interface_cfg, "history_steps", 0)),
            )
        )

    def _setup_mowa_hlc_gci_conditioning(self) -> None:
        interface_cfg = getattr(self.config, "interface", None)
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        role_enabled = getattr(self.config, "experiment_role", None) == "mowa_hlc_gci"
        self.mowa_hlc_gci_conditioning_enabled = bool(
            getattr(mowa_cfg, "enable_hlc_gci_conditioning", role_enabled)
        )
        self.mowa_hlc_gci = None
        self.mowa_hlc_gci_condition_token_count = 0
        if not self.mowa_hlc_gci_conditioning_enabled or interface_cfg is None:
            return
        cross_attn_dim = self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        self.mowa_hlc_gci_condition_token_count = int(getattr(interface_cfg, "condition_token_count", 4))
        self.mowa_hlc_gci = MoWAHLCGCI(
            MoWAHLCGCIConfig(
                history_latent_dim=int(getattr(interface_cfg, "history_latent_dim", 1024)),
                condition_hidden_dim=int(getattr(interface_cfg, "condition_hidden_dim", cross_attn_dim)),
                history_steps=int(getattr(interface_cfg, "history_steps", 10)),
                compressed_history_dim=int(getattr(interface_cfg, "compressed_history_dim", 512)),
                gate_hidden_dim=int(getattr(interface_cfg, "gate_hidden_dim", 256)),
            )
        )

    @staticmethod
    def _build_mowa_latent_batch(
        examples: List[dict],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not examples or not all(example.get(key) is not None for example in examples):
            return None
        return torch.stack(
            [
                torch.as_tensor(example[key], device=device, dtype=dtype)
                for example in examples
            ],
            dim=0,
        )

    def _pool_text_hidden(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool UMT5 text embeddings using the attention mask."""
        mask = attention_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
        pooled = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def _maybe_run_mowa_future_latent_prior_loss(
        self,
        current_context: torch.Tensor,
        pooled_text_hidden: torch.Tensor,
        examples: List[dict],
    ) -> dict | None:
        if not getattr(self, "mowa_future_latent_prior_loss_enabled", False):
            return None
        if self.mowa_future_latent_prior is None:
            raise RuntimeError("MoWA future latent prior loss is enabled but not initialized.")

        future_latent_target = self._build_mowa_latent_batch(
            examples,
            "mowa_future_latent_target",
            device=pooled_text_hidden.device,
            dtype=pooled_text_hidden.dtype,
        )
        if future_latent_target is None:
            return {
                "supervision_available": False,
                "loss": None,
                "losses": {},
            }
        future_valid_mask = self._build_mowa_latent_batch(
            examples, "mowa_future_valid_mask", device=pooled_text_hidden.device, dtype=torch.bool
        )
        done_target = self._build_mowa_latent_batch(
            examples, "mowa_future_done_target", device=pooled_text_hidden.device, dtype=pooled_text_hidden.dtype
        )
        history_latent = None
        history_valid_mask = None
        if self.mowa_future_latent_prior.config.history_steps > 0:
            history_latent = self._build_mowa_latent_batch(
                examples, "mowa_history_latent", device=pooled_text_hidden.device, dtype=pooled_text_hidden.dtype
            )
            history_valid_mask = self._build_mowa_latent_batch(
                examples, "mowa_history_valid_mask", device=pooled_text_hidden.device, dtype=torch.bool
            )

        latent_prior_dtype = next(self.mowa_future_latent_prior.parameters()).dtype
        loss, losses, output = self.mowa_future_latent_prior.compute_loss(
            current_context.to(dtype=latent_prior_dtype),
            pooled_text_hidden.to(dtype=latent_prior_dtype),
            future_latent_target.to(dtype=latent_prior_dtype),
            future_valid_mask=future_valid_mask,
            done_target=done_target,
            history_latent=history_latent.to(dtype=latent_prior_dtype) if history_latent is not None else None,
            history_valid_mask=history_valid_mask,
        )
        return {
            "supervision_available": True,
            "loss": loss,
            "losses": losses,
            "future_latent_mse": losses.get("future_latent_mse"),
            "done_loss": losses.get("done_loss"),
            "predicted_future_latent_shape": tuple(output.predicted_future_latent.shape),
        }

    def _maybe_apply_mowa_hlc_gci_conditioning(
        self,
        vl_embs_list: List[torch.Tensor],
        examples: List[dict],
    ) -> tuple[List[torch.Tensor], dict | None]:
        if not getattr(self, "mowa_hlc_gci_conditioning_enabled", False):
            return vl_embs_list, None
        if self.mowa_hlc_gci is None:
            raise RuntimeError("MoWA HLC-GCI conditioning is enabled but not initialized.")

        history_latent = self._build_mowa_latent_batch(
            examples,
            "mowa_history_latent",
            device=vl_embs_list[-1].device,
            dtype=vl_embs_list[-1].dtype,
        )
        if history_latent is None:
            return vl_embs_list, {
                "conditioned": False,
                "reason": "history_latent_missing",
            }

        history_steps = int(self.mowa_hlc_gci.config.history_steps)
        if history_latent.dim() != 3:
            raise ValueError(
                f"mowa_history_latent must be 3D [B, history_steps, D], got {tuple(history_latent.shape)}."
            )
        if history_latent.shape[1] != history_steps:
            raise ValueError(
                f"history_latent time steps mismatch: expected {history_steps}, "
                f"got {history_latent.shape[1]}."
            )
        if history_latent.shape[2] != self.mowa_hlc_gci.config.history_latent_dim:
            raise ValueError(
                "history_latent dim mismatch: expected "
                f"{self.mowa_hlc_gci.config.history_latent_dim}, got {history_latent.shape[2]}."
            )

        last_hidden = vl_embs_list[-1]
        token_count = min(self.mowa_hlc_gci_condition_token_count, int(last_hidden.shape[1]))
        if token_count <= 0:
            return vl_embs_list, {
                "conditioned": False,
                "reason": "no_condition_tokens_available",
            }
        condition_tokens = last_hidden[:, :token_count, :]
        hlc_dtype = next(self.mowa_hlc_gci.parameters()).dtype
        output = self.mowa_hlc_gci(
            history_latent.to(dtype=hlc_dtype),
            condition_tokens.to(dtype=hlc_dtype),
        )
        history_tokens = output.history_tokens
        conditioned_layers = [
            torch.cat((layer_hidden, history_tokens.to(device=layer_hidden.device, dtype=layer_hidden.dtype)), dim=1)
            for layer_hidden in vl_embs_list
        ]
        return conditioned_layers, {
            "conditioned": True,
            "history_latent_sequence_shape": tuple(history_latent.shape),
            "compressed_history_shape": tuple(output.compressed_history.shape),
            "gate_values_shape": tuple(output.gate_values.shape),
            "gated_condition_tokens_shape": tuple(output.gated_condition_tokens.shape),
            "history_tokens_shape": tuple(output.history_tokens.shape),
            "history_sequence_policy": "per_step_history_sequence_from_cache",
            "injection_policy": "append_history_tokens_to_each_layer_condition",
        }

    def _build_wan_inputs(self, examples: List[dict]) -> dict:
        """Prepare Wan2.2 build_inputs kwargs from examples.

        Prefer cached text/visual latents when present; fall back to raw
        instructions/images so the world model can encode on its own.
        """
        kwargs: dict = {}

        text_embeds = [example.get("text_embeds") for example in examples]
        text_attention_mask = [example.get("text_attention_mask") for example in examples]
        if all(t is not None for t in text_embeds) and all(m is not None for m in text_attention_mask):
            text_embeds_tensor = torch.stack([torch.as_tensor(t) for t in text_embeds], dim=0)
            text_attention_mask_tensor = torch.stack(
                [torch.as_tensor(m) for m in text_attention_mask], dim=0
            )
            # Cached latents may include a leading singleton batch dim [1, L, D].
            if text_embeds_tensor.dim() == 4 and text_embeds_tensor.shape[1] == 1:
                text_embeds_tensor = text_embeds_tensor.squeeze(1)
            if text_attention_mask_tensor.dim() == 3 and text_attention_mask_tensor.shape[1] == 1:
                text_attention_mask_tensor = text_attention_mask_tensor.squeeze(1)
            kwargs["text_embeds"] = text_embeds_tensor
            kwargs["text_attention_mask"] = text_attention_mask_tensor
        else:
            kwargs["instructions"] = [example["lang"] for example in examples]

        visual_latents = [example.get("visual_latent") for example in examples]
        if all(v is not None for v in visual_latents):
            kwargs["visual_latents"] = torch.stack([torch.as_tensor(v) for v in visual_latents], dim=0)
            normalized_flags = [bool(example.get("visual_latents_normalized", False)) for example in examples]
            if len(set(normalized_flags)) != 1:
                raise ValueError("A Wan batch cannot mix normalized and raw visual latent caches.")
            kwargs["visual_latents_normalized"] = normalized_flags[0]
        else:
            kwargs["images"] = [example["image"] for example in examples]

        return kwargs

    def _build_multiview_wan_inputs(self, examples: List[dict]) -> tuple[dict, dict]:
        """构造 [B,2,C,T,H,W] clean/noisy latent 和共享 timestep。"""

        device = next(self.backbone.transformer.parameters()).device
        history = self._build_mowa_latent_batch(
            examples, "mowa_multi_view_history_latents", device=device, dtype=torch.bfloat16
        )
        current = self._build_mowa_latent_batch(
            examples, "mowa_multi_view_current_latents", device=device, dtype=torch.bfloat16
        )
        future = self._build_mowa_latent_batch(
            examples, "mowa_multi_view_future_latents", device=device, dtype=torch.bfloat16
        )
        if history is None or current is None or future is None:
            raise ValueError("E003 multi-view requires synchronized history/current/future latent fields.")
        if current.dim() != 5 or history.dim() != 6 or future.dim() != 6:
            raise ValueError(
                "Expected history/future [B,V,T,C,H,W] and current [B,V,C,H,W], got "
                f"{tuple(history.shape)}, {tuple(current.shape)}, {tuple(future.shape)}."
            )
        if not (history.shape[:2] == current.shape[:2] == future.shape[:2]):
            raise ValueError("Main/wrist batch pairing or view count mismatch.")

        history = history.permute(0, 1, 3, 2, 4, 5)
        current = current.unsqueeze(3)
        future = future.permute(0, 1, 3, 2, 4, 5)
        clean = torch.cat((history, current, future), dim=3)
        if self.mowa_multiview_debug_mode == "zero_wrist":
            clean[:, 1] = 0
        elif self.mowa_multiview_debug_mode == "shuffle_wrist" and clean.shape[0] > 1:
            shuffled = torch.roll(torch.arange(clean.shape[0], device=device), 1)
            clean[:, 1] = clean[shuffled, 1]
        history = clean[:, :, :, : history.shape[3]]
        current = clean[:, :, :, history.shape[3] : history.shape[3] + 1]
        future = clean[:, :, :, history.shape[3] + 1 :]
        batch_size, num_views, _, total_time, height, width = clean.shape
        history_current_steps = history.shape[3] + 1
        future_steps = future.shape[3]

        timestep = torch.rand(batch_size, device=device, dtype=torch.float32)
        noise = torch.randn_like(future)
        t = timestep[:, None, None, None, None, None]
        noisy_future = (1.0 - t) * future + t * noise
        noisy = torch.cat((history, current, noisy_future), dim=3)
        flow_target = torch.zeros_like(clean)
        flow_target[:, :, :, history_current_steps:] = noise - future

        text_kwargs = self._build_wan_inputs(examples)
        text_embeds = text_kwargs.get("text_embeds")
        text_attention_mask = text_kwargs.get("text_attention_mask")
        if text_embeds is None or text_attention_mask is None:
            raise ValueError("E003 multi-view currently requires cached text embeddings.")
        text_embeds = text_embeds.to(device=device).repeat_interleave(num_views, dim=0)
        text_attention_mask = text_attention_mask.to(device=device).repeat_interleave(num_views, dim=0)
        wm_inputs = self.backbone.build_inputs(
            visual_latents=flatten_view_batch(noisy),
            visual_latents_normalized=True,
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        )
        p_t, p_h, p_w = self.backbone.transformer.config.patch_size
        patch_grid = MultiViewPatchGrid(
            batch_size=batch_size,
            num_views=num_views,
            time=total_time // p_t,
            height=height // p_h,
            width=width // p_w,
        )
        token_timestep = torch.zeros(
            batch_size * num_views,
            patch_grid.sequence_tokens,
            device=device,
            dtype=torch.long,
        )
        future_token_start = (history_current_steps // p_t) * patch_grid.spatial_tokens
        paired_timestep = timestep.repeat_interleave(num_views)
        token_timestep[:, future_token_start:] = (paired_timestep[:, None] * 1000).long()
        wm_inputs["timestep"] = token_timestep
        future_valid_mask = self._build_mowa_latent_batch(
            examples, "mowa_future_valid_mask", device=device, dtype=torch.bool
        )
        if future_valid_mask is None:
            future_valid_mask = torch.ones(batch_size, future_steps, device=device, dtype=torch.bool)
        future_valid_mask = future_valid_mask[:, None, :].expand(-1, num_views, -1)
        return wm_inputs, {
            "clean": clean,
            "flow_target": flow_target,
            "future_valid_mask": future_valid_mask,
            "grid": patch_grid,
            "history_current_steps": history_current_steps,
            "future_steps": future_steps,
            "future_token_start": future_token_start,
        }

    def _fuse_multiview_layerwise(self, vl_embs_list: List[torch.Tensor], context: dict) -> List[torch.Tensor]:
        grid = context["grid"]
        future_token_start = context["future_token_start"]
        view_ids = torch.arange(grid.num_views, device=vl_embs_list[-1].device)
        view_embedding = self.mowa_action_view_embeddings(view_ids)
        fused_layers = []
        for layer_hidden in vl_embs_list:
            paired = unflatten_view_batch(layer_hidden, grid.batch_size, grid.num_views)
            future_hidden = paired[:, :, future_token_start:, :]
            future_hidden = future_hidden + view_embedding[None, :, None, :].to(dtype=future_hidden.dtype)
            fused_layers.append(self.mowa_future_fusion(future_hidden))
        return fused_layers

    def _compute_multiview_future_loss(self, prediction: torch.Tensor, context: dict) -> dict:
        grid = context["grid"]
        paired_prediction = unflatten_view_batch(prediction, grid.batch_size, grid.num_views)
        target = context["flow_target"]
        future_start = context["history_current_steps"]
        prediction_future = paired_prediction[:, :, :, future_start:]
        target_future = target[:, :, :, future_start:]
        per_view = masked_future_flow_loss(
            prediction_future,
            target_future,
            context["future_valid_mask"],
        )
        loss_main = per_view[:, 0].mean()
        loss_wrist = per_view[:, 1].mean()
        loss_total = self.mowa_future_main_weight * loss_main + self.mowa_future_wrist_weight * loss_wrist
        return {
            "loss_future_main": loss_main,
            "loss_future_wrist": loss_wrist,
            "loss_future_total": loss_total,
            "main_future_pred_norm": prediction_future[:, 0].float().norm().detach(),
            "wrist_future_pred_norm": prediction_future[:, 1].float().norm().detach(),
        }

    def _compute_multiview_done_loss(
        self,
        last_hidden: torch.Tensor,
        context: dict,
        examples: List[dict],
    ) -> dict:
        grid = context["grid"]
        paired = unflatten_view_batch(last_hidden, grid.batch_size, grid.num_views)
        temporal_hidden = paired.reshape(
            grid.batch_size,
            grid.num_views,
            grid.time,
            grid.spatial_tokens,
            last_hidden.shape[-1],
        )
        future_start = context["future_token_start"] // grid.spatial_tokens
        future_hidden = temporal_hidden[:, :, future_start:]
        if future_hidden.shape[2] != context["future_steps"]:
            raise ValueError(
                "E003 done head requires one Wan temporal patch per latent timestep, "
                f"got {future_hidden.shape[2]} patches for {context['future_steps']} future steps."
            )
        done_logits = self.mowa_done_head(future_hidden)
        done_target = self._build_mowa_latent_batch(
            examples, "mowa_future_done_target", device=done_logits.device, dtype=done_logits.dtype
        )
        if done_target is None or done_target.shape != done_logits.shape:
            raise ValueError(
                "E003 multi-view done target must be [B,T_future], "
                f"got {None if done_target is None else tuple(done_target.shape)} for "
                f"{tuple(done_logits.shape)} logits."
            )
        if not torch.all((done_target == 0) | (done_target == 1)):
            raise ValueError("E003 multi-view done target must be binary.")
        done_loss = torch.nn.functional.binary_cross_entropy_with_logits(done_logits, done_target)
        return {
            "loss_done": done_loss,
            "done_logits": done_logits,
            "done_target": done_target,
        }

    def _validate_mowa_examples_once(self, examples: List[dict], *, phase: str) -> None:
        if not self.mowa_multiview_enabled:
            return
        if not examples:
            raise ValueError(f"MoWA {phase} contract requires a non-empty batch.")
        required = {
            "action",
            "state",
            "text_embeds",
            "text_attention_mask",
            "mowa_multi_view_history_latents",
            "mowa_multi_view_current_latents",
            "mowa_multi_view_future_latents",
            "mowa_future_valid_mask",
        }
        for batch_index, example in enumerate(examples):
            missing = sorted(key for key in required if key not in example or example[key] is None)
            if missing:
                raise ValueError(
                    f"MoWA {phase} contract sample {batch_index} is missing required fields: {missing}."
                )
            action = torch.as_tensor(example["action"])
            if action.ndim != 2 or action.shape[0] < self.action_horizon:
                raise ValueError(
                    f"MoWA {phase} action must be [T>={self.action_horizon},action_dim], "
                    f"got {tuple(action.shape)} at batch index {batch_index}."
                )
            if action.shape[1] != int(self.config.framework.action_model.action_dim):
                raise ValueError(
                    f"MoWA {phase} action_dim mismatch: expected "
                    f"{self.config.framework.action_model.action_dim}, got {action.shape[1]}."
                )
            _require_finite_tensor(f"{phase}.action[{batch_index}]", action)

            history = torch.as_tensor(example["mowa_multi_view_history_latents"])
            current = torch.as_tensor(example["mowa_multi_view_current_latents"])
            future = torch.as_tensor(example["mowa_multi_view_future_latents"])
            if history.ndim != 5 or current.ndim != 4 or future.ndim != 5:
                raise ValueError(
                    f"MoWA {phase} per-sample latent shapes must be history/future [V,T,C,H,W] "
                    f"and current [V,C,H,W], got {tuple(history.shape)}, "
                    f"{tuple(current.shape)}, {tuple(future.shape)}."
                )
            if not (history.shape[0] == current.shape[0] == future.shape[0] == 2):
                raise ValueError(f"MoWA {phase} requires exactly two synchronized views.")
            if not (history.shape[2:] == future.shape[2:] and current.shape[1:] == future.shape[2:]):
                raise ValueError(f"MoWA {phase} main/wrist latent channel/spatial shapes are inconsistent.")
            for name, tensor in (("history", history), ("current", current), ("future", future)):
                _require_finite_tensor(f"{phase}.{name}_latents[{batch_index}]", tensor)

            future_mask = torch.as_tensor(example["mowa_future_valid_mask"])
            if future_mask.ndim != 1 or future_mask.shape[0] != future.shape[1]:
                raise ValueError(
                    f"MoWA {phase} future mask must match future T={future.shape[1]}, "
                    f"got {tuple(future_mask.shape)}."
                )

    def _validate_mowa_wan_flow_once(
        self,
        *,
        phase: str,
        wm_inputs: dict,
        wm_outputs,
        captured_layers: List[torch.Tensor],
        action_layers: List[torch.Tensor],
        context: dict,
    ) -> None:
        grid = context["grid"]
        hidden_states = wm_inputs.get("hidden_states")
        if hidden_states is None or hidden_states.ndim != 5:
            raise ValueError(f"MoWA {phase} Wan hidden_states must be [2B,C,T,H,W].")
        expected_flat_batch = grid.batch_size * grid.num_views
        if hidden_states.shape[0] != expected_flat_batch:
            raise ValueError(
                f"MoWA {phase} flattened Wan batch mismatch: expected {expected_flat_batch}, "
                f"got {hidden_states.shape[0]}."
            )
        _require_finite_tensor(f"{phase}.wan_input", hidden_states)
        for key in ("encoder_hidden_states", "encoder_attention_mask", "timestep"):
            value = wm_inputs.get(key)
            if value is None or value.shape[0] != expected_flat_batch:
                raise ValueError(f"MoWA {phase} `{key}` is missing or not expanded to B*V.")
            _require_finite_tensor(f"{phase}.{key}", value)
        if len(captured_layers) != self._mowa_expected_num_blocks:
            raise ValueError(
                f"MoWA {phase} captured {len(captured_layers)} Wan layers, "
                f"expected {self._mowa_expected_num_blocks}."
            )
        for layer_index, hidden in enumerate(captured_layers):
            if hidden.ndim != 3 or hidden.shape[:2] != (expected_flat_batch, grid.sequence_tokens):
                raise ValueError(
                    f"MoWA {phase} Wan layer {layer_index} shape mismatch: {tuple(hidden.shape)}."
                )
            _require_finite_tensor(f"{phase}.wan_layer[{layer_index}]", hidden)
        if len(action_layers) != self._mowa_expected_num_blocks:
            raise ValueError(f"MoWA {phase} LayerwiseFM condition layer count mismatch.")
        for layer_index, hidden in enumerate(action_layers):
            if hidden.ndim != 3 or hidden.shape[0] != grid.batch_size:
                raise ValueError(
                    f"MoWA {phase} action condition layer {layer_index} must restore batch B, "
                    f"got {tuple(hidden.shape)}."
                )
            _require_finite_tensor(f"{phase}.action_condition[{layer_index}]", hidden)
        prediction = wm_outputs.sample
        if prediction.shape != hidden_states.shape:
            raise ValueError(
                f"MoWA {phase} Wan output shape {tuple(prediction.shape)} does not match input "
                f"{tuple(hidden_states.shape)}."
            )
        _require_finite_tensor(f"{phase}.wan_output", prediction)

    def _validate_mowa_losses_once(self, output: dict) -> None:
        required = (
            "action_loss",
            "loss_future_main",
            "loss_future_wrist",
            "loss_future_total",
            "loss_done",
            "loss_multiview_total",
        )
        for key in required:
            value = output.get(key)
            if value is None or value.numel() != 1:
                raise ValueError(f"MoWA training output `{key}` must be a scalar tensor.")
            _require_finite_tensor(f"train.{key}", value)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        validate_data_flow = (
            self.mowa_multiview_enabled
            and self.mowa_validate_data_flow
            and self._mowa_train_validation_count < self.mowa_validation_steps
        )
        if validate_data_flow:
            self._validate_mowa_examples_once(examples, phase="train")
        actions = [example["action"] for example in examples]

        multiview_context = None
        if self.mowa_multiview_enabled:
            wm_inputs, multiview_context = self._build_multiview_wan_inputs(examples)
            self._mowa_multiview_grid = multiview_context["grid"]
            self._mowa_multiview_future_steps = multiview_context["future_steps"]
        else:
            wm_inputs = self.backbone.build_inputs(**self._build_wan_inputs(examples))
        encoder_hidden_states = wm_inputs["encoder_hidden_states"]
        encoder_attention_mask = wm_inputs.get("encoder_attention_mask")
        pooled_text_hidden = (
            self._pool_text_hidden(encoder_hidden_states, encoder_attention_mask)
            if encoder_attention_mask is not None
            else encoder_hidden_states.mean(dim=1)
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            captured_layers = list(self._all_hidden_states)
            vl_embs_list = [self.wm_projector(h.to(dtype=torch.float32)) for h in captured_layers]
            if multiview_context is not None:
                vl_embs_list = self._fuse_multiview_layerwise(vl_embs_list, multiview_context)
                if validate_data_flow:
                    self._validate_mowa_wan_flow_once(
                        phase="train",
                        wm_inputs=wm_inputs,
                        wm_outputs=wm_outputs,
                        captured_layers=captured_layers,
                        action_layers=vl_embs_list,
                        context=multiview_context,
                    )
            base_hidden = vl_embs_list[-1]
            if multiview_context is not None and not self._mowa_multiview_debug_logged:
                grid = multiview_context["grid"]
                logger.info(
                    "[E003 multi-view] input=[B=%d,V=%d,C=%d,T=%d,H=%d,W=%d] flat_batch=%d "
                    "patch_grid=(%d,%d,%d) cross_view=[B*T=%d,V*S=%d,D] "
                    "future_tokens=%d action_condition=%s",
                    grid.batch_size,
                    grid.num_views,
                    wm_inputs["hidden_states"].shape[1],
                    wm_inputs["hidden_states"].shape[2],
                    wm_inputs["hidden_states"].shape[3],
                    wm_inputs["hidden_states"].shape[4],
                    wm_inputs["hidden_states"].shape[0],
                    grid.time,
                    grid.height,
                    grid.width,
                    grid.batch_size * grid.time,
                    grid.num_views * grid.spatial_tokens,
                    grid.sequence_tokens - multiview_context["future_token_start"],
                    tuple(base_hidden.shape),
                )
                self._mowa_multiview_debug_logged = True
        # gradient checkpointing 会在 backward 期间重新执行 Wan block hook；
        # 保留当前 grid，下一次 multi-view forward 会覆盖它。

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            vl_embs_list, mowa_hlc_gci = self._maybe_apply_mowa_hlc_gci_conditioning(vl_embs_list, examples)
            base_hidden = vl_embs_list[-1]
            current_context = base_hidden.mean(dim=1)
            if multiview_context is None:
                mowa_future_latent_prior = self._maybe_run_mowa_future_latent_prior_loss(
                    current_context,
                    pooled_text_hidden,
                    examples,
                )
                multiview_future_loss = None
            else:
                mowa_future_latent_prior = None
                multiview_future_loss = self._compute_multiview_future_loss(
                    wm_outputs.sample,
                    multiview_context,
                )
                multiview_done_loss = self._compute_multiview_done_loss(
                    captured_layers[-1], multiview_context, examples
                )

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
                if self.config and hasattr(self.config, "framework")
                else 2
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            action_valid_mask = self._build_mowa_latent_batch(
                examples, "mowa_action_valid_mask", device=base_hidden.device, dtype=torch.bool
            )
            if action_valid_mask is None:
                action_valid_mask = torch.ones(
                    actions_target.shape[:2], device=base_hidden.device, dtype=torch.bool
                )
            if action_valid_mask.shape != actions_target.shape[:2]:
                raise ValueError(
                    "MoWA action valid mask must match [B,action_horizon], "
                    f"got {tuple(action_valid_mask.shape)} for {tuple(actions_target.shape[:2])}."
                )
            action_valid_mask_repeated = action_valid_mask.repeat(repeated_diffusion_steps, 1)

            state_repeated = None
            if validate_data_flow:
                state = _prepare_action_state(
                    examples,
                    required=self.action_model.state_encoder is not None,
                    expected_dim=int(self.config.framework.action_model.state_dim),
                    device=base_hidden.device,
                    dtype=base_hidden.dtype,
                )
            else:
                raw_state = [example["state"] for example in examples] if "state" in examples[0] else None
                state = (
                    torch.as_tensor(np.asarray(raw_state), device=base_hidden.device, dtype=base_hidden.dtype)
                    if raw_state is not None
                    else None
                )
            if state is not None:
                # The dataloader returns state as [B, 1, state_dim]; keep the
                # singleton step dim so the action head's MLP emits
                # [B, 1, hidden_size] and can be concatenated with future/action
                # tokens along the sequence dimension.
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
                if self.mowa_multiview_enabled and not self._mowa_state_debug_logged:
                    logger.info(
                        "[E003 state] shape=%s dtype=%s norm=%.6f state_encoder=%s",
                        tuple(state.shape),
                        state.dtype,
                        float(state.float().norm()),
                        type(self.action_model.state_encoder).__name__,
                    )
                    self._mowa_state_debug_logged = True

            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                action_valid_mask=action_valid_mask_repeated,
            )

        output = {"action_loss": action_loss, "mowa_state_conditioned": state is not None}
        if multiview_future_loss is not None:
            output.update(multiview_future_loss)
            output.update(multiview_done_loss)
            output["loss_multiview_total"] = (
                multiview_future_loss["loss_future_total"]
                + self.mowa_done_loss_weight * multiview_done_loss["loss_done"]
            )
            output["mowa_future_latent_prior_loss"] = output["loss_multiview_total"]
            output["cross_view_gate_by_layer"] = {
                layer: adapter.gate.detach() for layer, adapter in self.cross_view_adapters.items()
            }
            if self.cross_view_adapters:
                output["cross_view_gate_mean"] = torch.stack(
                    [adapter.gate.float() for adapter in self.cross_view_adapters.values()]
                ).mean().detach()
                output["cross_view_output_norm"] = torch.stack(
                    [adapter.last_output_norm for adapter in self.cross_view_adapters.values()]
                ).mean()
                output["cross_view_grad_norm"] = torch.stack(
                    [adapter.last_grad_norm for adapter in self.cross_view_adapters.values()]
                ).mean()
        if mowa_future_latent_prior is not None:
            output["mowa_future_latent_prior_available"] = mowa_future_latent_prior["supervision_available"]
            if mowa_future_latent_prior["loss"] is not None:
                output["mowa_future_latent_prior_loss"] = mowa_future_latent_prior["loss"]
                output["mowa_future_latent_prior_losses"] = mowa_future_latent_prior["losses"]
                output["future_latent_mse"] = mowa_future_latent_prior["future_latent_mse"]
        if mowa_hlc_gci is not None:
            output["mowa_hlc_gci_conditioned"] = mowa_hlc_gci["conditioned"]
            output["mowa_hlc_gci_reason"] = mowa_hlc_gci.get("reason")
            if mowa_hlc_gci.get("compressed_history_shape") is not None:
                output["mowa_hlc_gci_compressed_history_shape"] = mowa_hlc_gci["compressed_history_shape"]
                output["mowa_hlc_gci_gate_values_shape"] = mowa_hlc_gci["gate_values_shape"]
                output["mowa_hlc_gci_gated_condition_tokens_shape"] = mowa_hlc_gci["gated_condition_tokens_shape"]
                output["mowa_hlc_gci_history_latent_sequence_shape"] = mowa_hlc_gci["history_latent_sequence_shape"]
                output["mowa_hlc_gci_history_tokens_shape"] = mowa_hlc_gci["history_tokens_shape"]
                output["mowa_hlc_gci_history_sequence_policy"] = mowa_hlc_gci["history_sequence_policy"]
                output["mowa_hlc_gci_injection_policy"] = mowa_hlc_gci["injection_policy"]
        if validate_data_flow:
            self._validate_mowa_losses_once(output)
            self._mowa_train_validation_count += 1
            logger.info(
                "[E003 contract] training data flow validation %d/%d passed.",
                self._mowa_train_validation_count,
                self.mowa_validation_steps,
            )
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        validate_data_flow = (
            self.mowa_multiview_enabled
            and self.mowa_validate_data_flow
            and self._mowa_predict_validation_count < self.mowa_validation_steps
        )
        if validate_data_flow:
            raise RuntimeError(
                "E003 multi-view inference data flow is not implemented: predict_action() currently "
                "uses the legacy single-view Wan path and cannot validate dual-view future rollout. "
                "Refusing silent single-view fallback."
            )

        # Apply obs resize only when using raw images, not cached visual latents.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size and not all(example.get("visual_latent") is not None for example in examples):
            for example in examples:
                example["image"] = to_pil_preserve(example["image"])
            examples = [
                {**example, "image": img}
                for example, img in zip(
                    examples,
                    resize_images(
                        [example["image"] for example in examples],
                        target_size=train_obs_image_size,
                    ),
                )
            ]

        wm_inputs = self.backbone.build_inputs(**self._build_wan_inputs(examples))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(self._all_hidden_states)
            vl_embs_list = [self.wm_projector(h) for h in vl_embs_list]

        if validate_data_flow:
            state = _prepare_action_state(
                examples,
                required=self.action_model.state_encoder is not None,
                expected_dim=int(self.config.framework.action_model.state_dim),
                device=vl_embs_list[-1].device,
                dtype=vl_embs_list[-1].dtype,
            )
        else:
            raw_state = [example["state"] for example in examples] if "state" in examples[0] else None
            state = (
                torch.as_tensor(
                    np.asarray(raw_state), device=vl_embs_list[-1].device, dtype=vl_embs_list[-1].dtype
                )
                if raw_state is not None
                else None
            )

        with torch.autocast("cuda", dtype=torch.float32):
            vl_embs_list, mowa_hlc_gci = self._maybe_apply_mowa_hlc_gci_conditioning(vl_embs_list, examples)
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions, "mowa_state_conditioned": state is not None}
        if mowa_hlc_gci is not None:
            output["mowa_hlc_gci_conditioned"] = mowa_hlc_gci["conditioned"]
            output["mowa_hlc_gci_reason"] = mowa_hlc_gci.get("reason")
        return output


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "WanPI"
    cfg.framework.qwenvl = {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
        "num_vl_layers": 30,
    }
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    }

    model: Wan_PI = Wan_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Predicted Action: {normalized_actions}")
    print("Finished")
