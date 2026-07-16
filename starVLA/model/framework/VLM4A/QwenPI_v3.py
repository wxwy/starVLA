# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI_v3 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key improvements over QwenPI
─────────────────────────────
1. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).  This reduces the
   action head parameter count by ~(vl_hidden / dit_hidden)² while keeping
   the full layer-wise cross-attention structure.

2. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``),
   following the π₀.5 design.  This lets the VLM attend to state without
   any extra encoder module.

Together these two features bring QwenPI_v3 close to all the core
capabilities of π₀.5 within a single open-weight VLM framework.

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.mowa import (
    MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
    MOWA_FUTURE_FEATURE_SOURCE_ALIASES,
    MOWA_FUTURE_FULL_HEADS,
    MOWA_FUTURE_HEAD_OUTPUT_DIMS,
    MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
    MoWAActionBridge,
    MoWAActionBridgeConfig,
    MoWAActionBridgeOutput,
    MoWAFutureFeatureHeads,
    MoWAFutureFeatureHeadsConfig,
    MoWAFutureGatedHeads,
    MoWAFutureGatedHeadsConfig,
    MoWAFutureFeatures,
    append_layerwise_bridge_tokens,
    resolve_mowa_action_head_binding,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI_v3
#  - Same shape as QwenPIDefaultConfig (see QwenPI.py) but introduces the
#    optional `action_dit_hidden_dim` knob inside diffusion_model_cfg.
#  - Setting action_dit_hidden_dim to a value smaller than the VLM hidden
#    size lets the Action DiT run at a "compressed" latent dim while
#    `Qwen_PI_v3.project_layers` does the LayerNorm+Linear compression of
#    each VL hidden state to that dim.
#  - Leaving it None (or omitting it from YAML) reproduces the QwenPI
#    behaviour: DiT hidden = VLM hidden, projection becomes nn.Identity().
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPI_v3DefaultConfig:
    """QwenPI_v3 framework default parameters.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v3"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
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
                # When set (e.g. 1024), DiT internal hidden = action_dit_hidden_dim
                # and Qwen_PI_v3.project_layers compress VL hidden to this dim.
                # When None, DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v3")
class Qwen_PI_v3(baseframework):
    """
    Multimodal vision-language-action model (QwenPI_v3 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    Focus: predict a future action chunk conditioned on multi-view images
    and a natural-language instruction (with optional discretised state prefix).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenPI_v3DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors
        # (embedding output + every layer's output), and we keep the last
        # `num_hidden_layers` of them for layer-wise cross-attn — so the DiT
        # depth and project_layers count must match `num_hidden_layers` exactly.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head,
        # so that LayerwiseFlowmatchingActionHead constructs DiT at the right size.
        # If the user did not specify it, fall back to the LLM hidden size
        # (i.e. behave like QwenPI: project_layers becomes nn.Identity()).
        #
        # NOTE: `action_dit_hidden_dim` is a framework-side hint only — it is
        # NOT a DiT constructor kwarg, so we keep it out of diffusion_model_cfg
        # and instead pass it through `populate_layerwise_dit_cfg`, which writes
        # the canonical DiT-shape fields (input_embedding_dim, cross_attention_dim,
        # num_attention_heads).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.  The action head
        # is intentionally agnostic of qwenvl.* — it only consumes this dict.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        # This explicitly decouples VL representation size from action DiT latent size.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self._setup_mowa_future_supervision_loss()
        self._setup_mowa_layerwise_bridge_coupling()
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        self.starflow_validate_data_flow = bool(
            getattr(self.config.framework, "name", None) == "StarFlowVLA"
            and getattr(mowa_cfg, "validate_data_flow", False)
        )
        self.starflow_validation_steps = int(getattr(mowa_cfg, "validation_steps", 2))
        if self.starflow_validation_steps <= 0:
            raise ValueError("framework.mowa.validation_steps must be positive.")
        self._starflow_train_validation_count = 0
        self._starflow_predict_validation_count = 0

    def _starflow_should_validate(self, phase: str) -> bool:
        count = (
            getattr(self, "_starflow_train_validation_count", 0)
            if phase == "train"
            else getattr(self, "_starflow_predict_validation_count", 0)
        )
        return bool(getattr(self, "starflow_validate_data_flow", False)) and count < int(
            getattr(self, "starflow_validation_steps", 2)
        )

    @staticmethod
    def _starflow_require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"StarFlow contract `{name}` must be a tensor.")
        if not torch.isfinite(value).all():
            raise ValueError(f"StarFlow contract `{name}` contains NaN or Inf.")

    def _validate_starflow_examples(self, examples: List[dict], *, phase: str) -> None:
        if not examples:
            raise ValueError(f"StarFlow {phase} contract requires a non-empty batch.")
        action_dim = int(self.config.framework.action_model.action_dim)
        state_mode = str(getattr(self.config.framework, "state_mode", "discretized_instruction"))
        required = {"image", "lang", "state"}
        if phase == "train":
            required.add("action")
            if self.mowa_future_supervision_loss_enabled:
                required.update(("mowa_future_targets", "mowa_future_masks"))
        for batch_index, example in enumerate(examples):
            missing = sorted(key for key in required if key not in example or example[key] is None)
            if missing:
                raise ValueError(
                    f"StarFlow {phase} sample {batch_index} is missing required fields: {missing}."
                )
            if not isinstance(example["lang"], str) or not example["lang"].strip():
                raise ValueError(f"StarFlow {phase} instruction must be a non-empty string.")
            images = example["image"] if isinstance(example["image"], (list, tuple)) else [example["image"]]
            if not images or any(image is None for image in images):
                raise ValueError(f"StarFlow {phase} sample {batch_index} has no valid image.")
            state = torch.as_tensor(example["state"])
            if state.ndim != 2 or state.shape[0] != 1:
                raise ValueError(
                    f"StarFlow {phase} state must be [1,state_dim], got {tuple(state.shape)}."
                )
            self._starflow_require_finite(f"{phase}.state[{batch_index}]", state)
            if state_mode == "continuous_head" and state.shape[1] != int(
                self.config.framework.action_model.state_dim
            ):
                raise ValueError("StarFlow continuous state_dim does not match action head config.")
            if phase == "train":
                action = torch.as_tensor(example["action"])
                if action.ndim != 2 or action.shape[0] < self.action_horizon or action.shape[1] != action_dim:
                    raise ValueError(
                        f"StarFlow train action must be [T>={self.action_horizon},{action_dim}], "
                        f"got {tuple(action.shape)}."
                    )
                self._starflow_require_finite(f"train.action[{batch_index}]", action)
                if self.mowa_future_supervision_loss_enabled:
                    targets = example["mowa_future_targets"]
                    masks = example["mowa_future_masks"]
                    for head in self.mowa_future_supervision_active_heads:
                        if head not in targets or head not in masks:
                            raise ValueError(f"StarFlow future supervision is missing head `{head}`.")
                        target = torch.as_tensor(targets[head])
                        self._starflow_require_finite(f"train.future_target.{head}", target)

    def _validate_starflow_hidden_flow(
        self,
        vl_embs_list: List[torch.Tensor],
        attention_mask: torch.Tensor | None,
        raw_last_hidden: torch.Tensor,
        *,
        phase: str,
    ) -> None:
        if len(vl_embs_list) != self.num_action_dit_layers:
            raise ValueError(
                f"StarFlow {phase} hidden layer count mismatch: got {len(vl_embs_list)}, "
                f"expected {self.num_action_dit_layers}."
            )
        batch_size = raw_last_hidden.shape[0]
        self._starflow_require_finite(f"{phase}.raw_last_hidden", raw_last_hidden)
        for layer_index, hidden in enumerate(vl_embs_list):
            if hidden.ndim != 3 or hidden.shape[0] != batch_size or hidden.shape[2] != self.action_dit_hidden_dim:
                raise ValueError(
                    f"StarFlow {phase} projected hidden layer {layer_index} shape mismatch: "
                    f"{tuple(hidden.shape)}."
                )
            self._starflow_require_finite(f"{phase}.projected_hidden[{layer_index}]", hidden)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
                raise ValueError(f"StarFlow {phase} attention mask shape mismatch.")
            self._starflow_require_finite(f"{phase}.attention_mask", attention_mask)

    def _validate_starflow_output(self, output: dict, *, phase: str, batch_size: int) -> None:
        if phase == "train":
            action_loss = output.get("action_loss")
            if action_loss is None or action_loss.numel() != 1:
                raise ValueError("StarFlow train output `action_loss` must be a scalar tensor.")
            self._starflow_require_finite("train.action_loss", action_loss)
            if self.mowa_layerwise_bridge_coupling_enabled and not output.get(
                "mowa_layerwise_bridge_coupled", False
            ):
                raise ValueError("StarFlow MoWA bridge is enabled but did not condition the action head.")
            future_loss = output.get("mowa_future_supervision_loss")
            if self.mowa_future_supervision_loss_enabled:
                if future_loss is None:
                    raise ValueError("StarFlow future supervision is enabled but produced no loss.")
                self._starflow_require_finite("train.mowa_future_supervision_loss", future_loss)
        else:
            actions = np.asarray(output.get("normalized_actions"))
            expected = (
                batch_size,
                self.action_horizon,
                int(self.config.framework.action_model.action_dim),
            )
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(
                    f"StarFlow predict output must be finite with shape {expected}, got {actions.shape}."
                )
            if self.mowa_layerwise_bridge_coupling_enabled and not output.get(
                "mowa_layerwise_bridge_coupled", False
            ):
                raise ValueError("StarFlow inference bypassed the enabled MoWA bridge.")

    def _setup_mowa_layerwise_bridge_coupling(self) -> None:
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        self.mowa_layerwise_bridge_coupling_enabled = bool(
            getattr(mowa_cfg, "enable_layerwise_bridge_token_coupling", False)
        )
        self.mowa_layerwise_bridge = None
        if not self.mowa_layerwise_bridge_coupling_enabled:
            return

        self.mowa_layerwise_bridge_token_intervention = getattr(
            mowa_cfg,
            "layerwise_bridge_token_intervention",
            "baseline",
        )
        self.mowa_layerwise_bridge_feature_source = getattr(
            mowa_cfg,
            "layerwise_bridge_feature_source",
            MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
        )
        supported_feature_sources = {
            MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
            *MOWA_FUTURE_FEATURE_SOURCE_ALIASES,
        }
        if self.mowa_layerwise_bridge_feature_source not in supported_feature_sources:
            supported = ", ".join(sorted(supported_feature_sources))
            raise ValueError(
                "Unsupported MoWA layerwise bridge feature source: "
                f"{self.mowa_layerwise_bridge_feature_source}. Supported: {supported}"
            )
        supported_interventions = {
            "baseline",
            "zero",
            "batch_shuffle",
            "head_mask_control",
        }
        if self.mowa_layerwise_bridge_token_intervention not in supported_interventions:
            supported = ", ".join(sorted(supported_interventions))
            raise ValueError(
                "Unsupported MoWA layerwise bridge token intervention: "
                f"{self.mowa_layerwise_bridge_token_intervention}. Supported: {supported}"
            )
        if (
            self.mowa_layerwise_bridge_token_intervention == "head_mask_control"
            and self.mowa_layerwise_bridge_feature_source not in MOWA_FUTURE_FEATURE_SOURCE_ALIASES
        ):
            raise ValueError(
                "MoWA head_mask_control intervention requires a future feature-head source, "
                f"got {self.mowa_layerwise_bridge_feature_source}."
            )

        action_head_type = getattr(self.config.framework.action_model, "action_model_type", None)
        binding = resolve_mowa_action_head_binding(action_head_type)
        if not binding.adapter_helper_implemented:
            raise ValueError(
                "MoWA layerwise bridge coupling is not implemented for "
                f"action_model_type={action_head_type}."
            )
        if not binding.framework_forward_integrated:
            raise ValueError(
                "MoWA layerwise bridge coupling is not integrated in this framework for "
                f"action_model_type={action_head_type}."
            )
        if binding.injection_mode != "append_bridge_tokens_to_condition_side":
            raise ValueError(
                "MoWA layerwise bridge coupling expects condition-side token append, "
                f"got injection_mode={binding.injection_mode}."
            )

        num_bridge_tokens = int(getattr(mowa_cfg, "num_bridge_tokens", 1))
        wam_feature_dim = int(getattr(mowa_cfg, "wam_feature_dim", self.action_dit_hidden_dim))
        action_hidden_dim = int(getattr(mowa_cfg, "action_hidden_dim", self.action_dit_hidden_dim))
        if action_hidden_dim != self.action_dit_hidden_dim:
            raise ValueError(
                "MoWA layerwise bridge coupling requires bridge action_hidden_dim to match "
                "the projected action condition hidden dim: "
                f"action_hidden_dim={action_hidden_dim}, action_dit_hidden_dim={self.action_dit_hidden_dim}."
            )
        self.mowa_layerwise_bridge_feature_projector = (
            nn.Identity()
            if wam_feature_dim == self.action_dit_hidden_dim
            else nn.Linear(self.action_dit_hidden_dim, wam_feature_dim)
        )
        self.mowa_layerwise_bridge_future_feature_heads = None
        self.mowa_layerwise_bridge_head_mask_projector = None
        self.mowa_future_gated_heads_enabled = self._mowa_gated_heads_enabled()
        self.mowa_last_layerwise_bridge_gated_heads_summary = None
        self.mowa_last_future_supervision_gated_heads_summary = None
        if self.mowa_layerwise_bridge_feature_source in MOWA_FUTURE_FEATURE_SOURCE_ALIASES:
            active_heads = getattr(
                mowa_cfg,
                "layerwise_bridge_active_heads",
                MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            )
            self.mowa_layerwise_bridge_active_heads = tuple(active_heads)
            unknown_heads = [
                head for head in self.mowa_layerwise_bridge_active_heads if head not in MOWA_FUTURE_FULL_HEADS
            ]
            if unknown_heads:
                raise ValueError(f"Unknown MoWA layerwise bridge active heads: {unknown_heads}")
            self.mowa_layerwise_bridge_future_feature_heads = self._build_mowa_future_head_module(
                hidden_dim=wam_feature_dim,
                action_outcome_loss_type="mse",
            )
            head_mask_dim = sum(
                MOWA_FUTURE_HEAD_OUTPUT_DIMS[head]
                for head in MOWA_FUTURE_CONSTRUCTIBLE_HEADS
            )
            self.mowa_layerwise_bridge_head_mask_projector = nn.Linear(
                head_mask_dim,
                wam_feature_dim,
            )
        self.mowa_layerwise_bridge = MoWAActionBridge(
            MoWAActionBridgeConfig(
                wam_feature_dim=wam_feature_dim,
                action_hidden_dim=action_hidden_dim,
                num_action_layers=self.num_action_dit_layers,
                num_bridge_tokens=num_bridge_tokens,
            )
        )

    def _setup_mowa_future_supervision_loss(self) -> None:
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        self.mowa_future_supervision_loss_enabled = bool(
            getattr(
                mowa_cfg,
                "enable_future_supervision_loss",
                getattr(mowa_cfg, "enable_p0_supervision_loss", False),
            )
        )
        self.mowa_future_supervision_probe = None
        self.mowa_last_future_supervision_gated_heads_summary = None
        self.mowa_future_supervision_active_heads = tuple(
            getattr(
                mowa_cfg,
                "future_supervision_active_heads",
                getattr(mowa_cfg, "p0_supervision_active_heads", MOWA_FUTURE_CONSTRUCTIBLE_HEADS),
            )
        )
        if not self.mowa_future_supervision_loss_enabled:
            return

        hidden_dim = int(
            getattr(
                mowa_cfg,
                "future_supervision_hidden_dim",
                getattr(mowa_cfg, "p0_supervision_hidden_dim", 32),
            )
        )
        action_outcome_loss_type = str(
            getattr(
                mowa_cfg,
                "future_supervision_action_outcome_loss_type",
                getattr(mowa_cfg, "p0_supervision_action_outcome_loss_type", "mse"),
            )
        )
        self.mowa_future_supervision_probe = self._build_mowa_future_head_module(
            hidden_dim=hidden_dim,
            action_outcome_loss_type=action_outcome_loss_type,
        )

    def _mowa_gated_heads_enabled(self) -> bool:
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        gated_cfg = getattr(mowa_cfg, "gated_heads", None) if mowa_cfg is not None else None
        return bool(getattr(gated_cfg, "enabled", False))

    def _build_mowa_future_head_module(
        self,
        *,
        hidden_dim: int,
        action_outcome_loss_type: str,
    ):
        heads_config = MoWAFutureFeatureHeadsConfig(
            input_dim=self.action_dit_hidden_dim,
            hidden_dim=hidden_dim,
            action_outcome_loss_type=action_outcome_loss_type,
        )
        if not self._mowa_gated_heads_enabled():
            return MoWAFutureFeatureHeads(heads_config)

        gated_cfg = getattr(getattr(self.config.framework, "mowa", None), "gated_heads", None)
        init_gate_value = float(getattr(gated_cfg, "init_gate_value", 0.5))
        comparison_scope = str(
            getattr(gated_cfg, "comparison_scope", "single_fullheads_control_only")
        )
        allow_per_head_sweep = bool(getattr(gated_cfg, "allow_per_head_sweep", False))
        return MoWAFutureGatedHeads(
            MoWAFutureGatedHeadsConfig(
                heads_config=heads_config,
                init_gate_value=init_gate_value,
                comparison_scope=comparison_scope,
                allow_per_head_sweep=allow_per_head_sweep,
            )
        )

    @staticmethod
    def _rewrite_mowa_checkpoint_state_dict_keys_for_compatibility(state_dict) -> None:
        future_prefix = "mowa_layerwise_bridge_future_feature_heads."
        legacy_prefixes = (
            "mowa_layerwise_bridge_future_heads.",
            "mowa_layerwise_bridge_p0_heads.",
        )
        for key in list(state_dict.keys()):
            for legacy_prefix in legacy_prefixes:
                if not key.startswith(legacy_prefix):
                    continue
                future_key = key.replace(legacy_prefix, future_prefix, 1)
                if future_key not in state_dict:
                    state_dict[future_key] = state_dict[key]
                state_dict.pop(key, None)
                break

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix == "":
            self._rewrite_mowa_checkpoint_state_dict_keys_for_compatibility(state_dict)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply_mowa_layerwise_bridge_head_mask_control(
        self,
        future_features: MoWAFutureFeatures,
    ) -> MoWAFutureFeatures:
        if self.mowa_layerwise_bridge_head_mask_projector is None:
            raise RuntimeError("MoWA head_mask_control requires initialized future feature heads.")
        head_vectors = []
        for head in MOWA_FUTURE_CONSTRUCTIBLE_HEADS:
            value = future_features.head_outputs[head]
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            head_vectors.append(value)
        controlled_features = self.mowa_layerwise_bridge_head_mask_projector(
            torch.cat(head_vectors, dim=-1)
        )
        return MoWAFutureFeatures(
            hidden_features=controlled_features,
            head_outputs={
                head: future_features.head_outputs[head]
                for head in MOWA_FUTURE_CONSTRUCTIBLE_HEADS
            },
            active_heads=MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            masked_heads=tuple(
                head for head in MOWA_FUTURE_FULL_HEADS if head not in MOWA_FUTURE_CONSTRUCTIBLE_HEADS
            ),
        )

    def _apply_mowa_layerwise_bridge_intervention(
        self,
        bridge_output: MoWAActionBridgeOutput,
    ) -> tuple[MoWAActionBridgeOutput, dict]:
        intervention = getattr(self, "mowa_layerwise_bridge_token_intervention", "baseline")
        metadata = {
            "intervention": intervention,
            "intervention_applied": intervention == "baseline",
            "intervention_note": None,
        }
        if intervention == "baseline":
            return bridge_output, metadata
        if intervention == "zero":
            metadata["intervention_applied"] = True
            return (
                MoWAActionBridgeOutput(
                    layerwise_condition_features=tuple(
                        torch.zeros_like(layer_features)
                        for layer_features in bridge_output.layerwise_condition_features
                    ),
                    attention_mask=bridge_output.attention_mask,
                    active_heads=bridge_output.active_heads,
                    masked_heads=bridge_output.masked_heads,
                ),
                metadata,
            )
        if intervention == "batch_shuffle":
            batch_size = bridge_output.layerwise_condition_features[0].shape[0]
            if batch_size <= 1:
                metadata["intervention_applied"] = False
                metadata["intervention_note"] = "batch_shuffle_not_applied_due_to_batch_size"
                return bridge_output, metadata
            metadata["intervention_applied"] = True
            return (
                MoWAActionBridgeOutput(
                    layerwise_condition_features=tuple(
                        torch.roll(layer_features, shifts=1, dims=0)
                        for layer_features in bridge_output.layerwise_condition_features
                    ),
                    attention_mask=torch.roll(bridge_output.attention_mask, shifts=1, dims=0),
                    active_heads=bridge_output.active_heads,
                    masked_heads=bridge_output.masked_heads,
                ),
                metadata,
            )
        if intervention == "head_mask_control":
            metadata["intervention_applied"] = True
            metadata["intervention_note"] = "rebuilt_from_constructible_head_outputs"
            return (
                MoWAActionBridgeOutput(
                    layerwise_condition_features=bridge_output.layerwise_condition_features,
                    attention_mask=bridge_output.attention_mask,
                    active_heads=bridge_output.active_heads,
                    masked_heads=bridge_output.masked_heads,
                ),
                metadata,
            )
        raise RuntimeError(f"Unhandled MoWA layerwise bridge token intervention: {intervention}")

    def _build_mowa_layerwise_bridge_future_features(self, hidden_features: torch.Tensor) -> MoWAFutureFeatures:
        source = getattr(
            self,
            "mowa_layerwise_bridge_feature_source",
            MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
        )
        if source == MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE:
            hidden_features = self.mowa_layerwise_bridge_feature_projector(hidden_features)
            return MoWAFutureFeatures(
                hidden_features=hidden_features,
                head_outputs={},
                active_heads=(MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,),
                masked_heads=(),
            )
        if source in MOWA_FUTURE_FEATURE_SOURCE_ALIASES:
            if self.mowa_layerwise_bridge_future_feature_heads is None:
                raise RuntimeError("MoWA future feature-head source is enabled but not initialized.")
            active_heads = getattr(
                self,
                "mowa_layerwise_bridge_active_heads",
                MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            )
            masks = {head: head in active_heads for head in MOWA_FUTURE_FULL_HEADS}
            future_features = self.mowa_layerwise_bridge_future_feature_heads.future_features(hidden_features, masks)
            gate_summary = getattr(
                self.mowa_layerwise_bridge_future_feature_heads,
                "gate_summary",
                None,
            )
            self.mowa_last_layerwise_bridge_gated_heads_summary = (
                gate_summary() if callable(gate_summary) else None
            )
            return future_features
        raise RuntimeError(f"Unhandled MoWA layerwise bridge feature source: {source}")

    @staticmethod
    def _mask_aware_pool_last_hidden(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask-aware pooling over the sequence dimension.

        When ``attention_mask`` is provided (shape ``[B, seq]``), padding
        positions are excluded before averaging; otherwise a plain ``mean``
        over dim=1 is used as a fallback.
        """
        if attention_mask is None:
            return last_hidden.mean(dim=1)
        mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _maybe_run_mowa_future_supervision_loss(
        self,
        hidden_features: torch.Tensor,
        examples: List[dict],
    ) -> dict | None:
        if not bool(getattr(self, "mowa_future_supervision_loss_enabled", False)):
            return None
        supervision_probe = getattr(self, "mowa_future_supervision_probe", None)
        if supervision_probe is None:
            raise RuntimeError("MoWA future supervision loss is enabled but not initialized.")
        if not examples or not all("mowa_future_targets" in example for example in examples):
            return {
                "supervision_available": False,
                "loss": None,
                "losses": {},
                "active_heads": (),
                "masked_heads": MOWA_FUTURE_FULL_HEADS,
            }

        probe_dtype = next(supervision_probe.parameters()).dtype
        hidden_features = hidden_features.to(dtype=probe_dtype)
        targets = {}
        masks = {}
        device = hidden_features.device
        active_heads = tuple(
            getattr(
                self,
                "mowa_future_supervision_active_heads",
                MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            )
        )
        for head in MOWA_FUTURE_FULL_HEADS:
            head_active = head in active_heads and all(
                bool((example.get("mowa_future_masks") or {}).get(head, False)) for example in examples
            )
            masks[head] = head_active
            if not head_active:
                continue
            values = [example["mowa_future_targets"][head] for example in examples]
            targets[head] = torch.as_tensor(values, device=device, dtype=hidden_features.dtype)

        if not any(masks.values()):
            return {
                "supervision_available": False,
                "loss": None,
                "losses": {},
                "active_heads": (),
                "masked_heads": MOWA_FUTURE_FULL_HEADS,
            }

        loss, losses, _ = supervision_probe.compute_loss(hidden_features, targets, masks)
        gate_summary = getattr(supervision_probe, "gate_summary", None)
        self.mowa_last_future_supervision_gated_heads_summary = (
            gate_summary() if callable(gate_summary) else None
        )
        return {
            "supervision_available": True,
            "loss": loss,
            "losses": losses,
            "active_heads": tuple(head for head in MOWA_FUTURE_FULL_HEADS if bool(masks.get(head, False))),
            "masked_heads": tuple(head for head in MOWA_FUTURE_FULL_HEADS if not bool(masks.get(head, False))),
            "gated_heads_summary": self.mowa_last_future_supervision_gated_heads_summary,
        }

    def _maybe_apply_mowa_layerwise_bridge_coupling(
        self,
        vl_embs_list: List[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> tuple[List[torch.Tensor], Optional[torch.Tensor], dict | None]:
        if not getattr(self, "mowa_layerwise_bridge_coupling_enabled", False):
            return vl_embs_list, encoder_attention_mask, None
        if self.mowa_layerwise_bridge is None:
            raise RuntimeError("MoWA layerwise bridge coupling is enabled but not initialized.")

        hidden_features = self._mask_aware_pool_last_hidden(
            vl_embs_list[-1], encoder_attention_mask
        )
        future_features = self._build_mowa_layerwise_bridge_future_features(hidden_features)
        if (
            getattr(self, "mowa_layerwise_bridge_token_intervention", "baseline")
            == "head_mask_control"
        ):
            future_features = self._apply_mowa_layerwise_bridge_head_mask_control(future_features)
        bridge_output = self.mowa_layerwise_bridge(future_features)
        bridge_output, intervention_metadata = self._apply_mowa_layerwise_bridge_intervention(bridge_output)
        adapted_vl_embs_list, adapted_attention_mask = append_layerwise_bridge_tokens(
            vl_embs_list,
            encoder_attention_mask,
            bridge_output,
        )
        first_layer_tokens = bridge_output.layerwise_condition_features[0]
        metadata = {
            "coupled": True,
            "token_shape": tuple(first_layer_tokens.shape),
            "bridge_token_shape": tuple(first_layer_tokens.shape),
            "adapted_vl_embed_shape": (
                tuple(adapted_vl_embs_list[0].shape) if adapted_vl_embs_list else None
            ),
            "attention_mask_shape": (
                tuple(adapted_attention_mask.shape) if adapted_attention_mask is not None else None
            ),
            "feature_source": self.mowa_layerwise_bridge_feature_source,
            "active_heads": bridge_output.active_heads,
            "masked_heads": bridge_output.masked_heads,
        }
        metadata.update(intervention_metadata)
        return adapted_vl_embs_list, adapted_attention_mask, metadata

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        align_dtype = bool(getattr(mowa_cfg, "enable_qwenpi_projector_dtype_alignment", False))
        projected = []
        for proj, vl_h in zip(self.project_layers, vl_embs_list):
            first_param = next(proj.parameters(), None) if align_dtype else None
            if align_dtype and first_param is not None:
                vl_h = vl_h.to(dtype=first_param.dtype)
            projected.append(proj(vl_h))
        return projected

    def _encode_vl_hidden_states(
        self, batch_images: List, instructions: List[str]
    ) -> tuple:
        """Run QwenVL, project hidden states, and return (layer-wise embeddings, attention_mask) for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            raw_last_hidden = qwenvl_outputs.hidden_states[-1]
            vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers:])
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list, attention_mask, raw_last_hidden

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        validate_data_flow = self._starflow_should_validate("train")
        if validate_data_flow:
            self._validate_starflow_examples(examples, phase="train")
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples]  # List[ndarray (T, action_dim)]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        raw_state = state
        instructions, state = self._prepare_state_condition(instructions, state)
        if validate_data_flow:
            state_mode = str(getattr(self.config.framework, "state_mode", "discretized_instruction"))
            if state_mode == "discretized_instruction":
                if state is not None or raw_state is None or any("[STATE]" not in item for item in instructions):
                    raise ValueError("StarFlow discretized state did not enter the instruction path.")
            elif state_mode == "continuous_head" and state is None:
                raise ValueError("StarFlow continuous state did not enter the action-head path.")

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask, raw_last_hidden = self._encode_vl_hidden_states(
            batch_images,
            instructions,
        )
        if validate_data_flow:
            self._validate_starflow_hidden_flow(
                vl_embs_list,
                backbone_attention_mask,
                raw_last_hidden,
                phase="train",
            )
        base_hidden = vl_embs_list[-1]
        pooled_text_hidden = self._mask_aware_pool_last_hidden(raw_last_hidden, backbone_attention_mask)

        # Step 2: compute flow-matching loss over the action chunk
        with torch.autocast("cuda", dtype=torch.float32):
            # Align labels: keep only the last action_horizon timesteps.
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)
            mowa_future_supervision = self._maybe_run_mowa_future_supervision_loss(
                self._mask_aware_pool_last_hidden(base_hidden, backbone_attention_mask), examples
            )

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 16) if self.config and self.config.trainer else 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat every VLM layer embedding to match the duplicated action batch.
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            vl_embs_list_repeated, backbone_attention_mask, mowa_bridge_metadata = (
                self._maybe_apply_mowa_layerwise_bridge_coupling(
                    vl_embs_list_repeated,
                    backbone_attention_mask,
                )
            )
            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
            )

        output = {"action_loss": action_loss}
        if mowa_bridge_metadata is not None:
            output["mowa_layerwise_bridge_coupled"] = mowa_bridge_metadata["coupled"]
            output["mowa_layerwise_bridge_intervention"] = mowa_bridge_metadata["intervention"]
            output["mowa_layerwise_bridge_intervention_applied"] = mowa_bridge_metadata[
                "intervention_applied"
            ]
            output["mowa_layerwise_bridge_intervention_note"] = mowa_bridge_metadata[
                "intervention_note"
            ]
            output["mowa_layerwise_bridge_token_shape"] = mowa_bridge_metadata["token_shape"]
            output["mowa_layerwise_bridge_bridge_token_shape"] = mowa_bridge_metadata[
                "bridge_token_shape"
            ]
            output["mowa_layerwise_bridge_adapted_vl_embed_shape"] = mowa_bridge_metadata[
                "adapted_vl_embed_shape"
            ]
            output["mowa_layerwise_bridge_attention_mask_shape"] = mowa_bridge_metadata[
                "attention_mask_shape"
            ]
            output["mowa_layerwise_bridge_feature_source"] = mowa_bridge_metadata["feature_source"]
            output["mowa_layerwise_bridge_active_heads"] = mowa_bridge_metadata["active_heads"]
            output["mowa_layerwise_bridge_masked_heads"] = mowa_bridge_metadata["masked_heads"]
            if self.mowa_last_layerwise_bridge_gated_heads_summary is not None:
                output["mowa_future_gated_heads_enabled"] = True
                output["mowa_layerwise_bridge_gated_heads_summary"] = (
                    self.mowa_last_layerwise_bridge_gated_heads_summary
                )
        if mowa_future_supervision is not None:
            output["mowa_future_supervision_available"] = mowa_future_supervision["supervision_available"]
            output["mowa_future_supervision_active_heads"] = mowa_future_supervision["active_heads"]
            output["mowa_future_supervision_masked_heads"] = mowa_future_supervision["masked_heads"]
            if mowa_future_supervision["loss"] is not None:
                output["mowa_future_supervision_loss"] = mowa_future_supervision["loss"]
                output["mowa_future_supervision_losses"] = mowa_future_supervision["losses"]
            if mowa_future_supervision.get("gated_heads_summary") is not None:
                output["mowa_future_gated_heads_enabled"] = True
                output["mowa_future_supervision_gated_heads_summary"] = mowa_future_supervision[
                    "gated_heads_summary"
                ]
        if validate_data_flow:
            self._validate_starflow_output(output, phase="train", batch_size=len(examples))
            self._starflow_train_validation_count += 1
            logger.info(
                "[StarFlow contract] training data flow validation %d/%d passed.",
                self._starflow_train_validation_count,
                self.starflow_validation_steps,
            )
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """

        validate_data_flow = self._starflow_should_validate("predict")
        if validate_data_flow:
            self._validate_starflow_examples(examples, phase="predict")
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None

        raw_state = state
        instructions, state = self._prepare_state_condition(instructions, state)
        if validate_data_flow:
            state_mode = str(getattr(self.config.framework, "state_mode", "discretized_instruction"))
            if state_mode == "discretized_instruction":
                if state is not None or raw_state is None or any("[STATE]" not in item for item in instructions):
                    raise ValueError("StarFlow discretized state did not enter the inference instruction path.")
            elif state_mode == "continuous_head" and state is None:
                raise ValueError("StarFlow continuous state did not enter the inference action-head path.")

        # Optionally resize images to the resolution used during training.
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask, raw_last_hidden = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        if validate_data_flow:
            self._validate_starflow_hidden_flow(
                vl_embs_list,
                backbone_attention_mask,
                raw_last_hidden,
                phase="predict",
            )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        # Step 2: run the flow-matching sampler to produce the denoised action chunk.
        with torch.autocast("cuda", dtype=torch.float32):
            vl_embs_list, backbone_attention_mask, mowa_bridge_metadata = (
                self._maybe_apply_mowa_layerwise_bridge_coupling(
                    vl_embs_list,
                    backbone_attention_mask,
                )
            )
            pred_actions = self.action_model.predict_action(
                vl_embs_list, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions}
        if mowa_bridge_metadata is not None:
            output["mowa_layerwise_bridge_coupled"] = mowa_bridge_metadata["coupled"]
            output["mowa_layerwise_bridge_intervention"] = mowa_bridge_metadata["intervention"]
            output["mowa_layerwise_bridge_intervention_applied"] = mowa_bridge_metadata[
                "intervention_applied"
            ]
            output["mowa_layerwise_bridge_intervention_note"] = mowa_bridge_metadata[
                "intervention_note"
            ]
            output["mowa_layerwise_bridge_token_shape"] = mowa_bridge_metadata["token_shape"]
            output["mowa_layerwise_bridge_bridge_token_shape"] = mowa_bridge_metadata[
                "bridge_token_shape"
            ]
            output["mowa_layerwise_bridge_adapted_vl_embed_shape"] = mowa_bridge_metadata[
                "adapted_vl_embed_shape"
            ]
            output["mowa_layerwise_bridge_attention_mask_shape"] = mowa_bridge_metadata[
                "attention_mask_shape"
            ]
            output["mowa_layerwise_bridge_feature_source"] = mowa_bridge_metadata["feature_source"]
            output["mowa_layerwise_bridge_active_heads"] = mowa_bridge_metadata["active_heads"]
            output["mowa_layerwise_bridge_masked_heads"] = mowa_bridge_metadata["masked_heads"]
        if validate_data_flow:
            self._validate_starflow_output(output, phase="predict", batch_size=len(examples))
            self._starflow_predict_validation_count += 1
            logger.info(
                "[StarFlow contract] inference data flow validation %d/%d passed.",
                self._starflow_predict_validation_count,
                self.starflow_validation_steps,
            )
        return output

    def _prepare_state_condition(
        self,
        instructions: List[str],
        state: Optional[List[np.ndarray]],
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        """Default QwenPI_v3 state path: encode state into instruction tokens."""
        if state is None:
            return instructions, None
        instructions = self.add_discretized_state_to_instruction(instructions, state)
        return instructions, None

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            state_str = self.state2str_transform(state[0])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    args.config_yaml = "examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = Qwen_PI_v3(cfg)
    # mdl = Qwen_PI.from_pretrained(ckpt)
    # ckpt = "/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/"
    # ckpt += "1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # # Advance: try forward model with dataloader
    # # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True

    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch)
