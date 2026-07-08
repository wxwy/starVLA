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
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.mowa import (
    MoWAFutureLatentPrior,
    MoWAFutureLatentPriorConfig,
    MoWAHLCGCI,
    MoWAHLCGCIConfig,
)
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


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

        # Project world model features to action model's cross-attention dim
        cross_attn_dim = self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_attn_dim)

        # Sync vl_hidden_dim so LayerwiseFM action head stays consistent
        self.config.framework.qwenvl.vl_hidden_dim = cross_attn_dim
        self.config.framework.qwenvl.num_vl_layers = num_blocks

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Register hooks for ALL transformer blocks
        self._all_hidden_states = []
        self._all_hooks = []
        self._register_all_hooks()

        # MoWA modules (future latent prior + HLC-GCI)
        self._setup_mowa_future_latent_prior_loss()
        self._setup_mowa_hlc_gci_conditioning()

    def _register_all_hooks(self):
        """Register forward hooks on ALL transformer blocks for layerwise features."""
        for hook in self._all_hooks:
            hook.remove()
        self._all_hooks.clear()

        if hasattr(self.backbone.transformer, "transformer_blocks"):
            blocks = self.backbone.transformer.transformer_blocks
        else:
            blocks = self.backbone.transformer.blocks
        for block in blocks:
            hook = block.register_forward_hook(self._capture_all_hook)
            self._all_hooks.append(hook)

    def _capture_all_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._all_hidden_states.append(output[0])
        else:
            self._all_hidden_states.append(output)

    def _setup_mowa_future_latent_prior_loss(self) -> None:
        interface_cfg = getattr(self.config, "interface", None)
        mowa_cfg = getattr(self.config.framework, "mowa", None)
        role_enabled = getattr(self.config, "experiment_role", None) == "mowa_future_latent_prior"
        self.mowa_future_latent_prior_loss_enabled = bool(
            getattr(mowa_cfg, "enable_future_latent_prior_loss", role_enabled)
        )
        self.mowa_future_latent_prior = None
        if not self.mowa_future_latent_prior_loss_enabled or interface_cfg is None:
            return
        self.mowa_future_latent_prior = MoWAFutureLatentPrior(
            MoWAFutureLatentPriorConfig(
                current_latent_dim=int(getattr(interface_cfg, "current_latent_dim", 1024)),
                text_hidden_dim=int(getattr(interface_cfg, "text_hidden_dim", 4096)),
                hidden_dim=int(getattr(interface_cfg, "hidden_dim", 2048)),
                future_latent_dim=int(getattr(interface_cfg, "future_latent_dim", 1024)),
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
        pooled_text_hidden: torch.Tensor,
        examples: List[dict],
    ) -> dict | None:
        if not getattr(self, "mowa_future_latent_prior_loss_enabled", False):
            return None
        if self.mowa_future_latent_prior is None:
            raise RuntimeError("MoWA future latent prior loss is enabled but not initialized.")

        current_latent = self._build_mowa_latent_batch(
            examples,
            "mowa_current_latent",
            device=pooled_text_hidden.device,
            dtype=pooled_text_hidden.dtype,
        )
        future_latent_target = self._build_mowa_latent_batch(
            examples,
            "mowa_future_latent_target",
            device=pooled_text_hidden.device,
            dtype=pooled_text_hidden.dtype,
        )
        if current_latent is None or future_latent_target is None:
            return {
                "supervision_available": False,
                "loss": None,
                "losses": {},
            }

        latent_prior_dtype = next(self.mowa_future_latent_prior.parameters()).dtype
        loss, losses, output = self.mowa_future_latent_prior.compute_loss(
            current_latent.to(dtype=latent_prior_dtype),
            pooled_text_hidden.to(dtype=latent_prior_dtype),
            future_latent_target.to(dtype=latent_prior_dtype),
        )
        return {
            "supervision_available": True,
            "loss": loss,
            "losses": losses,
            "future_latent_mse": losses.get("future_latent_mse"),
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

        conditioned_layers = list(vl_embs_list)
        last_hidden = conditioned_layers[-1]
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
        conditioned_layers[-1] = torch.cat(
            (
                output.gated_condition_tokens.to(dtype=last_hidden.dtype),
                last_hidden[:, token_count:, :],
            ),
            dim=1,
        )
        return conditioned_layers, {
            "conditioned": True,
            "history_latent_sequence_shape": tuple(history_latent.shape),
            "compressed_history_shape": tuple(output.compressed_history.shape),
            "gate_values_shape": tuple(output.gate_values.shape),
            "gated_condition_tokens_shape": tuple(output.gated_condition_tokens.shape),
            "history_sequence_policy": "per_step_history_sequence_from_cache",
        }

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
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
            vl_embs_list = list(self._all_hidden_states)
            vl_embs_list = [self.wm_projector(h) for h in vl_embs_list]
            base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            vl_embs_list, mowa_hlc_gci = self._maybe_apply_mowa_hlc_gci_conditioning(vl_embs_list, examples)
            base_hidden = vl_embs_list[-1]
            mowa_future_latent_prior = self._maybe_run_mowa_future_latent_prior_loss(
                pooled_text_hidden,
                examples,
            )

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
                if self.config and hasattr(self.config, "framework")
                else 2
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)

        output = {"action_loss": action_loss}
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
                output["mowa_hlc_gci_history_sequence_policy"] = mowa_hlc_gci["history_sequence_policy"]
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(self._all_hidden_states)
            vl_embs_list = [self.wm_projector(h) for h in vl_embs_list]

        state = (
            torch.from_numpy(np.array(state)).to(vl_embs_list[-1].device, dtype=vl_embs_list[-1].dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            vl_embs_list, mowa_hlc_gci = self._maybe_apply_mowa_hlc_gci_conditioning(vl_embs_list, examples)
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions}
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
