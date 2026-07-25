# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanOFT Framework — Wan2.2-TI2V World Model + MLP Regression for Action Prediction.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone with a lightweight MLP action head (L1 regression).

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D
    → hidden_states [B, N, 3072]
    → Global avg pool → [B, 3072]
    → Linear projection → [B, chunk_len, 3072]
    → MLP (L1 regression) → action predictions [B, chunk_len, action_dim]

Key differences from WanGR00T:
  - Action head: MLP L1 regression (not flow-matching diffusion)
  - No repeated_diffusion_steps (single forward pass)
  - Faster training & inference
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
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanOFTDefaultConfig:
    """WanOFT default parameters."""

    name: str = "WanOFT"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
            # WanOFT 训练/官方评测使用的 VAE 输入格式：480x832、不按 4n+1
            # 补帧、latent_dist.sample()。MoWA/WanPI 路径不使用该标志。
            "legacy_vae_input": True,
            # 无持久化表时，可由训练入口一次性预编码全部去重指令后释放 UMT5。
            "preload_text_cache": False,
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )

    # === Action head (MLP L1 regression) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "action_hidden_dim": 3072,
            "future_action_window_size": 8,
            "past_action_window_size": 0,
        }
    )
    multi_view_residual: dict = field(
        default_factory=lambda: {
            "enabled": False,
        }
    )


@FRAMEWORK_REGISTRY.register("WanOFT")
class Wan_OFT(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V + MLP regression.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Adaptive pooling + MLP regression head (L1 loss)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanOFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = wm_hidden

        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon

        self.action_query_proj = nn.Linear(wm_hidden, self.chunk_len * wm_hidden)  # Project into a two-layer MLP
        self.multi_view_residual_enabled = bool(
            self.config.framework.multi_view_residual.enabled
        )
        self.multi_view_fusion = None
        if self.multi_view_residual_enabled:
            self.multi_view_fusion = nn.Linear(2 * wm_hidden, wm_hidden)
            self._reset_multi_view_fusion()

        self._mowa_instruction_text_cache = None
        self.l1_loss = nn.L1Loss()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """允许旧 WanOFT checkpoint 缺失零初始化的多视角融合层。"""
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        allowed_missing = (
            {"multi_view_fusion.weight", "multi_view_fusion.bias"}
            if self.multi_view_residual_enabled
            else set()
        )
        unexpected_missing = [key for key in result.missing_keys if key not in allowed_missing]
        if strict and (unexpected_missing or result.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict: "
                f"missing={unexpected_missing}, unexpected={result.unexpected_keys}."
            )
        if result.missing_keys and not unexpected_missing:
            logger.info(
                "[WanOFT multi-view residual] checkpoint lacks fusion parameters; "
                "using [I,0] initialization."
            )
        return result

    def _reset_multi_view_fusion(self) -> None:
        """初始化为 fused=原生 [main,wrist] 联合编码特征。"""
        if self.multi_view_fusion is None:
            return
        hidden_size = self.multi_view_fusion.out_features
        with torch.no_grad():
            self.multi_view_fusion.weight.zero_()
            self.multi_view_fusion.weight[:, :hidden_size].copy_(
                torch.eye(hidden_size, dtype=self.multi_view_fusion.weight.dtype)
            )
            self.multi_view_fusion.bias.zero_()

    def _pool_action_feature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        if pooled.dim() != 2:
            raise ValueError(f"WanOFT pooled feature must be [B,H], got {tuple(pooled.shape)}.")
        return pooled

    def _fuse_action_features(
        self,
        base_feature: torch.Tensor,
        wrist_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.multi_view_residual_enabled:
            return base_feature
        if wrist_feature is None or self.multi_view_fusion is None:
            raise ValueError("WanOFT multi-view residual requires an independently encoded wrist feature.")
        if wrist_feature.shape != base_feature.shape:
            raise ValueError(
                "WanOFT base/wrist feature shape mismatch: "
                f"{tuple(base_feature.shape)} vs {tuple(wrist_feature.shape)}."
            )
        fusion_dtype = self.multi_view_fusion.weight.dtype
        return self.multi_view_fusion(
            torch.cat([base_feature, wrist_feature], dim=-1).to(dtype=fusion_dtype)
        )

    @staticmethod
    def _wrist_only_images(batch_images: List) -> List[List]:
        wrist_images = []
        for sample_index, images in enumerate(batch_images):
            if not isinstance(images, (list, tuple)) or len(images) != 2:
                raise ValueError(
                    "WanOFT multi-view residual requires image=[main,wrist] for every sample; "
                    f"sample {sample_index} has {type(images).__name__}."
                )
            wrist_images.append([images[1]])
        return wrist_images

    def _extract_base_and_wrist_features(
        self,
        batch_images: List,
        instructions: List[str],
        text_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 训练 batch 可能是 ndarray，服务推理已转为 PIL；统一在共享视觉入口
        # 规范化，避免同一 RGB 因输入容器类型不同而得到不同 VAE 特征。
        batch_images = [to_pil_preserve(images) for images in batch_images]
        text_kwargs = text_kwargs or {}
        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            **text_kwargs,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            base_feature = self._pool_action_feature(wm_outputs.hidden_states[-1])
            if not self.multi_view_residual_enabled:
                return base_feature, None

            wrist_inputs = self.backbone.build_inputs(
                images=self._wrist_only_images(batch_images),
                instructions=instructions,
                **text_kwargs,
            )
            wrist_outputs = self.backbone(
                **wrist_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            wrist_feature = self._pool_action_feature(wrist_outputs.hidden_states[-1])
        return base_feature, wrist_feature

    @staticmethod
    def _collect_cached_text_inputs(examples: List[dict]) -> dict[str, torch.Tensor]:
        """收集 batch 内一致的 UMT5 缓存，缺失时回退到原始文本路径。"""
        text_embeds = [example.get("text_embeds") for example in examples]
        text_masks = [example.get("text_attention_mask") for example in examples]
        has_embeds = [value is not None for value in text_embeds]
        has_masks = [value is not None for value in text_masks]
        if not any(has_embeds) and not any(has_masks):
            return {}
        if not all(has_embeds) or not all(has_masks):
            raise ValueError(
                "WanOFT batch must provide text_embeds/text_attention_mask for every sample or none."
            )

        embeds = torch.stack([torch.as_tensor(value) for value in text_embeds], dim=0)
        masks = torch.stack([torch.as_tensor(value) for value in text_masks], dim=0)
        if embeds.ndim == 4 and embeds.shape[1] == 1:
            embeds = embeds.squeeze(1)
        if masks.ndim == 3 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        if embeds.ndim != 3 or masks.ndim != 2 or embeds.shape[:2] != masks.shape:
            raise ValueError(
                "WanOFT cached text shape mismatch: "
                f"text_embeds={tuple(embeds.shape)}, text_attention_mask={tuple(masks.shape)}."
            )
        return {"text_embeds": embeds, "text_attention_mask": masks}

    def _populate_mowa_cached_text(self, examples: List[dict]) -> List[dict]:
        """在启用 UMT5 cache 时，按指令补齐训练和在线推理的文本条件。"""
        if not getattr(self.backbone, "use_text_cache", False):
            return examples
        if all(
            example.get("text_embeds") is not None
            and example.get("text_attention_mask") is not None
            for example in examples
        ):
            return examples

        if getattr(self.backbone, "_instruction_text_cache", None) is not None:
            return examples
        cache_path = getattr(getattr(self.config, "latent_cache", None), "instruction_text_latent", None)
        if not cache_path:
            raise ValueError(
                "WanOFT use_text_cache=True requires latent_cache.instruction_text_latent."
            )
        if self._mowa_instruction_text_cache is None:
            from starVLA.dataloader.mowa.instruction_text_latent_cache import MoWAInstructionTextLatentCache

            self._mowa_instruction_text_cache = MoWAInstructionTextLatentCache(Path(cache_path))
            logger.info(
                "[WanOFT] loaded UMT5 instruction cache=%s entries=%d",
                cache_path,
                len(self._mowa_instruction_text_cache),
            )

        resolved = []
        for index, example in enumerate(examples):
            instruction = str(example.get("lang", ""))
            entry = self._mowa_instruction_text_cache.lookup(instruction)
            if entry is None:
                raise KeyError(
                    f"WanOFT UMT5 cache misses instruction at batch index {index}: {instruction!r}."
                )
            resolved.append(
                {
                    **example,
                    "text_embeds": entry["text_embeds"],
                    "text_attention_mask": entry["attention_mask"],
                }
            )
        return resolved

    def _extract_action_feature(
        self,
        batch_images: List,
        instructions: List[str],
        text_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        base_feature, wrist_feature = self._extract_base_and_wrist_features(
            batch_images, instructions, text_kwargs=text_kwargs
        )
        if not self.multi_view_residual_enabled:
            return base_feature
        return self._fuse_action_features(base_feature, wrist_feature)

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, H = hidden_states.shape
        queries = self.action_query_proj(hidden_states)
        action_queries = queries.view(B, self.chunk_len, H)
        return action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        examples = self._populate_mowa_cached_text(examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        text_kwargs = self._collect_cached_text_inputs(examples)
        if state is not None and text_kwargs:
            raise ValueError("WanOFT cached UMT5 text is incompatible with discretized state instructions.")

        # Optionally prepend discretised proprioceptive state tokens (π₀.5 style).
        instructions = (
            add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        action_feature = self._extract_action_feature(batch_images, instructions, text_kwargs=text_kwargs)

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(action_feature)  # B, chunk_len, hidden_dim
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        examples = self._populate_mowa_cached_text(examples)
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        text_kwargs = self._collect_cached_text_inputs(examples)
        if state is not None and text_kwargs:
            raise ValueError("WanOFT cached UMT5 text is incompatible with discretized state instructions.")

        instructions = (
            add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_feature = self._extract_action_feature(batch_images, instructions, text_kwargs=text_kwargs)

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(action_feature)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


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

    cfg.framework.name = "WanOFT"
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    }

    model: Wan_OFT = Wan_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
