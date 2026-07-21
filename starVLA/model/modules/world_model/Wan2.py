# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Wan2.2-TI2V World Model Interface.

Wraps Wan-AI/Wan2.2-TI2V-5B-Diffusers (diffusion-based Text+Image-to-Video model)
as a world-model backend for starVLA action prediction frameworks.

Architecture (diffusers format):
  - UMT5EncoderModel: text instruction → text embeddings [B, L_text, 4096]
  - AutoencoderKLWan (VAE): observation image → video latents [B, 48, T, H/16, W/16]
  - WanTransformer3DModel: 30-layer DiT, hidden_dim=3072 (24 heads × 128 dim)
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Note: The diffusers version of Wan2.2-TI2V-5B uses WanPipeline (text-only
conditioning) with expand_timesteps=True for TI2V mode. There is NO CLIP
image_encoder in this model variant — image conditioning is achieved through
per-token timestep expansion where the first frame's latent is conditioned
via timestep=0 (clean).

Key differences from CosmoPredict2:
  - Text encoder: UMT5 (dim=4096) vs T5 (dim=1024)
  - VAE latent channels: 48 vs 16
  - DiT hidden dim: 3072 (24×128) vs 2048 (16×128)
  - Scheduler: UniPCMultistepScheduler vs FlowMatchEulerDiscreteScheduler
  - No condition_mask / padding_mask (those are Cosmos-specific)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.modules.world_model.wan_vae_utils import (
    encode_wan_vae_video,
    prepare_wan_vae_video_tensor,
    wan_padded_frame_count,
)
from starVLA.model.modules.lora_utils import load_pretrained_lora_compatible

logger = initialize_overwatch(__name__)


_WAN_LORA_TARGET_GROUPS = {
    "cross_attention": ".attn2.",
    "self_attention": ".attn1.",
    "mlp": ".ffn.",
}


def resolve_wan_lora_target_modules(
    transformer: nn.Module,
    *,
    target_groups: list[str],
    start_layer: int = 0,
    end_layer: int | None = None,
) -> list[str]:
    """解析 Wan DiT 中指定层范围内可注入 LoRA 的线性投影。"""
    unknown_groups = sorted(set(target_groups).difference(_WAN_LORA_TARGET_GROUPS))
    if unknown_groups:
        raise ValueError(
            "Wan LoRA target_groups only supports "
            f"{sorted(_WAN_LORA_TARGET_GROUPS)}, got {unknown_groups}."
        )
    if not target_groups:
        raise ValueError("Wan LoRA target_groups must not be empty when LoRA is enabled.")
    if start_layer < 0 or (end_layer is not None and end_layer < start_layer):
        raise ValueError(
            f"Invalid Wan LoRA layer range: start_layer={start_layer}, end_layer={end_layer}."
        )

    target_markers = tuple(_WAN_LORA_TARGET_GROUPS[group] for group in target_groups)
    target_modules = []
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear) or not name.startswith("blocks."):
            continue
        parts = name.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        layer_index = int(parts[1])
        if layer_index < start_layer or (end_layer is not None and layer_index > end_layer):
            continue
        if any(marker in f".{name}" for marker in target_markers):
            target_modules.append(name)

    if not target_modules:
        raise ValueError(
            "Wan LoRA target selection matched no linear modules: "
            f"target_groups={target_groups}, start_layer={start_layer}, end_layer={end_layer}."
        )
    return target_modules


class _Wan2_Interface(nn.Module):
    """
    World model wrapper for Wan2.2-TI2V-5B-Diffusers.

    The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      collected into a [B, N_tokens, hidden_dim] tensor that the action
      head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        )
        self.model_name = model_name
        self.config = config
        self.use_text_cache = bool(wm_cfg.get("use_text_cache", False))
        self.use_visual_cache = bool(wm_cfg.get("use_visual_cache", False))

        from diffusers import (
            AutoencoderKLWan,
            UniPCMultistepScheduler,
            WanTransformer3DModel,
        )
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(
            f"Loading Wan2.2-TI2V from {model_name} "
            f"(use_text_cache={self.use_text_cache}, use_visual_cache={self.use_visual_cache})"
        )

        # --- Text encoder: UMT5-XXL ---
        # When use_text_cache=True we skip loading the 4.7B text encoder to save VRAM.
        # Cached text_embeds/attention_mask must be supplied to build_inputs().
        self.tokenizer = None
        self.text_encoder = None
        if not self.use_text_cache:
            self.tokenizer = T5TokenizerFast.from_pretrained(
                model_name, subfolder="tokenizer"
            )
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
            )

        # --- DiT transformer ---
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self._configure_lora(wm_cfg.get("lora", {}))
        if bool(wm_cfg.get("gradient_checkpointing", False)):
            self.transformer.enable_gradient_checkpointing()
            logger.info("Wan transformer gradient checkpointing enabled.")

        # --- VAE (image → latents for DiT input, z_dim=48) ---
        # Read VAE config to get scale factors without loading weights when visual cache is used.
        self.vae = None
        self._load_vae_config(model_name)
        if not self.use_visual_cache:
            self._load_vae_weights()

        # --- Scheduler ---
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Freeze VAE and text encoder by default when they are loaded
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.vae is not None:
            self.vae.requires_grad_(False)

        # DiT: 24 heads × 128 dim = 3072
        self._hidden_size = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

        # Config-like shim for framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

    def _configure_lora(self, lora_cfg) -> None:
        """按配置向 Wan DiT 注入 Diffusers/PEFT LoRA 适配器。"""
        self.lora_enabled = bool(lora_cfg.get("enabled", False))
        self.lora_adapter_name = None
        self.lora_target_modules = ()
        if not self.lora_enabled:
            return

        try:
            from peft import LoraConfig
        except ImportError as error:
            raise ImportError(
                "Wan LoRA is enabled but `peft` is unavailable. Install `peft>=0.17.0` in this environment."
            ) from error

        rank = int(lora_cfg.get("rank", 8))
        alpha = int(lora_cfg.get("alpha", 16))
        dropout = float(lora_cfg.get("dropout", 0.0))
        target_groups = list(lora_cfg.get("target_groups", ["cross_attention", "self_attention"]))
        start_layer = int(lora_cfg.get("start_layer", 0))
        raw_end_layer = lora_cfg.get("end_layer", None)
        end_layer = None if raw_end_layer is None else int(raw_end_layer)
        if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"Invalid Wan LoRA settings: rank={rank}, alpha={alpha}, dropout={dropout}."
            )

        target_modules = resolve_wan_lora_target_modules(
            self.transformer,
            target_groups=target_groups,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.lora_adapter_name = "mowa_wan"
        self.transformer.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
            ),
            adapter_name=self.lora_adapter_name,
        )
        self.transformer.set_adapter(self.lora_adapter_name)
        self.lora_target_modules = tuple(target_modules)
        lora_param_count = sum(
            parameter.numel()
            for name, parameter in self.transformer.named_parameters()
            if "lora_" in name
        )
        logger.info(
            "Wan LoRA enabled: adapter=%s, groups=%s, layers=%s:%s, rank=%d, alpha=%d, "
            "dropout=%.3f, targets=%d, params=%d",
            self.lora_adapter_name,
            target_groups,
            start_layer,
            end_layer if end_layer is not None else "last",
            rank,
            alpha,
            dropout,
            len(target_modules),
            lora_param_count,
        )

    def _load_vae_weights(self) -> None:
        """按需加载 Wan VAE，避免 visual cache 训练路径占用额外显存。"""
        if self.vae is not None:
            return
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.vae.requires_grad_(False)

    def ensure_vae_for_inference(self) -> None:
        """为在线原始相机帧推理惰性加载 VAE 并迁移到 Wan transformer 设备。"""
        self._load_vae_weights()
        device = next(self.transformer.parameters()).device
        self.vae.to(device=device)
        self.vae.eval()

    def load_pretrained_state_dict(self, state_dict):
        """兼容 LoRA 注入前导出的 Wan backbone checkpoint。"""
        return load_pretrained_lora_compatible(
            self,
            state_dict,
            optional_prefixes=("text_encoder.", "vae."),
        )

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.backbone.model.config.hidden_size"""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _load_vae_config(self, model_name: str) -> None:
        """Load VAE scale factors and normalization constants from config.

        This lets the world model consume pre-computed visual latents without
        loading the full VAE weights into GPU memory.
        """
        import json

        config_path = Path(model_name) / "vae" / "config.json"
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as file:
                vae_config = json.load(file)
            temperal_downsample = vae_config.get("temperal_downsample", [False, True, True])
            self._vae_latents_mean = vae_config.get("latents_mean")
            self._vae_latents_std = vae_config.get("latents_std")
            self._vae_z_dim = vae_config.get("z_dim", 48)
        else:
            temperal_downsample = [False, True, True]
            self._vae_latents_mean = None
            self._vae_latents_std = None
            self._vae_z_dim = 48
        self.vae_scale_factor_spatial = 2 ** len(temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(int(x) for x in temperal_downsample)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using UMT5.

        Returns:
            Tuple of (text_embeds [B, L, 4096], attention_mask [B, L]).
        """
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 4096]

        return text_embeds.to(dtype=torch.bfloat16), text_inputs.attention_mask.to(dtype=torch.bfloat16)

    def _encode_images_vae(self, images, num_frames=None):
        """Encode observation images through VAE to get latent tokens.

        Two-pass approach (same as CosmoPredict2):
          Pass 1: preprocess each sample, record real frame counts.
          Determine target_frames = num_frames if given, else batch max.
          Pass 2: truncate or pad each sample to target_frames.

        VAE config: z_dim=48, scale_factor_spatial=16, scale_factor_temporal=4
        T_latent = (target_frames - 1) // 4 + 1

        Args:
            images: List of List of PIL Images [B, [imgs...]]
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.

        Returns:
            latents: [B, 48, T_latent, H/16, W/16] video latent tensor
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        # Pass 1: preprocess each sample, record real frame counts
        preprocessed = []
        frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = prepare_wan_vae_video_tensor(self.video_processor, list(sample_imgs))
            video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        target_frames = wan_padded_frame_count(
            num_frames if num_frames is not None else max(frame_counts)
        )

        # Pass 2: truncate or pad each sample to target_frames
        batch_videos = []
        for video_tensor in preprocessed:
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
            elif n < target_frames:
                # Pad with last-frame repetition (matches official Wan pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        return encode_wan_vae_video(self.vae, video)

    def _normalize_cached_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the same normalization used by the official Wan pipeline.

        Cached visual latents from legacy stores are raw VAE samples; the DiT
        expects normalized latents. New temporal stores are marked normalized
        and bypass this compatibility path.
        """
        if self._vae_latents_mean is None or self._vae_latents_std is None:
            raise RuntimeError(
                "VAE latents_mean/latents_std not available; cannot normalize cached latents."
            )
        latents_mean = (
            torch.tensor(self._vae_latents_mean)
            .view(1, self._vae_z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self._vae_latents_std)
            .view(1, self._vae_z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        return (latents - latents_mean) * latents_std

    def build_inputs(self, images=None, instructions=None, **kwargs):
        """Build inputs for the Wan DiT world model.

        Supports two modes:

        1. Raw mode (default): ``images`` and ``instructions`` are provided; the
           world model runs UMT5 + VAE encoding internally.
        2. Cache mode: ``text_embeds`` / ``text_attention_mask`` and/or
           ``visual_latents`` are provided. When the corresponding
           ``use_*_cache`` flag is enabled, the heavy encoders are skipped.

        The cached ``visual_latents`` should have shape ``[B, C, T, H, W]`` or
        ``[B, C, H, W]`` (the latter is treated as ``T=1``).
        """
        device = next(self.transformer.parameters()).device

        # --- Text conditioning ---
        text_embeds = kwargs.get("text_embeds")
        text_attention_mask = kwargs.get("text_attention_mask")
        if text_embeds is not None and text_attention_mask is not None:
            text_embeds = text_embeds.to(device=device, dtype=torch.bfloat16)
            text_attention_mask = text_attention_mask.to(device=device, dtype=torch.bfloat16)
        elif instructions is not None:
            if self.use_text_cache:
                raise ValueError(
                    "use_text_cache=True but text_embeds/text_attention_mask not provided."
                )
            text_embeds, text_attention_mask = self._encode_text(instructions)
        else:
            raise ValueError("Either instructions or text_embeds/text_attention_mask must be provided.")

        # --- Visual conditioning ---
        visual_latents = kwargs.get("visual_latents")
        if visual_latents is not None:
            latents = torch.as_tensor(visual_latents, device=device, dtype=torch.bfloat16)
            if latents.dim() == 4:
                # [B, C, H, W] -> [B, C, 1, H, W]
                latents = latents.unsqueeze(2)
            if latents.dim() != 5:
                raise ValueError(
                    f"visual_latents must be 4D or 5D, got {latents.dim()}D with shape {tuple(latents.shape)}"
                )
            if not kwargs.get("visual_latents_normalized", False):
                latents = self._normalize_cached_vae_latents(latents)
        elif images is not None:
            if self.use_visual_cache:
                raise ValueError(
                    "use_visual_cache=True but visual_latents not provided."
                )
            latents = self._encode_images_vae(images)
        else:
            raise ValueError("Either images or visual_latents must be provided.")

        batch_size = latents.shape[0]

        # Wan2.2 TI2V uses expand_timesteps: timestep is per-token
        # Shape: [B, seq_len] where seq_len = T_lat * (H_lat//p_h) * (W_lat//p_w)
        # For feature extraction at σ≈0, use zeros (clean input)
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T, H, W = latents.shape
        patch_grid = (T // p_t, H // p_h, W // p_w)
        seq_len = patch_grid[0] * patch_grid[1] * patch_grid[2]
        rope_max_seq_len = int(self.transformer.config.rope_max_seq_len)
        if any(size > rope_max_seq_len for size in patch_grid):
            raise ValueError(
                f"Wan patch grid {patch_grid} exceeds per-axis rope_max_seq_len={rope_max_seq_len}."
            )
        timestep = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "encoder_attention_mask": text_attention_mask,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the Wan DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        kwargs.pop("_is_wm_input", False) # pop internal routing flags from kwargs to avoid passing them downstream
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)
        kwargs.pop("encoder_attention_mask", None)  # not consumed by WanTransformer3DModel forward

        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
            )

        # Collect features from hooks
        # WanTransformer3DModel blocks output [B, seq_len, hidden_dim] (already flattened)
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # Fallback: use transformer output directly
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, sample=None, loss=None):
                self.hidden_states = hidden_states_tuple
                self.sample = sample
                self.loss = loss # TODO if you want to add loss for image reconstruction or other auxiliary objectives, you can include it here and return it in the forward pass

        return _WMOutput(hidden_states_tuple=tuple(extracted), sample=dit_output.sample)

    def generate(self, **kwargs):
        """Video generation using the WanPipeline.

        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import WanPipeline

        pipe = WanPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )
        return pipe(**kwargs)
