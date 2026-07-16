"""Shared Wan2.2 VAE preprocessing and deterministic encoding helpers."""

from __future__ import annotations

from typing import Any

import torch


WAN_VAE_INPUT_HEIGHT = 256
WAN_VAE_INPUT_WIDTH = 256
WAN_VAE_TEMPORAL_FACTOR = 4
WAN_VAE_PREPROCESS_POLICY = "center_crop_square_then_resize_256"


def wan_padded_frame_count(frame_count: int) -> int:
    """Return the smallest Wan-compatible temporal length (``4n + 1``)."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}.")
    return 1 + ((frame_count - 1 + WAN_VAE_TEMPORAL_FACTOR - 1) // WAN_VAE_TEMPORAL_FACTOR) * WAN_VAE_TEMPORAL_FACTOR


def prepare_wan_vae_frames(frames: list[Any]) -> list[Any]:
    """Center-crop non-square frames, resize to 256x256, then temporally pad."""
    if not frames:
        raise ValueError("Wan2.2 VAE requires at least one frame.")

    from PIL import Image

    prepared: list[Any] = []
    for frame in frames:
        image = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.fromarray(frame).convert("RGB")
        if image.width != image.height:
            side = min(image.width, image.height)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        if image.size != (WAN_VAE_INPUT_WIDTH, WAN_VAE_INPUT_HEIGHT):
            image = image.resize(
                (WAN_VAE_INPUT_WIDTH, WAN_VAE_INPUT_HEIGHT),
                Image.Resampling.LANCZOS,
            )
        prepared.append(image)

    return prepared + [prepared[-1]] * (wan_padded_frame_count(len(prepared)) - len(prepared))


def prepare_wan_vae_video_tensor(video_processor: Any, frames: list[Any]) -> torch.Tensor:
    """Return one preprocessed continuous Wan video with shape ``[1,C,T,256,256]``."""
    return video_processor.preprocess_video(
        prepare_wan_vae_frames(frames),
        height=WAN_VAE_INPUT_HEIGHT,
        width=WAN_VAE_INPUT_WIDTH,
    )


def normalize_wan_vae_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Apply Wan's channel-wise latent standardization exactly once."""
    mean = torch.as_tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    reciprocal_std = 1.0 / torch.as_tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    return (latents - mean) * reciprocal_std


def encode_wan_vae_video(vae: Any, video: torch.Tensor) -> torch.Tensor:
    """Deterministically encode and normalize a batch of continuous Wan videos."""
    with torch.inference_mode():
        latents = vae.encode(video).latent_dist.mode()
    return normalize_wan_vae_latents(vae, latents)
