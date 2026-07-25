"""Shared Wan2.2 VAE preprocessing and deterministic encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    prepared = prepare_wan_vae_spatial_frames(frames)
    return prepared + [prepared[-1]] * (wan_padded_frame_count(len(prepared)) - len(prepared))


def prepare_wan_vae_spatial_frames(frames: list[Any]) -> list[Any]:
    """Apply the production 256x256 spatial policy without temporal padding."""
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

    return prepared


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


@dataclass
class _WanVAEStreamState:
    feature_cache: list[Any]
    pending_frames: list[Any] = field(default_factory=list)
    regular_latents: list[torch.Tensor] = field(default_factory=list)
    initialized: bool = False


class WanVAEStreamEncoder:
    """Stateful causal Wan VAE encoder matching whole-episode cache encoding.

    Each stream consumes one initialization frame followed by contiguous
    four-frame chunks. Encoder feature caches survive across calls; callers
    must reset the stream at every episode boundary.
    """

    def __init__(self, vae: Any, video_processor: Any) -> None:
        self.vae = vae
        self.video_processor = video_processor
        self._states: dict[str, _WanVAEStreamState] = {}

    def reset(self, stream_id: str) -> None:
        self._states.pop(str(stream_id), None)

    def clear(self) -> None:
        self._states.clear()

    def encode(
        self,
        stream_id: str,
        frames: list[Any],
        *,
        reset: bool = False,
        max_regular_latents: int = 1,
    ) -> tuple[torch.Tensor, ...]:
        """Consume new contiguous frames and return retained regular latents."""
        stream_id = str(stream_id)
        if not stream_id:
            raise ValueError("Wan VAE streaming requires a non-empty stream_id.")
        if reset:
            self.reset(stream_id)
        if max_regular_latents <= 0:
            raise ValueError(f"max_regular_latents must be positive, got {max_regular_latents}.")
        if not frames:
            state = self._states.get(stream_id)
            if state is None or not state.regular_latents:
                raise ValueError(f"Wan VAE stream {stream_id!r} has no frames or regular latent.")
            return tuple(state.regular_latents[-max_regular_latents:])

        state = self._states.get(stream_id)
        if state is None:
            encoder_conv_count = int(self.vae._cached_conv_counts["encoder"])
            state = _WanVAEStreamState(feature_cache=[None] * encoder_conv_count)
            self._states[stream_id] = state
        state.pending_frames.extend(frames)

        if not state.initialized:
            if not state.pending_frames:
                raise ValueError(f"Wan VAE stream {stream_id!r} has no initialization frame.")
            self._encode_chunk(state, state.pending_frames[:1])
            del state.pending_frames[:1]
            state.initialized = True

        while len(state.pending_frames) >= WAN_VAE_TEMPORAL_FACTOR:
            chunk = state.pending_frames[:WAN_VAE_TEMPORAL_FACTOR]
            del state.pending_frames[:WAN_VAE_TEMPORAL_FACTOR]
            latent = self._encode_chunk(state, chunk)
            state.regular_latents.append(latent)
            if len(state.regular_latents) > max_regular_latents:
                del state.regular_latents[:-max_regular_latents]

        if not state.regular_latents:
            return ()
        return tuple(state.regular_latents[-max_regular_latents:])

    def _encode_chunk(self, state: _WanVAEStreamState, frames: list[Any]) -> torch.Tensor:
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        prepared = prepare_wan_vae_spatial_frames(frames)
        video = self.video_processor.preprocess_video(
            prepared,
            height=WAN_VAE_INPUT_HEIGHT,
            width=WAN_VAE_INPUT_WIDTH,
        )
        device = next(self.vae.parameters()).device
        video = video.to(device=device, dtype=self.vae.dtype)
        if self.vae.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

            video = patchify(video, patch_size=self.vae.config.patch_size)
        if self.vae.use_tiling and (
            video.shape[-1] > self.vae.tile_sample_min_width
            or video.shape[-2] > self.vae.tile_sample_min_height
        ):
            raise RuntimeError("Stateful Wan VAE streaming does not support tiled encoding.")

        feature_index = [0]
        with torch.inference_mode():
            encoded = self.vae.encoder(
                video,
                feat_cache=state.feature_cache,
                feat_idx=feature_index,
            )
            posterior = DiagonalGaussianDistribution(self.vae.quant_conv(encoded))
            latent = posterior.mode()
        return normalize_wan_vae_latents(self.vae, latent)
