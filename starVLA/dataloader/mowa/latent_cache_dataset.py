"""MoWA latent cache read-only dataset wrapper.

This module provides a unified entry point for MoWA latent caches.  It keeps the
legacy window-level ``.pt``/JSON artifact backend working unchanged and adds a
backend for the episode-level store + window manifest design (E-003/E-004).

Public API
----------
``MoWALatentCacheDataset(cache_root, manifest_path=None)`` auto-detects the
cache layout and exposes:

- ``sample_keys``: tuple of ``(episode_index, anchor_index, anchor_video_key)``.
- ``get_sample(episode_index=..., anchor_index=..., video_key=...) -> dict``
  with keys ``current_latent``, ``future_latent``, ``history_latent``,
  ``metadata``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from starVLA.dataloader.mowa.episode_latent_store import MoWAEpisodeLatentStore
from starVLA.dataloader.mowa.latent_cache_builder import (
    MoWALatentCacheDataset as _LegacyMoWALatentCacheDataset,
)
from starVLA.dataloader.mowa.window_latent_sample import (
    MoWAWindowLatentSampleDataset,
    WindowLatentSample,
)

__all__ = [
    "MoWALatentCacheDataset",
    "MoWAWindowLatentSampleDataset",
    "WindowLatentSample",
]


def _episode_index_from_id(episode_id: str) -> int:
    """Parse the numeric index from an episode id such as ``ep_000123``."""
    try:
        return int(episode_id.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unable to parse episode index from {episode_id!r}") from exc


class _MoWAEpisodeLevelLatentCacheDataset:
    """Backend that serves window samples from episode-level latent stores.

    This backend is used when ``cache_root`` contains ``ep_*.h5`` stores and a
    ``window_manifest*.parquet`` (or when a parquet manifest is passed explicitly).
    It exposes the same ``sample_keys`` / ``get_sample`` API as the legacy
    window-level artifact loader so that ``gr00t_lerobot/datasets.py`` does not
    need to change.
    """

    def __init__(
        self,
        cache_root: Path | str,
        manifest_path: Path | str | None = None,
        instruction_text_latent: Path | str | Any | None = None,
        history_steps: int | None = None,
        future_steps: int | None = None,
        action_chunk_steps: int | None = None,
        video_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self._resolve_manifest_path()
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"MoWA window manifest not found: {self.manifest_path}"
            )

        self._window_dataset = MoWAWindowLatentSampleDataset(
            self.manifest_path,
            instruction_text_latent=instruction_text_latent,
            history_steps=history_steps,
            future_steps=future_steps,
            action_chunk_steps=action_chunk_steps,
            episode_latent_path_prefix=self.cache_root,
            video_keys=video_keys,
        )
        self._sample_keys = tuple(sorted(set(self._window_dataset.sample_keys())))
        self._key_to_index = {
            key: idx for idx, key in enumerate(self._sample_keys)
        }

    def _resolve_manifest_path(self) -> Path:
        manifests = self._candidate_manifest_paths()
        if not manifests:
            raise FileNotFoundError(
                f"No window_manifest*.parquet found near episode cache root: {self.cache_root}"
            )
        return manifests[0]

    def _candidate_manifest_paths(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        candidates.extend(sorted(self.cache_root.glob("window_manifest*.parquet")))
        sibling_dirs = (
            self.cache_root.parent / f"{self.cache_root.name}_manifests",
            self.cache_root.parent / "window_manifests",
        )
        for sibling_dir in sibling_dirs:
            candidates.extend(sorted(sibling_dir.glob("window_manifest*.parquet")))
        return tuple(candidates)

    def __len__(self) -> int:
        return len(self._sample_keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, anchor_index, video_key = self._sample_keys[index]
        return self.get_sample(
            episode_index=episode_index,
            anchor_index=anchor_index,
            video_key=video_key,
        )

    @property
    def sample_keys(self) -> tuple[tuple[int, int, str], ...]:
        return self._sample_keys

    def get_sample(
        self,
        *,
        episode_index: int,
        anchor_index: int,
        video_key: str,
    ) -> dict[str, Any]:
        key = (episode_index, anchor_index, video_key)
        if key not in self._key_to_index:
            raise KeyError(
                f"No cache sample for episode={episode_index}, "
                f"anchor={anchor_index}, video_key={video_key!r}."
            )

        sample_index = self._key_to_index[key]
        sample = self._window_dataset[sample_index]
        entry = self._window_dataset.get_entry(sample_index)

        current_latent = sample.current_latent
        # The window assembler already squeezes a single-frame current latent.
        # Future latents may contain multiple frames; mean-pool to a single
        # vector to match the legacy model contract ``[D]``.
        future_latents = sample.future_latents
        future_latent = _mean_pool_latent(future_latents)

        # History latent keeps the time dimension; pool each frame separately.
        history_latent = _mean_pool_latent_per_frame(sample.history_latents)

        # Pool the current-frame latent to a 1-D vector for MoWA auxiliary heads
        # (future latent prior, HLC-GCI) which expect [B, D] or [B, H, D].
        mowa_current_latent = _mean_pool_latent(current_latent)

        # WanPI cache path: pass the raw current-frame visual latent and cached
        # text embeddings so the world model can skip VAE + UMT5 loading.
        language = sample.language
        text_embeds = None
        text_attention_mask = None
        lang = None
        if isinstance(language, dict):
            text_embeds = language.get("text_embeds")
            text_attention_mask = language.get("attention_mask")
            lang = language.get("instruction")
        else:
            lang = language

        latents = {
            "current_latent": current_latent,
            "future_latent": future_latent,
            "mowa_future_latent_sequence": _mean_pool_latent_per_frame(future_latents),
            "history_latent": history_latent,
            "future_latents": future_latents,
            "history_latents": sample.history_latents,
            "mowa_current_latent": mowa_current_latent,
            "history_valid_mask": sample.history_valid_mask,
            "future_valid_mask": sample.future_valid_mask,
            "action_valid_mask": sample.action_valid_mask,
            "history_action_valid_mask": sample.history_action_valid_mask,
        }
        store_attrs = MoWAEpisodeLatentStore(entry.episode_latent_path).attrs
        metadata = {
            **sample.metadata,
            "episode_id": sample.episode_id,
            "sample_id": sample.sample_id,
            "episode_latent_path": entry.episode_latent_path,
            "latent_type": store_attrs.get("latent_type", "unknown"),
            "vae_latents_normalized": bool(store_attrs.get("vae_latents_normalized", False)),
        }
        result = {
            "episode_index": episode_index,
            "anchor_index": anchor_index,
            "anchor_video_key": entry.anchor_video_key,
            "current_latent": current_latent,
            "future_latent": future_latent,
            "history_latent": history_latent,
            "mowa_current_latent": mowa_current_latent,
            "latents": latents,
            "metadata": metadata,
            "visual_latent": current_latent,
            "visual_latents_normalized": metadata["vae_latents_normalized"],
            "mowa_history_valid_mask": sample.history_valid_mask,
            "mowa_future_valid_mask": sample.future_valid_mask,
            "mowa_action_valid_mask": sample.action_valid_mask,
            "mowa_history_action_valid_mask": sample.history_action_valid_mask,
            "mowa_future_done_target": sample.future_done_target,
            "mowa_multi_view_video_keys": sample.multi_view_video_keys,
            "mowa_multi_view_history_latents": sample.multi_view_history_latents,
            "mowa_multi_view_current_latents": sample.multi_view_current_latents,
            "mowa_multi_view_future_latents": sample.multi_view_future_latents,
            "mowa_current_robot_state": sample.current_robot_state,
            "mowa_future_robot_states": sample.future_robot_states,
            "mowa_history_state_valid_mask": sample.history_state_valid_mask,
            "mowa_future_state_valid_mask": sample.future_state_valid_mask,
            "lang": lang,
        }
        if text_embeds is not None:
            result["text_embeds"] = text_embeds
        if text_attention_mask is not None:
            result["text_attention_mask"] = text_attention_mask
        return result


class MoWALatentCacheDataset:
    """Unified read-only cache loader that auto-detects the cache backend.

    The backend is selected as follows:

    1. If ``manifest_path`` is a ``.parquet`` file (or its name starts with
       ``window_manifest``), use the episode-level backend.
    2. Otherwise, if ``cache_root`` contains ``ep_*.h5`` stores **and**
       ``window_manifest*.parquet``, use the episode-level backend.
    3. Otherwise fall back to the legacy ``.pt``/JSON window-level artifact
       backend.
    """

    def __init__(
        self,
        cache_root: Path | str,
        manifest_path: Path | str | None = None,
        instruction_text_latent: Path | str | Any | None = None,
        history_steps: int | None = None,
        future_steps: int | None = None,
        action_chunk_steps: int | None = None,
        video_keys: tuple[str, ...] | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else None
        )
        self.instruction_text_latent = instruction_text_latent

        if self._is_episode_level():
            self._backend: _LegacyMoWALatentCacheDataset | _MoWAEpisodeLevelLatentCacheDataset = (
                _MoWAEpisodeLevelLatentCacheDataset(
                    self.cache_root,
                    self.manifest_path,
                    instruction_text_latent=instruction_text_latent,
                    history_steps=history_steps,
                    future_steps=future_steps,
                    action_chunk_steps=action_chunk_steps,
                    video_keys=video_keys,
                )
            )
        else:
            legacy_manifest = (
                str(self.manifest_path) if self.manifest_path is not None else None
            )
            self._backend = _LegacyMoWALatentCacheDataset(
                self.cache_root, legacy_manifest
            )

        # Normalize the manifest path to the backend's resolved value.
        self.manifest_path = self._backend.manifest_path

    def _is_episode_level(self) -> bool:
        if self.manifest_path is not None:
            if self.manifest_path.suffix == ".parquet":
                return True
            if (
                self.manifest_path.name.startswith("window_manifest")
                and self.manifest_path.is_file()
            ):
                return True
        return bool(list(self.cache_root.glob("ep_*.h5"))) and bool(
            self._candidate_manifest_paths()
        )

    def _candidate_manifest_paths(self) -> tuple[Path, ...]:
        candidates: list[Path] = []
        candidates.extend(sorted(self.cache_root.glob("window_manifest*.parquet")))
        sibling_dirs = (
            self.cache_root.parent / f"{self.cache_root.name}_manifests",
            self.cache_root.parent / "window_manifests",
        )
        for sibling_dir in sibling_dirs:
            candidates.extend(sorted(sibling_dir.glob("window_manifest*.parquet")))
        return tuple(candidates)

    def __len__(self) -> int:
        return len(self._backend)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._backend[index]

    @property
    def sample_keys(self) -> tuple[tuple[int, int, str], ...]:
        return self._backend.sample_keys

    def get_sample(
        self,
        *,
        episode_index: int,
        anchor_index: int,
        video_key: str,
    ) -> dict[str, Any]:
        return self._backend.get_sample(
            episode_index=episode_index,
            anchor_index=anchor_index,
            video_key=video_key,
        )


def _entry_belongs_to_cache_root(episode_latent_path: str, cache_root: Path) -> bool:
    """Scope a global manifest to the dataset-specific cache root of this loader."""
    try:
        Path(episode_latent_path).resolve().relative_to(cache_root.resolve())
    except ValueError:
        return False
    return True


def _mean_pool_latent(latent: torch.Tensor) -> torch.Tensor:
    """Collapse a multi-frame / spatial latent to a 1-D vector per frame group.

    Supports pooled vectors and VAE spatial latents.  The result is a flat
    1-D tensor compatible with the legacy QwenPI / StarFlow model contract.
    """
    if latent.ndim == 1:
        return latent
    if latent.ndim == 2:
        # [T, D] -> [D]
        return latent.mean(dim=0)
    if latent.ndim == 3:
        # [C, h, w] -> [C]
        return latent.mean(dim=(1, 2))
    if latent.ndim == 4:
        # [T, C, h, w] -> [C]
        return latent.mean(dim=(0, 2, 3))
    raise ValueError(
        f"Unsupported latent ndim={latent.ndim}, shape={tuple(latent.shape)}"
    )


def _mean_pool_latent_per_frame(latent: torch.Tensor) -> torch.Tensor:
    """Pool each frame of a history / future latent sequence independently.

    Supports [H, D] pooled history and [H, C, h, w] spatial history.
    """
    if latent.ndim == 2:
        return latent
    if latent.ndim == 4:
        # [H, C, h, w] -> [H, C]
        return latent.mean(dim=(2, 3))
    raise ValueError(
        f"Unsupported history latent ndim={latent.ndim}, shape={tuple(latent.shape)}"
    )
