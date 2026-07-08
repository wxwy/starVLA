"""MoWA latent cache read-only dataset wrapper.

This module provides a unified entry point for MoWA latent caches.  It keeps the
legacy window-level ``.pt``/JSON artifact backend working unchanged and adds a
backend for the episode-level store + window manifest design (E-003/E-004).

Public API
----------
``MoWALatentCacheDataset(cache_root, manifest_path=None)`` auto-detects the
cache layout and exposes:

- ``sample_keys``: tuple of ``(episode_index, anchor_index, video_key)``.
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

        self._window_dataset = MoWAWindowLatentSampleDataset(self.manifest_path)
        self._sample_keys = tuple(
            sorted(
                {
                    (
                        _episode_index_from_id(entry.episode_id),
                        entry.anchor_index,
                        entry.video_key,
                    )
                    for entry in self._window_dataset._entries
                }
            )
        )
        self._key_to_index = {
            key: idx for idx, key in enumerate(self._sample_keys)
        }
        self._checked_store_paths: set[str] = set()

    def _resolve_manifest_path(self) -> Path:
        manifests = sorted(self.cache_root.glob("window_manifest*.parquet"))
        if not manifests:
            raise FileNotFoundError(
                f"No window_manifest*.parquet found in episode cache root: {self.cache_root}"
            )
        return manifests[0]

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
        self._assert_pooled_vector(entry.episode_latent_path)

        current_latent = sample.current_latent
        # The window assembler already squeezes a single-frame current latent.
        # Future latents may contain multiple frames; mean-pool to a single
        # vector to match the legacy model contract ``[D]``.
        future_latents = sample.future_latents
        if future_latents.ndim == 2:
            future_latent = future_latents.mean(dim=0)
        elif future_latents.ndim == 1:
            future_latent = future_latents
        else:
            raise ValueError(
                f"Unexpected future_latents ndim={future_latents.ndim}, "
                f"shape={tuple(future_latents.shape)}"
            )

        history_latent = sample.history_latents
        latents = {
            "current_latent": current_latent,
            "future_latent": future_latent,
            "history_latent": history_latent,
        }
        metadata = {
            **sample.metadata,
            "episode_id": sample.episode_id,
            "sample_id": sample.sample_id,
            "episode_latent_path": entry.episode_latent_path,
        }
        return {
            "episode_index": episode_index,
            "anchor_index": anchor_index,
            "video_key": video_key,
            "current_latent": current_latent,
            "future_latent": future_latent,
            "history_latent": history_latent,
            "latents": latents,
            "metadata": metadata,
        }

    def _assert_pooled_vector(self, episode_latent_path: str) -> None:
        if episode_latent_path in self._checked_store_paths:
            return
        store = MoWAEpisodeLatentStore(episode_latent_path)
        latent_type = store.attrs.get("latent_type", "unknown")
        if latent_type != "pooled_vector":
            raise ValueError(
                f"Episode latent store {episode_latent_path} uses latent_type={latent_type!r}. "
                "The train_starvla data path only supports pooled_vector episode stores. "
                "Please rebuild the cache with latent_type='pooled_vector'."
            )
        self._checked_store_paths.add(episode_latent_path)


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
    ) -> None:
        self.cache_root = Path(cache_root)
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else None
        )

        if self._is_episode_level():
            self._backend: _LegacyMoWALatentCacheDataset | _MoWAEpisodeLevelLatentCacheDataset = (
                _MoWAEpisodeLevelLatentCacheDataset(
                    self.cache_root, self.manifest_path
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
            list(self.cache_root.glob("window_manifest*.parquet"))
        )

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
