"""MoWA latent cache manifest smoke utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.schema import DATA_GATE


@dataclass(frozen=True)
class MoWALatentCacheManifestEntry:
    episode_index: int
    video_key: str
    video_path: str
    video_exists: bool
    cache_key: str
    cache_status: str = DATA_GATE
    cache_relative_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "video_key": self.video_key,
            "video_path": self.video_path,
            "video_exists": self.video_exists,
            "cache_key": self.cache_key,
            "cache_status": self.cache_status,
            "cache_relative_path": self.cache_relative_path,
        }


@dataclass(frozen=True)
class MoWALatentCacheManifestSmoke:
    dataset_path: str
    sampled_episode_indices: tuple[int, ...]
    video_keys: tuple[str, ...]
    entries: tuple[MoWALatentCacheManifestEntry, ...]
    missing_video_count: int
    latent_shape_status: str = DATA_GATE
    cache_hash_status: str = DATA_GATE
    encoder_status: str = DATA_GATE
    notes: tuple[str, ...] = (
        "Manifest smoke only checks video paths and deterministic cache keys.",
        "No Wan encoder, VAE, latent tensor, cache file or training code is executed.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "sampled_episode_indices": self.sampled_episode_indices,
            "video_keys": self.video_keys,
            "entries": [entry.to_dict() for entry in self.entries],
            "missing_video_count": self.missing_video_count,
            "latent_shape_status": self.latent_shape_status,
            "cache_hash_status": self.cache_hash_status,
            "encoder_status": self.encoder_status,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWALatentCacheContractEntry:
    cache_key: str
    cache_path: str
    video_path: str
    video_exists: bool
    cache_exists: bool
    encoder_name: str
    cache_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "video_path": self.video_path,
            "video_exists": self.video_exists,
            "cache_exists": self.cache_exists,
            "encoder_name": self.encoder_name,
            "cache_status": self.cache_status,
        }


@dataclass(frozen=True)
class MoWALatentCacheContractSmoke:
    dataset_path: str
    cache_root: str
    encoder_name: str
    entries: tuple[MoWALatentCacheContractEntry, ...]
    missing_video_count: int
    missing_cache_count: int
    duplicate_cache_key_count: int
    latent_shape_status: str = DATA_GATE
    cache_artifact_status: str = DATA_GATE
    encoder_status: str = DATA_GATE
    future_action_input_status: str = "not_used_as_input"
    notes: tuple[str, ...] = (
        "Contract smoke plans deterministic latent cache artifact paths only.",
        "No Wan encoder, VAE, latent tensor, cache file or training code is executed.",
        "Future action remains target/invariant only and is not part of cache inputs.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "cache_root": self.cache_root,
            "encoder_name": self.encoder_name,
            "entries": [entry.to_dict() for entry in self.entries],
            "missing_video_count": self.missing_video_count,
            "missing_cache_count": self.missing_cache_count,
            "duplicate_cache_key_count": self.duplicate_cache_key_count,
            "latent_shape_status": self.latent_shape_status,
            "cache_artifact_status": self.cache_artifact_status,
            "encoder_status": self.encoder_status,
            "future_action_input_status": self.future_action_input_status,
            "notes": list(self.notes),
        }


def build_mowa_latent_cache_manifest_smoke(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
    video_keys: tuple[str, ...] = (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    ),
) -> MoWALatentCacheManifestSmoke:
    """生成只读 latent cache manifest smoke，不执行编码。"""

    root = Path(dataset_path)
    entries = []
    for episode_index in episode_indices:
        for video_key in video_keys:
            relative_path = (
                Path("videos")
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path = root / relative_path
            cache_key = _cache_key(root, episode_index, video_key)
            entries.append(
                MoWALatentCacheManifestEntry(
                    episode_index=episode_index,
                    video_key=video_key,
                    video_path=str(video_path),
                    video_exists=video_path.is_file(),
                    cache_key=cache_key,
                    cache_relative_path=f"{cache_key}.pt",
                )
            )

    return MoWALatentCacheManifestSmoke(
        dataset_path=str(root),
        sampled_episode_indices=episode_indices,
        video_keys=video_keys,
        entries=tuple(entries),
        missing_video_count=sum(1 for entry in entries if not entry.video_exists),
    )


def build_mowa_latent_cache_contract_smoke(
    dataset_path: Path | str,
    *,
    cache_root: Path | str,
    encoder_name: str = "wan2.2-ti2v-smoke-contract",
    episode_indices: tuple[int, ...] = (0, 1, 4),
    video_keys: tuple[str, ...] = (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    ),
) -> MoWALatentCacheContractSmoke:
    """Plan deterministic P1 latent cache artifact paths without encoding."""

    manifest = build_mowa_latent_cache_manifest_smoke(
        dataset_path,
        episode_indices=episode_indices,
        video_keys=video_keys,
    )
    cache_root_path = Path(cache_root)
    entries = []
    for entry in manifest.entries:
        cache_path = cache_root_path / f"{entry.cache_key}.pt"
        entries.append(
            MoWALatentCacheContractEntry(
                cache_key=entry.cache_key,
                cache_path=str(cache_path),
                video_path=entry.video_path,
                video_exists=entry.video_exists,
                cache_exists=cache_path.is_file(),
                encoder_name=encoder_name,
                cache_status="planned" if not cache_path.is_file() else "exists_unverified",
            )
        )

    cache_keys = [entry.cache_key for entry in entries]
    duplicate_count = len(cache_keys) - len(set(cache_keys))
    return MoWALatentCacheContractSmoke(
        dataset_path=manifest.dataset_path,
        cache_root=str(cache_root_path),
        encoder_name=encoder_name,
        entries=tuple(entries),
        missing_video_count=manifest.missing_video_count,
        missing_cache_count=sum(1 for entry in entries if not entry.cache_exists),
        duplicate_cache_key_count=duplicate_count,
    )


def _cache_key(dataset_path: Path, episode_index: int, video_key: str) -> str:
    payload = f"{dataset_path}:{episode_index}:{video_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
