"""MoWA future latent cache builder / validator / fake-encoder utilities.

This module provides the first executable P1 cache loop:

- deterministic fake encoder adapter
- cache writer with manifest emission
- cache validator

It does not touch the real Wan encoder / VAE stack yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from starVLA.dataloader.mowa.sampler import select_mowa_smoke_anchor_index


_WINDOW_ROLE_CURRENT = "current"
_WINDOW_ROLE_FUTURE = "future"
_WINDOW_ROLE_HISTORY = "history"
_WINDOW_ROLE_CHOICES = (
    _WINDOW_ROLE_CURRENT,
    _WINDOW_ROLE_FUTURE,
    _WINDOW_ROLE_HISTORY,
)


@dataclass(frozen=True)
class MoWALatentWindowSpec:
    role: str
    steps: int

    def validate(self) -> None:
        if self.role not in _WINDOW_ROLE_CHOICES:
            raise ValueError(f"Unsupported latent window role: {self.role!r}.")
        if self.steps <= 0:
            raise ValueError(f"Latent window steps must be positive, got {self.steps}.")


@dataclass(frozen=True)
class MoWALatentCacheBuildConfig:
    dataset_path: Path
    cache_root: Path
    encoder_name: str = "wan-fake-encoder"
    encoder_version: str = "fake-v1"
    latent_dim: int = 1024
    video_keys: tuple[str, ...] = (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    )
    current_window_steps: int = 1
    future_window_steps: int = 8
    history_window_steps: int = 8
    episode_indices: tuple[int, ...] | None = None
    anchor_indices: tuple[int, ...] | None = None
    anchor_mode: str = "smoke"
    allow_partial_windows: bool = False
    overwrite: bool = False
    dry_run: bool = True
    manifest_name: str = "manifest.json"
    source_video_relpath_template: str = (
        "videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
    )

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}.")
        if not self.video_keys:
            raise ValueError("video_keys must be non-empty.")
        if self.current_window_steps <= 0:
            raise ValueError("current_window_steps must be positive.")
        if self.future_window_steps <= 0:
            raise ValueError("future_window_steps must be positive.")
        if self.history_window_steps <= 0:
            raise ValueError("history_window_steps must be positive.")
        if self.anchor_mode not in {"smoke", "all", "custom"}:
            raise ValueError(
                f"anchor_mode must be smoke/all/custom, got {self.anchor_mode!r}."
            )
        if self.anchor_mode == "custom" and not self.anchor_indices:
            raise ValueError("anchor_indices must be provided when anchor_mode='custom'.")

    def window_specs(self) -> tuple[MoWALatentWindowSpec, ...]:
        specs = [
            MoWALatentWindowSpec(role=_WINDOW_ROLE_CURRENT, steps=self.current_window_steps),
            MoWALatentWindowSpec(role=_WINDOW_ROLE_FUTURE, steps=self.future_window_steps),
        ]
        if self.history_window_steps > 0:
            specs.append(
                MoWALatentWindowSpec(role=_WINDOW_ROLE_HISTORY, steps=self.history_window_steps)
            )
        for spec in specs:
            spec.validate()
        return tuple(specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "cache_root": str(self.cache_root),
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "latent_dim": self.latent_dim,
            "video_keys": self.video_keys,
            "current_window_steps": self.current_window_steps,
            "future_window_steps": self.future_window_steps,
            "history_window_steps": self.history_window_steps,
            "episode_indices": self.episode_indices,
            "anchor_indices": self.anchor_indices,
            "anchor_mode": self.anchor_mode,
            "allow_partial_windows": self.allow_partial_windows,
            "overwrite": self.overwrite,
            "dry_run": self.dry_run,
            "manifest_name": self.manifest_name,
            "source_video_relpath_template": self.source_video_relpath_template,
        }


@dataclass(frozen=True)
class MoWALatentCacheArtifact:
    cache_key: str
    cache_path: str
    dataset_path: str
    episode_index: int
    anchor_index: int
    video_key: str
    window_role: str
    frame_indices: tuple[int, ...]
    latent_shape: tuple[int, ...]
    dtype: str
    encoder_name: str
    encoder_version: str
    source_identity: str
    source_sha256: str
    latent_sha256: str
    video_exists: bool
    cache_status: str
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "dataset_path": self.dataset_path,
            "episode_index": self.episode_index,
            "anchor_index": self.anchor_index,
            "video_key": self.video_key,
            "window_role": self.window_role,
            "frame_indices": self.frame_indices,
            "latent_shape": self.latent_shape,
            "dtype": self.dtype,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "source_identity": self.source_identity,
            "source_sha256": self.source_sha256,
            "latent_sha256": self.latent_sha256,
            "video_exists": self.video_exists,
            "cache_status": self.cache_status,
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class MoWALatentCacheBuildReport:
    dataset_path: str
    cache_root: str
    manifest_path: str
    encoder_name: str
    encoder_version: str
    latent_dim: int
    dry_run: bool
    episode_count: int
    anchor_count: int
    planned_artifact_count: int
    written_artifact_count: int
    skipped_missing_episode_count: int
    skipped_boundary_count: int
    skipped_existing_count: int
    preview_artifacts: tuple[MoWALatentCacheArtifact, ...]
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...] = (
        "This builder uses a deterministic fake encoder and CPU-only tensor I/O.",
        "No Wan encoder, VAE or training code is executed.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "cache_root": self.cache_root,
            "manifest_path": self.manifest_path,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "latent_dim": self.latent_dim,
            "dry_run": self.dry_run,
            "episode_count": self.episode_count,
            "anchor_count": self.anchor_count,
            "planned_artifact_count": self.planned_artifact_count,
            "written_artifact_count": self.written_artifact_count,
            "skipped_missing_episode_count": self.skipped_missing_episode_count,
            "skipped_boundary_count": self.skipped_boundary_count,
            "skipped_existing_count": self.skipped_existing_count,
            "preview_artifacts": [artifact.to_dict() for artifact in self.preview_artifacts],
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWALatentCacheValidationReport:
    cache_root: str
    manifest_path: str
    manifest_exists: bool
    artifact_count: int
    readable_artifact_count: int
    missing_artifact_count: int
    shape_mismatch_count: int
    hash_mismatch_count: int
    duplicate_cache_key_count: int
    all_ok: bool
    summary: dict[str, Any]
    go_no_go: str
    notes: tuple[str, ...] = (
        "Validation checks manifest consistency, file presence and latent hashes.",
        "It does not attempt real encoder parity or training integration.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_root": self.cache_root,
            "manifest_path": self.manifest_path,
            "manifest_exists": self.manifest_exists,
            "artifact_count": self.artifact_count,
            "readable_artifact_count": self.readable_artifact_count,
            "missing_artifact_count": self.missing_artifact_count,
            "shape_mismatch_count": self.shape_mismatch_count,
            "hash_mismatch_count": self.hash_mismatch_count,
            "duplicate_cache_key_count": self.duplicate_cache_key_count,
            "all_ok": self.all_ok,
            "summary": self.summary,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


class MoWALatentEncoderAdapter(Protocol):
    encoder_name: str
    encoder_version: str
    latent_dim: int

    def encode(
        self,
        *,
        dataset_path: Path,
        episode_index: int,
        anchor_index: int,
        video_key: str,
        window_role: str,
        frame_indices: tuple[int, ...],
        row_count: int,
        source_path: Path | None,
        source_identity: str,
    ) -> torch.Tensor:
        """Encode one deterministic latent window."""


@dataclass(frozen=True)
class MoWAFakeLatentEncoderAdapter:
    encoder_name: str = "wan-fake-encoder"
    encoder_version: str = "fake-v1"
    latent_dim: int = 1024
    seed_salt: str = "mowa-fake-latent-cache"

    def encode(
        self,
        *,
        dataset_path: Path,
        episode_index: int,
        anchor_index: int,
        video_key: str,
        window_role: str,
        frame_indices: tuple[int, ...],
        row_count: int,
        source_path: Path | None,
        source_identity: str,
    ) -> torch.Tensor:
        del dataset_path, source_path, row_count
        payload = "|".join(
            (
                self.seed_salt,
                self.encoder_name,
                self.encoder_version,
                str(episode_index),
                str(anchor_index),
                video_key,
                window_role,
                ",".join(str(index) for index in frame_indices),
                source_identity,
            )
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        latent = rng.standard_normal(self.latent_dim, dtype=np.float32)
        return torch.from_numpy(latent)


def build_mowa_latent_cache(
    config: MoWALatentCacheBuildConfig,
    *,
    encoder: MoWALatentEncoderAdapter | None = None,
) -> MoWALatentCacheBuildReport:
    """Build a deterministic fake latent cache and optionally write it to disk."""

    config.validate()
    encoder = encoder or MoWAFakeLatentEncoderAdapter(
        encoder_name=config.encoder_name,
        encoder_version=config.encoder_version,
        latent_dim=config.latent_dim,
    )

    dataset_path = Path(config.dataset_path)
    cache_root = Path(config.cache_root)
    manifest_path = cache_root / config.manifest_name
    parquet_paths = _list_episode_parquet_paths(dataset_path)
    episode_paths = _filter_episode_parquet_paths(parquet_paths, config.episode_indices)

    planned_artifacts: list[MoWALatentCacheArtifact] = []
    skipped_missing_episode_count = 0
    skipped_boundary_count = 0
    skipped_existing_count = 0
    written_artifact_count = 0
    anchor_count = 0

    window_specs = config.window_specs()
    for parquet_path in episode_paths:
        episode_index = _episode_index_from_path(parquet_path)
        row_count = _read_parquet_row_count(parquet_path)
        anchor_indices = _resolve_anchor_indices(
            row_count=row_count,
            config=config,
            window_specs=window_specs,
        )
        if not anchor_indices:
            skipped_missing_episode_count += 1
            continue
        anchor_count += len(anchor_indices)
        for anchor_index in anchor_indices:
            for video_key in config.video_keys:
                source_path = _resolve_video_path(dataset_path, video_key, episode_index)
                source_identity = _source_identity(dataset_path, episode_index, video_key, source_path)
                source_sha256 = _sha256_for_source(source_path, source_identity)
                for window_spec in window_specs:
                    frame_indices = _frame_indices_for_window(
                        window_role=window_spec.role,
                        anchor_index=anchor_index,
                        row_count=row_count,
                        steps=window_spec.steps,
                    )
                    if frame_indices is None:
                        skipped_boundary_count += 1
                        continue

                    cache_key = _cache_key(
                        dataset_path=dataset_path,
                        episode_index=episode_index,
                        anchor_index=anchor_index,
                        video_key=video_key,
                        window_role=window_spec.role,
                        frame_indices=frame_indices,
                        encoder_name=encoder.encoder_name,
                        encoder_version=encoder.encoder_version,
                        latent_dim=encoder.latent_dim,
                    )
                    cache_path = cache_root / f"{cache_key}.pt"
                    latent = encoder.encode(
                        dataset_path=dataset_path,
                        episode_index=episode_index,
                        anchor_index=anchor_index,
                        video_key=video_key,
                        window_role=window_spec.role,
                        frame_indices=frame_indices,
                        row_count=row_count,
                        source_path=source_path,
                        source_identity=source_identity,
                    ).detach().cpu()
                    if latent.ndim != 1:
                        raise ValueError(
                            f"Latent encoder must return a 1D tensor, got shape {tuple(latent.shape)}."
                        )
                    if latent.shape[0] != encoder.latent_dim:
                        raise ValueError(
                            f"Latent dim mismatch: expected {encoder.latent_dim}, got {latent.shape[0]}."
                        )

                    latent_sha256 = _sha256_for_tensor(latent)
                    artifact = MoWALatentCacheArtifact(
                        cache_key=cache_key,
                        cache_path=str(cache_path),
                        dataset_path=str(dataset_path),
                        episode_index=episode_index,
                        anchor_index=anchor_index,
                        video_key=video_key,
                        window_role=window_spec.role,
                        frame_indices=frame_indices,
                        latent_shape=tuple(int(dim) for dim in latent.shape),
                        dtype=str(latent.dtype).replace("torch.", ""),
                        encoder_name=encoder.encoder_name,
                        encoder_version=encoder.encoder_version,
                        source_identity=source_identity,
                        source_sha256=source_sha256,
                        latent_sha256=latent_sha256,
                        video_exists=source_path.is_file() if source_path is not None else False,
                        cache_status="planned"
                        if config.dry_run
                        else ("exists_unverified" if cache_path.exists() and not config.overwrite else "written"),
                        row_count=row_count,
                    )
                    planned_artifacts.append(artifact)
                    if cache_path.exists() and not config.overwrite:
                        skipped_existing_count += 1
                    elif not config.dry_run:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "latent": latent,
                                "metadata": artifact.to_dict(),
                            },
                            cache_path,
                        )
                        written_artifact_count += 1

    cache_root.mkdir(parents=True, exist_ok=True)
    if not config.dry_run:
        _write_manifest(
            manifest_path,
            config=config,
            artifacts=planned_artifacts,
            written_artifact_count=written_artifact_count,
        )

    summary = {
        "episode_paths": len(episode_paths),
        "window_role_count": len(window_specs),
        "video_key_count": len(config.video_keys),
        "dry_run": config.dry_run,
        "manifest_written": not config.dry_run,
        "source_policy": "fake_encoder_cpu_only",
    }
    return MoWALatentCacheBuildReport(
        dataset_path=str(dataset_path),
        cache_root=str(cache_root),
        manifest_path=str(manifest_path),
        encoder_name=encoder.encoder_name,
        encoder_version=encoder.encoder_version,
        latent_dim=encoder.latent_dim,
        dry_run=config.dry_run,
        episode_count=len(episode_paths),
        anchor_count=anchor_count,
        planned_artifact_count=len(planned_artifacts),
        written_artifact_count=written_artifact_count,
        skipped_missing_episode_count=skipped_missing_episode_count,
        skipped_boundary_count=skipped_boundary_count,
        skipped_existing_count=skipped_existing_count,
        preview_artifacts=tuple(planned_artifacts[:8]),
        summary=summary,
        go_no_go=(
            "TBD: fake latent cache planned; execute and validate before integration"
            if planned_artifacts
            else "No-Go: no latent cache artifacts were planned"
        ),
    )


def validate_mowa_latent_cache(
    cache_root: Path | str,
    *,
    manifest_path: Path | str | None = None,
) -> MoWALatentCacheValidationReport:
    """Validate a fake latent cache manifest and artifact set."""

    cache_root = Path(cache_root)
    manifest_path = Path(manifest_path) if manifest_path is not None else cache_root / "manifest.json"
    manifest_exists = manifest_path.is_file()
    artifacts = _read_manifest_artifacts(manifest_path) if manifest_exists else _scan_cache_artifacts(cache_root)

    missing_artifact_count = 0
    readable_artifact_count = 0
    shape_mismatch_count = 0
    hash_mismatch_count = 0
    duplicate_cache_key_count = 0
    cache_keys: list[str] = []

    for artifact in artifacts:
        cache_keys.append(artifact["cache_key"])
        cache_path = Path(artifact["cache_path"])
        if not cache_path.is_file():
            missing_artifact_count += 1
            continue
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            shape_mismatch_count += 1
            continue
        if not isinstance(payload, dict) or "latent" not in payload or "metadata" not in payload:
            shape_mismatch_count += 1
            continue
        latent = payload["latent"]
        metadata = payload["metadata"]
        if not isinstance(latent, torch.Tensor):
            shape_mismatch_count += 1
            continue
        if tuple(int(dim) for dim in latent.shape) != tuple(artifact["latent_shape"]):
            shape_mismatch_count += 1
            continue
        readable_artifact_count += 1
        if _sha256_for_tensor(latent.detach().cpu()) != artifact["latent_sha256"]:
            hash_mismatch_count += 1
        if isinstance(metadata, dict):
            if metadata.get("cache_key") != artifact["cache_key"]:
                hash_mismatch_count += 1
        else:
            shape_mismatch_count += 1

    duplicate_cache_key_count = len(cache_keys) - len(set(cache_keys))
    all_ok = (
        manifest_exists
        and missing_artifact_count == 0
        and shape_mismatch_count == 0
        and hash_mismatch_count == 0
        and duplicate_cache_key_count == 0
    )
    summary = {
        "cache_root": str(cache_root),
        "manifest_entries": len(artifacts),
        "manifest_exists": manifest_exists,
    }
    return MoWALatentCacheValidationReport(
        cache_root=str(cache_root),
        manifest_path=str(manifest_path),
        manifest_exists=manifest_exists,
        artifact_count=len(artifacts),
        readable_artifact_count=readable_artifact_count,
        missing_artifact_count=missing_artifact_count,
        shape_mismatch_count=shape_mismatch_count,
        hash_mismatch_count=hash_mismatch_count,
        duplicate_cache_key_count=duplicate_cache_key_count,
        all_ok=all_ok,
        summary=summary,
        go_no_go=(
            "TBD: latent cache validation passed; cache remains fake-encoder only"
            if all_ok
            else "No-Go: latent cache validation failed"
        ),
    )


class MoWALatentCacheDataset:
    """Read-only cache loader that groups artifacts by sample key."""

    def __init__(self, cache_root: Path | str, manifest_path: Path | str | None = None):
        self.cache_root = Path(cache_root)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else self.cache_root / "manifest.json"
        self._artifacts = self._load_artifacts()
        self._sample_keys = tuple(
            sorted(
                {
                    (
                        artifact["episode_index"],
                        artifact["anchor_index"],
                        artifact["video_key"],
                    )
                    for artifact in self._artifacts
                }
            )
        )

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
    def artifacts(self) -> tuple[dict[str, Any], ...]:
        return self._artifacts

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
        matching = [
            artifact
            for artifact in self._artifacts
            if artifact["episode_index"] == episode_index
            and artifact["anchor_index"] == anchor_index
            and artifact["video_key"] == video_key
        ]
        if not matching:
            raise KeyError(
                f"No cache artifacts for episode={episode_index}, anchor={anchor_index}, video_key={video_key!r}."
            )
        grouped: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for artifact in matching:
            payload = torch.load(Path(artifact["cache_path"]), map_location="cpu")
            latent = payload["latent"]
            grouped[f"{artifact['window_role']}_latent"] = latent
            metadata[artifact["window_role"]] = payload["metadata"]
        return {
            "episode_index": episode_index,
            "anchor_index": anchor_index,
            "video_key": video_key,
            "current_latent": grouped.get("current_latent"),
            "future_latent": grouped.get("future_latent"),
            "history_latent": grouped.get("history_latent"),
            "latents": grouped,
            "metadata": metadata,
        }

    def _load_artifacts(self) -> tuple[dict[str, Any], ...]:
        if self.manifest_path.is_file():
            return _read_manifest_artifacts(self.manifest_path)
        return _scan_cache_artifacts(self.cache_root)


def _list_episode_parquet_paths(dataset_path: Path) -> tuple[Path, ...]:
    data_dir = dataset_path / "data" / "chunk-000"
    if not data_dir.is_dir():
        return ()
    return tuple(sorted(data_dir.glob("episode_*.parquet")))


def _filter_episode_parquet_paths(
    parquet_paths: tuple[Path, ...],
    episode_indices: tuple[int, ...] | None,
) -> tuple[Path, ...]:
    if episode_indices is None:
        return parquet_paths
    allowed = set(episode_indices)
    return tuple(
        parquet_path
        for parquet_path in parquet_paths
        if _episode_index_from_path(parquet_path) in allowed
    )


def _episode_index_from_path(parquet_path: Path) -> int:
    stem = parquet_path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unable to parse episode index from {parquet_path}") from exc


def _read_parquet_row_count(parquet_path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA latent cache builder requires pyarrow.") from exc

    return int(pq.ParquetFile(parquet_path).metadata.num_rows)


def _resolve_anchor_indices(
    *,
    row_count: int,
    config: MoWALatentCacheBuildConfig,
    window_specs: tuple[MoWALatentWindowSpec, ...],
) -> tuple[int, ...]:
    if row_count <= 0:
        return ()
    if config.anchor_indices is not None:
        valid = []
        for anchor_index in config.anchor_indices:
            if 0 <= anchor_index < row_count:
                valid.append(anchor_index)
        return tuple(valid)
    if config.anchor_mode == "all":
        first_anchor = max(
            max(spec.steps - 1 if spec.role == _WINDOW_ROLE_HISTORY else 0 for spec in window_specs),
            0,
        )
        last_anchor = min(
            row_count - 1,
            min(
                row_count - 1 if spec.role == _WINDOW_ROLE_CURRENT else row_count - spec.steps - 1
                for spec in window_specs
            ),
        )
        if last_anchor < first_anchor:
            return ()
        return tuple(range(first_anchor, last_anchor + 1))

    max_history = max((spec.steps for spec in window_specs if spec.role == _WINDOW_ROLE_HISTORY), default=1)
    max_future = max((spec.steps for spec in window_specs if spec.role == _WINDOW_ROLE_FUTURE), default=1)
    smoke_window = _select_smoke_window_config(max_history=max_history, max_future=max_future)
    return (select_mowa_smoke_anchor_index(row_count, smoke_window),)


def _select_smoke_window_config(
    *,
    max_history: int,
    max_future: int,
) -> Any:
    from starVLA.dataloader.mowa.schema import MoWAWindowConfig

    return MoWAWindowConfig(
        history_steps=max_history,
        future_steps=max_future,
        action_chunk_steps=1,
    )


def _frame_indices_for_window(
    *,
    window_role: str,
    anchor_index: int,
    row_count: int,
    steps: int,
) -> tuple[int, ...] | None:
    if window_role == _WINDOW_ROLE_CURRENT:
        if anchor_index >= row_count:
            return None
        return (anchor_index,)
    if window_role == _WINDOW_ROLE_FUTURE:
        start = anchor_index + 1
        end = min(row_count, anchor_index + 1 + steps)
        if end - start < steps:
            return None
        return tuple(range(start, end))
    if window_role == _WINDOW_ROLE_HISTORY:
        start = anchor_index - steps + 1
        end = anchor_index + 1
        if start < 0:
            return None
        return tuple(range(start, end))
    raise ValueError(f"Unsupported window role: {window_role!r}.")


def _resolve_video_path(
    dataset_path: Path,
    video_key: str,
    episode_index: int,
) -> Path:
    relative_path = Path(
        f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
    )
    return dataset_path / relative_path


def _source_identity(
    dataset_path: Path,
    episode_index: int,
    video_key: str,
    source_path: Path | None,
) -> str:
    return "|".join(
        (
            str(dataset_path),
            str(episode_index),
            video_key,
            str(source_path) if source_path is not None else "missing-source-path",
        )
    )


def _sha256_for_source(source_path: Path | None, source_identity: str) -> str:
    if source_path is not None and source_path.is_file():
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    return hashlib.sha256(source_identity.encode("utf-8")).hexdigest()


def _sha256_for_tensor(latent: torch.Tensor) -> str:
    array = latent.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _cache_key(
    *,
    dataset_path: Path,
    episode_index: int,
    anchor_index: int,
    video_key: str,
    window_role: str,
    frame_indices: tuple[int, ...],
    encoder_name: str,
    encoder_version: str,
    latent_dim: int,
) -> str:
    payload = "|".join(
        (
            str(dataset_path),
            str(episode_index),
            str(anchor_index),
            video_key,
            window_role,
            ",".join(str(index) for index in frame_indices),
            encoder_name,
            encoder_version,
            str(latent_dim),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _write_manifest(
    manifest_path: Path,
    *,
    config: MoWALatentCacheBuildConfig,
    artifacts: list[MoWALatentCacheArtifact],
    written_artifact_count: int,
) -> None:
    payload = {
        "config": config.to_dict(),
        "written_artifact_count": written_artifact_count,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_manifest_artifacts(manifest_path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", [])
    return tuple(dict(artifact) for artifact in artifacts)


def _scan_cache_artifacts(cache_root: Path) -> tuple[dict[str, Any], ...]:
    artifacts = []
    for cache_path in sorted(cache_root.glob("*.pt")):
        if cache_path.name == "manifest.json":
            continue
        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        artifacts.append(metadata)
    return tuple(artifacts)


# Runtime-facing aliases.
MoWAFakeLatentEncoder = MoWAFakeLatentEncoderAdapter
MoWAFutureLatentCacheBuildConfig = MoWALatentCacheBuildConfig
MoWAFutureLatentCacheArtifact = MoWALatentCacheArtifact
MoWAFutureLatentCacheBuildReport = MoWALatentCacheBuildReport
MoWAFutureLatentCacheValidationReport = MoWALatentCacheValidationReport
MoWAFutureLatentWindowSpec = MoWALatentWindowSpec
MoWAFutureLatentCacheDataset = MoWALatentCacheDataset
build_mowa_future_latent_cache = build_mowa_latent_cache
validate_mowa_future_latent_cache = validate_mowa_latent_cache
