"""MoWA episode-level latent store builder / reader / validator.

This module implements the default P1 cache strategy:

- One HDF5 file per episode (``latent_cache/ep_{episode_id}.h5``).
- Only high-dimensional visual information is pre-encoded into latents.
- Low-dimensional robot state / action / language are stored raw, not latent.
- Window manifest references slices into the episode store; dataloader gathers
  history / current / future latents dynamically.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from starVLA.model.modules.world_model.wan_vae_utils import (
    WAN_VAE_INPUT_HEIGHT,
    WAN_VAE_INPUT_WIDTH,
    WAN_VAE_PREPROCESS_POLICY,
    WAN_VAE_TEMPORAL_FACTOR,
    encode_wan_vae_video,
    prepare_wan_vae_video_tensor,
)


_DATA_GATE = "Data Gate"
_TIME_ORDER = "oldest_to_latest"


class MoWALatentEncoderAdapter(Protocol):
    """Adapter for encoding a contiguous video clip into visual latents."""

    encoder_name: str
    encoder_version: str

    def encode_video_clip(
        self,
        *,
        video_path: Path,
        frame_indices: tuple[int, ...],
        video_key: str,
    ) -> np.ndarray:
        """Return visual latents for the requested frame indices.

        The returned array must have shape ``[len(frame_indices), *latent_shape_per_frame]``
        and dtype that can be stored in HDF5 (typically ``float16`` or ``float32``).
        """


@dataclass(frozen=True)
class MoWAFakeLatentEncoderAdapter:
    """Deterministic fake encoder for unit tests and smoke checks.

    Produces per-frame 1D latent vectors so that the fake latent store can be
    exercised without touching a real VAE.
    """

    encoder_name: str = "mowa-fake-encoder"
    encoder_version: str = "fake-v1"
    latent_dim: int = 16
    seed_salt: str = "mowa-fake-latent-cache"

    def encode_video_clip(
        self,
        *,
        video_path: Path,
        frame_indices: tuple[int, ...],
        video_key: str,
    ) -> np.ndarray:
        del video_path
        latents = []
        for frame_index in frame_indices:
            payload = "|".join(
                (
                    self.seed_salt,
                    self.encoder_name,
                    self.encoder_version,
                    video_key,
                    str(frame_index),
                )
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            latents.append(rng.standard_normal(self.latent_dim, dtype=np.float32))
        return np.stack(latents, axis=0)


@dataclass
class MoWAWanVaeEpisodeEncoderAdapter:
    """Encode a full episode with the Wan2.2 video VAE.

    Wan2.2 is a temporal VAE: a continuous ``[C, T, H, W]`` video is encoded
    to ``[C_z, (T - 1) // 4 + 1, h, w]``.  Frames must therefore never be
    moved into the batch dimension.  The cache stores the temporal latent
    sequence and a source-frame-to-latent mapping alongside it.
    Output shape depends on ``latent_type``:

    - ``vae_spatial``: ``[T_z, C, h, w]`` — raw temporal VAE spatial latent.
    - ``pooled_vector``: ``[T_z, D]`` — mean-pooled temporal latent, optionally
      projected to ``latent_dim``.
    """

    model_path: Path | str
    encoder_name: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    encoder_version: str = "wan2.2-vae-v1"
    latent_type: str = "vae_spatial"
    latent_dim: int = 1024
    flatten_policy: str = "none"
    video_backend: str = "opencv"
    vae_batch_size: int = 1
    height: int = WAN_VAE_INPUT_HEIGHT
    width: int = WAN_VAE_INPUT_WIDTH
    _projection_seed: int = 42
    temporal_compression_factor: int = WAN_VAE_TEMPORAL_FACTOR

    def __post_init__(self) -> None:
        if self.latent_type not in {"vae_spatial", "pooled_vector"}:
            raise ValueError(f"Unsupported latent_type: {self.latent_type!r}.")
        if self.vae_batch_size <= 0:
            raise ValueError(f"vae_batch_size must be positive, got {self.vae_batch_size}.")
        if (self.height, self.width) != (256, 256):
            raise ValueError(
                "Wan2.2 Robocasa encoder requires a 256x256 VAE input; "
                f"got {self.height}x{self.width}."
            )
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wan2.2 model path not found: {model_path}")

    def encode_video_clip(
        self,
        *,
        video_path: Path,
        frame_indices: tuple[int, ...],
        video_key: str,
    ) -> np.ndarray:
        del video_key
        if self.video_backend not in {"opencv", "decord"}:
            raise NotImplementedError(
                f"Unsupported video_backend={self.video_backend!r}; only 'opencv' and 'decord' are available."
            )
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing source video for Wan latent encoding: {video_path}")

        self._ensure_loaded()
        assert self._vae is not None and self._video_processor is not None and self._torch_dtype is not None

        frames = self.load_video_frames(video_path, frame_indices)

        if not frames:
            raise ValueError(f"No frames available for Wan latent encoding: {video_path}")
        return self.encode_frames(frames)

    def load_video_frames(self, video_path: Path, frame_indices: tuple[int, ...]) -> list[Any]:
        """CPU-side video decode suitable for a prefetch worker."""
        if self.video_backend == "decord":
            return self._load_all_frames_decord(video_path, frame_indices)
        return self._load_all_frames_opencv(video_path, frame_indices)

    def encode_frames(self, frames: list[Any]) -> np.ndarray:
        """Encode predecoded frames on the owning GPU process."""
        full_latent = self._encode_video(frames)  # [T_z, C, h, w]

        if self.latent_type == "pooled_vector":
            full_latent = self._to_pooled_vector(full_latent)

        return full_latent.detach().cpu().numpy().astype(np.float32)

    def _ensure_loaded(self) -> None:
        if self._vae is not None and self._video_processor is not None:
            return
        try:
            from diffusers import AutoencoderKLWan
            from diffusers.video_processor import VideoProcessor
        except ImportError as exc:
            raise RuntimeError("MoWA Wan episode encoder requires diffusers.") from exc

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wan2.2 model path not found: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        print(
            f"[episode-cache] loading Wan2.2 VAE model={model_path} "
            f"device={device} compute_dtype={dtype}",
            flush=True,
        )
        self._vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)
        self._video_processor = VideoProcessor(vae_scale_factor=2 ** len(self._vae.temperal_downsample))
        self._torch_dtype = dtype
        print("[episode-cache] Wan2.2 VAE ready", flush=True)

    def _load_all_frames_decord(self, source_path: Path, frame_indices: tuple[int, ...]) -> list[Any]:
        try:
            from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_indices
        except ImportError as exc:
            raise RuntimeError("MoWA Wan encoder requires gr00t_lerobot video helpers.") from exc
        try:
            frames = get_frames_by_indices(
                str(source_path),
                list(frame_indices),
                video_backend="decord",
            )
        except ImportError as exc:
            raise RuntimeError("MoWA Wan encoder requires decord for video_backend='decord'.") from exc
        return [frame for frame in frames]

    def latent_source_frame_indices(self, frame_count: int) -> np.ndarray:
        """Return the final source-frame index represented by each temporal latent."""
        if frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count}.")
        last_latent_index = (frame_count - 1 + self.temporal_compression_factor - 1) // self.temporal_compression_factor
        indices = np.arange(last_latent_index + 1, dtype=np.int64) * self.temporal_compression_factor
        return np.minimum(indices, frame_count - 1)

    def _load_all_frames_opencv(
        self,
        source_path: Path,
        frame_indices: tuple[int, ...],
    ) -> list[np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("MoWA Wan encoder requires opencv-python.") from exc

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source_path}")

        target_indices = list(frame_indices)
        idx_pointer = 0
        frame_counter = 0
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx_pointer < len(target_indices) and frame_counter == target_indices[idx_pointer]:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    idx_pointer += 1
                frame_counter += 1
        finally:
            cap.release()

        if idx_pointer != len(target_indices):
            raise ValueError(
                f"Could not read all requested frames from {source_path}: "
                f"got {idx_pointer}/{len(target_indices)}"
            )
        return frames

    def _encode_video(self, frames: list[Any]) -> torch.Tensor:
        assert self._vae is not None and self._video_processor is not None and self._torch_dtype is not None
        video_tensor = prepare_wan_vae_video_tensor(self._video_processor, frames)
        device = next(self._vae.parameters()).device
        video_tensor = video_tensor.to(device=device, dtype=self._torch_dtype)
        latents = encode_wan_vae_video(self._vae, video_tensor)
        if latents.dim() != 5 or latents.shape[0] != 1:
            raise ValueError(f"Expected one 5D Wan video latent, got shape {tuple(latents.shape)}.")
        return latents.squeeze(0).permute(1, 0, 2, 3).to(dtype=torch.float32)

    def _to_pooled_vector(self, spatial_latent: torch.Tensor) -> torch.Tensor:
        """Convert [T, C, h, w] spatial latent to [T, D] pooled vector."""
        pooled = spatial_latent.float().mean(dim=(2, 3))  # [T, C]
        if self.flatten_policy == "mean_pool" and self.latent_dim != pooled.shape[-1]:
            if self._projection is None:
                with torch.random.fork_rng():
                    torch.manual_seed(self._projection_seed)
                    self._projection = nn.Linear(
                        pooled.shape[-1],
                        self.latent_dim,
                    ).to(device=pooled.device, dtype=pooled.dtype).eval()
            pooled = self._projection(pooled)
        return pooled

    _vae: Any = field(default=None, init=False, repr=False)
    _video_processor: Any = field(default=None, init=False, repr=False)
    _torch_dtype: Any = field(default=None, init=False, repr=False)
    _projection: nn.Linear | None = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class MoWAEpisodeLatentStoreConfig:
    """Configuration for building an episode-level latent store."""

    dataset_path: Path
    cache_root: Path
    video_keys: tuple[str, ...] = (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    )
    encoder_kind: str = "fake"
    encoder_name: str = "mowa-fake-encoder"
    encoder_version: str = "fake-v1"
    encoder_model_path: Path | None = None
    latent_model: str = "Wan2.2-VAE"
    latent_model_version: str = "TBD"
    latent_type: str = "pooled_vector"
    latent_shape_per_frame: tuple[str | int, ...] = ("D",)
    latent_dim: int = 1024
    flatten_policy: str = "none"
    dtype: str = "float32"
    video_backend: str = "opencv"
    vae_batch_size: int = 1
    obs_fps: float | str = _DATA_GATE
    action_hz: float | str = _DATA_GATE
    wam_hz: float | str = _DATA_GATE
    time_order: str = _TIME_ORDER
    dry_run: bool = True
    overwrite: bool = False
    language_source: str = "meta/tasks.jsonl"
    downsample_policy: str = "none"
    episode_indices: tuple[int, ...] | None = None
    shard_index: int = 0
    num_shards: int = 1
    num_workers: int = 6

    def validate(self) -> None:
        if not self.video_keys:
            raise ValueError("video_keys must be non-empty.")
        if self.time_order != _TIME_ORDER:
            raise ValueError(f"time_order must be {_TIME_ORDER!r}, got {self.time_order!r}.")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype!r}.")
        if self.encoder_kind not in {"fake", "wan2.2-vae"}:
            raise ValueError(f"encoder_kind must be fake/wan2.2-vae, got {self.encoder_kind!r}.")
        if self.encoder_kind == "wan2.2-vae" and self.encoder_model_path is None:
            raise ValueError("encoder_model_path must be provided when encoder_kind='wan2.2-vae'.")
        if self.encoder_kind == "wan2.2-vae" and self.latent_type not in {"vae_spatial", "pooled_vector"}:
            raise ValueError(
                f"Wan VAE encoder supports latent_type=vae_spatial/pooled_vector, got {self.latent_type!r}."
            )
        if self.vae_batch_size <= 0:
            raise ValueError(f"vae_batch_size must be positive, got {self.vae_batch_size}.")
        if self.num_shards <= 0 or not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards).")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "cache_root": str(self.cache_root),
            "video_keys": self.video_keys,
            "encoder_kind": self.encoder_kind,
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "encoder_model_path": str(self.encoder_model_path) if self.encoder_model_path is not None else None,
            "latent_model": self.latent_model,
            "latent_model_version": self.latent_model_version,
            "latent_type": self.latent_type,
            "latent_shape_per_frame": self.latent_shape_per_frame,
            "latent_dim": self.latent_dim,
            "flatten_policy": self.flatten_policy,
            "dtype": self.dtype,
            "video_backend": self.video_backend,
            "vae_batch_size": self.vae_batch_size,
            "obs_fps": self.obs_fps,
            "action_hz": self.action_hz,
            "wam_hz": self.wam_hz,
            "time_order": self.time_order,
            "dry_run": self.dry_run,
            "overwrite": self.overwrite,
            "language_source": self.language_source,
            "downsample_policy": self.downsample_policy,
            "episode_indices": self.episode_indices,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class MoWAEpisodeLatentStoreBuildReport:
    dataset_path: str
    cache_root: str
    episode_count: int
    written_count: int
    skipped_count: int
    failed_count: int
    dry_run: bool
    details: tuple[dict[str, Any], ...]
    go_no_go: str
    notes: tuple[str, ...] = (
        "Episode-level latent store keeps visual latents bound to original frames.",
        "Window manifest slices are generated separately and can vary history/future/stride.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "cache_root": self.cache_root,
            "episode_count": self.episode_count,
            "written_count": self.written_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "dry_run": self.dry_run,
            "details": list(self.details),
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MoWAEpisodeLatentStoreValidationReport:
    cache_root: str
    checked_count: int
    valid_count: int
    invalid_count: int
    missing_count: int
    go_no_go: str
    notes: tuple[str, ...] = (
        "Validation checks file presence, required groups, attrs and latent shape consistency.",
        "It does not check real encoder parity or training integration.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_root": self.cache_root,
            "checked_count": self.checked_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "missing_count": self.missing_count,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


class MoWAEpisodeLatentStore:
    """Read-only accessor for a single episode latent store HDF5 file."""

    def __init__(self, store_path: Path | str) -> None:
        self.store_path = Path(store_path)
        if not self.store_path.is_file():
            raise FileNotFoundError(f"Episode latent store not found: {self.store_path}")

    @property
    def attrs(self) -> dict[str, Any]:
        with h5py.File(self.store_path, "r") as file:
            return dict(file.attrs)

    def list_video_keys(self) -> tuple[str, ...]:
        with h5py.File(self.store_path, "r") as file:
            latents = file.get("/latents")
            if latents is None:
                return ()
            return tuple(sorted(latents.keys()))

    def num_frames(self, video_key: str) -> int:
        with h5py.File(self.store_path, "r") as file:
            dataset = file.get(f"/latents/{video_key}")
            if dataset is None:
                raise KeyError(f"No latent group for video_key={video_key!r}")
            source_indices = file.get(f"/indices/latent_source_frame_indices/{video_key}")
            if source_indices is not None:
                return int(source_indices[-1]) + 1
            return int(dataset.shape[0])

    def num_latent_frames(self, video_key: str) -> int:
        with h5py.File(self.store_path, "r") as file:
            dataset = file.get(f"/latents/{video_key}")
            if dataset is None:
                raise KeyError(f"No latent group for video_key={video_key!r}")
            return int(dataset.shape[0])

    def get_latent_source_frame_indices(self, video_key: str) -> np.ndarray | None:
        """Return the source-frame endpoint for every physical latent frame."""
        with h5py.File(self.store_path, "r") as file:
            dataset = file.get(f"/indices/latent_source_frame_indices/{video_key}")
            if dataset is None:
                return None
            return np.asarray(dataset[()], dtype=np.int64)

    def get_latents_by_latent_indices(
        self,
        video_key: str,
        latent_indices: tuple[int, ...],
    ) -> torch.Tensor:
        """Gather physical VAE latent frames without source-frame remapping."""
        with h5py.File(self.store_path, "r") as file:
            dataset = file[f"/latents/{video_key}"]
            if not latent_indices:
                return torch.from_numpy(np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype))
            indices = np.asarray(latent_indices, dtype=np.int64)
            if indices.min() < 0 or indices.max() >= dataset.shape[0]:
                raise IndexError(f"Requested latent indices out of range: {latent_indices}")
            unique_indices, inverse = np.unique(indices, return_inverse=True)
            array = np.asarray(dataset[unique_indices.tolist(), ...])[inverse]
        return torch.from_numpy(array)

    def get_latents(
        self,
        video_key: str,
        frame_indices: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Gather visual latents for the requested frames.

        If ``frame_indices`` is None, return the full episode latent sequence.
        """
        with h5py.File(self.store_path, "r") as file:
            dataset = file[f"/latents/{video_key}"]
            if frame_indices is None:
                array = dataset[()]
            else:
                source_indices = file.get(f"/indices/latent_source_frame_indices/{video_key}")
                if source_indices is None:
                    array = dataset[list(frame_indices), ...]
                else:
                    source_frame_indices = np.asarray(source_indices[()])
                    requested = np.asarray(frame_indices, dtype=np.int64)
                    if requested.size == 0 or requested.min() < 0 or requested.max() >= self.num_frames(video_key):
                        raise IndexError(f"Requested source frame indices out of range: {frame_indices}")
                    latent_indices = np.searchsorted(source_frame_indices, requested, side="right") - 1
                    latent_indices = np.clip(latent_indices, 0, len(source_frame_indices) - 1)
                    unique_indices, inverse = np.unique(latent_indices, return_inverse=True)
                    array = np.asarray(dataset[unique_indices.tolist(), ...])[inverse]
        return torch.from_numpy(np.asarray(array))

    def get_valid_mask(self, video_key: str) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            dataset = file.get(f"/valid/{video_key}")
            if dataset is None:
                raise KeyError(f"No valid mask for video_key={video_key!r}")
            return torch.from_numpy(np.asarray(dataset[()]))

    def get_robot_state(self) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            return torch.from_numpy(np.asarray(file["/robot/state"][()]))

    def get_robot_action(self) -> torch.Tensor:
        with h5py.File(self.store_path, "r") as file:
            return torch.from_numpy(np.asarray(file["/robot/action"][()]))

    def get_language(self) -> str:
        with h5py.File(self.store_path, "r") as file:
            raw = file["/language/instruction"][()]
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)

    def get_frame_indices(self) -> tuple[int, ...]:
        with h5py.File(self.store_path, "r") as file:
            array = np.asarray(file["/indices/frame_indices"][()])
            return tuple(int(value) for value in array)


def build_mowa_episode_latent_store(
    config: MoWAEpisodeLatentStoreConfig,
    *,
    encoder: MoWALatentEncoderAdapter | None = None,
) -> MoWAEpisodeLatentStoreBuildReport:
    """Build episode-level latent stores from a LeRobot-style dataset."""

    config.validate()
    encoder = encoder or _build_encoder_adapter(config)

    dataset_path = Path(config.dataset_path)
    cache_root = Path(config.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    parquet_paths = _list_episode_parquet_paths(dataset_path)
    parquet_paths = _filter_episode_parquet_paths(parquet_paths, config.episode_indices)
    parquet_paths = tuple(
        path for path in parquet_paths
        if _episode_index_from_path(path) % config.num_shards == config.shard_index
    )

    details: list[dict[str, Any]] = []
    written_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.monotonic()
    total_count = len(parquet_paths)
    print("[episode-cache] configuration", flush=True)
    print(f"  dataset_path: {dataset_path}", flush=True)
    print(f"  cache_root: {cache_root}", flush=True)
    print(f"  model_path: {config.encoder_model_path or 'N/A'}", flush=True)
    print(
        f"  encoder: {config.encoder_kind} | video_backend: {config.video_backend} | "
        f"storage_dtype: {config.dtype}",
        flush=True,
    )
    progress_bar = tqdm(
        total=total_count,
        desc="Encoding",
        unit="ep",
        dynamic_ncols=True,
    )
    print(
        f"  video_keys ({len(config.video_keys)}): {', '.join(config.video_keys)}",
        flush=True,
    )
    print(
        f"  episodes: {total_count} | shard: {config.shard_index}/{config.num_shards} | "
        f"overwrite: {config.overwrite} | execute: {not config.dry_run}",
        flush=True,
    )

    for parquet_path in parquet_paths:
        episode_index = _episode_index_from_path(parquet_path)
        store_path = cache_root / f"ep_{episode_index:06d}.h5"
        detail: dict[str, Any] = {
            "episode_index": episode_index,
            "store_path": str(store_path),
        }

        if store_path.exists() and not config.overwrite:
            skipped_count += 1
            detail["status"] = "skipped_exists"
            details.append(detail)
            progress_bar.set_postfix(ep=f"{episode_index:06d}", status="skipped", refresh=False)
            progress_bar.update(1)
            continue

        if config.dry_run:
            detail["status"] = "planned"
            details.append(detail)
            progress_bar.set_postfix(ep=f"{episode_index:06d}", status="planned", refresh=False)
            progress_bar.update(1)
            continue

        try:
            episode_started_at = time.monotonic()
            row_count = _read_parquet_row_count(parquet_path)
            frame_indices = tuple(range(row_count))

            # Write to a temporary file in the same directory and atomically rename
            # to the final store path so that concurrent readers never see a partial
            # HDF5 file.
            tmp_path = store_path.with_suffix(".h5.tmp")
            with h5py.File(tmp_path, "w") as file:
                _write_metadata(
                    file,
                    config,
                    episode_index,
                    row_count,
                    source_episode_path=parquet_path,
                )
                _write_indices(file, row_count)
                _write_robot_and_language(file, dataset_path, episode_index, row_count)

                video_paths = {
                    video_key: _resolve_video_path(dataset_path, video_key, episode_index)
                    for video_key in config.video_keys
                }
                predecoded_frames: dict[str, list[Any]] = {}
                data_load_started_at = time.monotonic()
                load_frames = getattr(encoder, "load_video_frames", None)
                encode_frames = getattr(encoder, "encode_frames", None)
                if load_frames is not None and encode_frames is not None:
                    encoder._ensure_loaded()
                    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
                        futures = {
                            video_key: executor.submit(load_frames, video_path, frame_indices)
                            for video_key, video_path in video_paths.items()
                            if video_path.is_file()
                        }
                        predecoded_frames = {video_key: future.result() for video_key, future in futures.items()}
                data_load_seconds = time.monotonic() - data_load_started_at
                vae_encode_started_at = time.monotonic()
                for video_key, video_path in video_paths.items():
                    if not video_path.is_file():
                        detail.setdefault("warnings", []).append(f"missing video: {video_path}")
                        continue
                    latents = (
                        encode_frames(predecoded_frames[video_key])
                        if predecoded_frames else encoder.encode_video_clip(
                            video_path=video_path, frame_indices=frame_indices, video_key=video_key
                        )
                    )
                    latents = np.asarray(
                        latents,
                        dtype=np.float16 if config.dtype == "float16" else np.float32,
                    )
                    get_source_indices = getattr(encoder, "latent_source_frame_indices", None)
                    source_frame_indices = (
                        get_source_indices(row_count) if get_source_indices is not None else None
                    )
                    _write_video_latent(
                        file,
                        video_key,
                        latents,
                        source_frame_indices=source_frame_indices,
                    )
                vae_encode_seconds = time.monotonic() - vae_encode_started_at

                # Record the actual per-frame latent shape from encoded output rather
                # than relying on the config placeholder (e.g. ["D"]).
                if "latents" in locals():
                    file.attrs["latent_shape_per_frame"] = json.dumps(list(latents.shape[1:]))
                if getattr(encoder, "temporal_compression_factor", None) is not None:
                    file.attrs["temporal_compression_factor"] = encoder.temporal_compression_factor
                    file.attrs["source_frame_to_latent_policy"] = "causal_endpoint"
                    file.attrs["vae_input_height"] = encoder.height
                    file.attrs["vae_input_width"] = encoder.width
                    file.attrs["vae_preprocess_policy"] = WAN_VAE_PREPROCESS_POLICY
                    file.attrs["vae_latents_normalized"] = True

            tmp_path.replace(store_path)
            written_count += 1
            detail["status"] = "written"
            detail["latent_chunk_count"] = int(latents.shape[0])
            detail["data_load_seconds"] = round(data_load_seconds, 3)
            detail["vae_encode_seconds"] = round(vae_encode_seconds, 3)
            detail["write_seconds"] = round(time.monotonic() - episode_started_at - data_load_seconds - vae_encode_seconds, 3)
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            detail["status"] = "failed"
            detail["error"] = str(exc)
            progress_bar.write(f"[episode-cache] failed ep_{episode_index:06d}: {exc}")
            if store_path.exists():
                store_path.unlink()
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()

        details.append(detail)
        frame_to_chunk = (
            f"{row_count}->{detail['latent_chunk_count']}"
            if detail.get("latent_chunk_count") is not None
            else str(row_count) if "row_count" in locals() else "-"
        )
        progress_bar.set_postfix(
            ep=f"{episode_index:06d}",
            frames_chunks=frame_to_chunk,
            load=f"{detail.get('data_load_seconds', 0):.1f}s",
            vae=f"{detail.get('vae_encode_seconds', 0):.1f}s",
            skipped=skipped_count,
            failed=failed_count,
            refresh=False,
        )
        progress_bar.update(1)

    progress_bar.close()
    print(
        f"[episode-cache] finished written={written_count} skipped={skipped_count} "
        f"failed={failed_count} elapsed={(time.monotonic() - started_at) / 60:.1f} min",
        flush=True,
    )

    go_no_go = (
        "TBD: episode latent store write completed; validate before integration"
        if written_count > 0
        else (
            "TBD: episode latent store planned; execute and validate before integration"
            if not config.dry_run
            else "No-Go: no episode latent stores were planned"
        )
    )
    if failed_count > 0:
        go_no_go = "No-Go: some episode latent stores failed to build"

    return MoWAEpisodeLatentStoreBuildReport(
        dataset_path=str(dataset_path),
        cache_root=str(cache_root),
        episode_count=len(parquet_paths),
        written_count=written_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        dry_run=config.dry_run,
        details=tuple(details),
        go_no_go=go_no_go,
    )


def validate_mowa_episode_latent_store(
    cache_root: Path | str,
    video_keys: tuple[str, ...] | None = None,
) -> MoWAEpisodeLatentStoreValidationReport:
    """Validate all episode latent stores under ``cache_root``."""

    cache_root = Path(cache_root)
    store_paths = sorted(cache_root.glob("ep_*.h5"))

    checked_count = 0
    valid_count = 0
    invalid_count = 0
    missing_count = 0

    for store_path in store_paths:
        checked_count += 1
        try:
            store = MoWAEpisodeLatentStore(store_path)
            attrs = store.attrs
            required_attrs = {
                "episode_id",
                "latent_model",
                "latent_model_version",
                "latent_type",
                "latent_shape_per_frame",
                "flatten_policy",
                "dtype",
                "time_order",
            }
            missing_attrs = required_attrs - set(attrs.keys())
            if missing_attrs:
                invalid_count += 1
                continue

            keys = video_keys or store.list_video_keys()
            for video_key in keys:
                if video_key not in store.list_video_keys():
                    invalid_count += 1
                    break
                num_frames = store.num_latent_frames(video_key)
                valid = store.get_valid_mask(video_key)
                if valid.shape[0] != num_frames:
                    invalid_count += 1
                    break
            else:
                valid_count += 1
        except FileNotFoundError:
            missing_count += 1
        except Exception:  # noqa: BLE001
            invalid_count += 1

    all_ok = checked_count > 0 and invalid_count == 0 and missing_count == 0
    return MoWAEpisodeLatentStoreValidationReport(
        cache_root=str(cache_root),
        checked_count=checked_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        missing_count=missing_count,
        go_no_go=(
            "TBD: episode latent store validation passed"
            if all_ok
            else "No-Go: episode latent store validation failed"
        ),
    )


def _build_encoder_adapter(config: MoWAEpisodeLatentStoreConfig) -> MoWALatentEncoderAdapter:
    if config.encoder_kind == "wan2.2-vae":
        if config.encoder_model_path is None:
            raise ValueError("encoder_model_path is required for Wan2.2 VAE encoder.")
        return MoWAWanVaeEpisodeEncoderAdapter(
            model_path=config.encoder_model_path,
            encoder_name=config.encoder_name,
            encoder_version=config.encoder_version,
            latent_type=config.latent_type,
            latent_dim=config.latent_dim,
            flatten_policy=config.flatten_policy,
            video_backend=config.video_backend,
            vae_batch_size=config.vae_batch_size,
        )
    if config.encoder_kind == "fake":
        return MoWAFakeLatentEncoderAdapter(
            encoder_name=config.encoder_name,
            encoder_version=config.encoder_version,
            latent_dim=config.latent_dim,
        )
    raise NotImplementedError(f"Encoder kind {config.encoder_kind!r} not yet supported.")


def _write_metadata(
    file: h5py.File,
    config: MoWAEpisodeLatentStoreConfig,
    episode_index: int,
    row_count: int,
    *,
    source_episode_path: Path,
) -> None:
    file.attrs["episode_id"] = f"ep_{episode_index:06d}"
    file.attrs["episode_index"] = episode_index
    file.attrs["frame_count"] = row_count
    file.attrs["dataset_path"] = str(Path(config.dataset_path).resolve())
    file.attrs["source_episode_path"] = str(source_episode_path.resolve())
    file.attrs["latent_model"] = config.latent_model
    file.attrs["latent_model_version"] = config.latent_model_version
    file.attrs["latent_type"] = config.latent_type
    file.attrs["latent_shape_per_frame"] = json.dumps(config.latent_shape_per_frame)
    file.attrs["latent_dim"] = config.latent_dim
    file.attrs["flatten_policy"] = config.flatten_policy
    file.attrs["dtype"] = config.dtype
    file.attrs["obs_fps"] = config.obs_fps if isinstance(config.obs_fps, (int, float)) else _DATA_GATE
    file.attrs["action_hz"] = config.action_hz if isinstance(config.action_hz, (int, float)) else _DATA_GATE
    file.attrs["wam_hz"] = config.wam_hz if isinstance(config.wam_hz, (int, float)) else _DATA_GATE
    file.attrs["time_order"] = config.time_order
    file.attrs["downsample_policy"] = config.downsample_policy
    file.attrs["encoder_kind"] = config.encoder_kind
    file.attrs["encoder_name"] = config.encoder_name
    file.attrs["encoder_version"] = config.encoder_version
    file.attrs["encoder_model_path"] = str(config.encoder_model_path) if config.encoder_model_path is not None else ""
    file.attrs["video_backend"] = config.video_backend
    file.attrs["vae_batch_size"] = config.vae_batch_size


def _write_indices(file: h5py.File, row_count: int) -> None:
    frame_indices = np.arange(row_count, dtype=np.int64)
    file.create_dataset("/indices/frame_indices", data=frame_indices)


def _write_video_latent(
    file: h5py.File,
    video_key: str,
    latents: np.ndarray,
    *,
    source_frame_indices: np.ndarray | None = None,
) -> None:
    file.create_dataset(f"/latents/{video_key}", data=latents, compression="gzip")
    valid = np.ones(latents.shape[0], dtype=bool)
    file.create_dataset(f"/valid/{video_key}", data=valid)
    if source_frame_indices is not None:
        source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
        if source_frame_indices.shape != (latents.shape[0],):
            raise ValueError(
                "Wan source-frame mapping length must match temporal latent length: "
                f"{source_frame_indices.shape} != {(latents.shape[0],)}"
            )
        file.create_dataset(
            f"/indices/latent_source_frame_indices/{video_key}",
            data=source_frame_indices,
        )


def _write_robot_and_language(
    file: h5py.File,
    dataset_path: Path,
    episode_index: int,
    row_count: int,
) -> None:
    state, action = _read_state_and_action(dataset_path, episode_index, row_count)
    file.create_dataset("/robot/state", data=state)
    file.create_dataset("/robot/action", data=action)
    instruction = _read_instruction(dataset_path, episode_index)
    file.create_dataset("/language/instruction", data=instruction.encode("utf-8"))


def _read_state_and_action(
    dataset_path: Path,
    episode_index: int,
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA episode latent store builder requires pyarrow.") from exc

    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    table = pq.read_table(parquet_path)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[0] != row_count:
        state = state.reshape(row_count, -1)
    if action.ndim != 2 or action.shape[0] != row_count:
        action = action.reshape(row_count, -1)
    return state, action


def _read_instruction(dataset_path: Path, episode_index: int) -> str:
    """Read the language instruction for an episode.

    LeRobot-style datasets store the episode-to-task mapping in
    ``meta/episodes.jsonl`` and the task definitions in ``meta/tasks.jsonl``.
    Prefer ``episodes.jsonl`` because it maps ``episode_index`` directly to the
    list of task descriptions.
    """
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        try:
            with episodes_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if int(data.get("episode_index", -1)) == episode_index:
                        tasks = data.get("tasks")
                        if isinstance(tasks, list) and tasks:
                            return "; ".join(str(task) for task in tasks if task)
        except Exception:  # noqa: BLE001
            pass

    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.is_file():
        try:
            with tasks_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if int(data.get("episode_index", -1)) == episode_index:
                        return str(data.get("task", "TBD"))
        except Exception:  # noqa: BLE001
            pass
    return "TBD"


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
        raise RuntimeError("MoWA episode latent store builder requires pyarrow.") from exc
    return int(pq.ParquetFile(parquet_path).metadata.num_rows)


def _resolve_video_path(dataset_path: Path, video_key: str, episode_index: int) -> Path:
    relative_path = Path(f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4")
    return dataset_path / relative_path
