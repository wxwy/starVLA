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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import h5py
import numpy as np
import torch
import torch.nn as nn


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


@dataclass(frozen=True)
class MoWAWanVaeEpisodeEncoderAdapter:
    """Encode a full episode with Wan2.2 VAE and store per-frame visual latents.

    The encoder processes frames in small batches to keep GPU memory bounded.
    Output shape depends on ``latent_type``:

    - ``vae_spatial``: ``[T, C, h, w]`` — raw VAE spatial latent per frame.
    - ``pooled_vector``: ``[T, D]`` — mean-pooled per-frame vector, optionally
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
    height: int = 480
    width: int = 832
    _projection_seed: int = 42

    def __post_init__(self) -> None:
        if self.latent_type not in {"vae_spatial", "pooled_vector"}:
            raise ValueError(f"Unsupported latent_type: {self.latent_type!r}.")
        if self.vae_batch_size <= 0:
            raise ValueError(f"vae_batch_size must be positive, got {self.vae_batch_size}.")
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

        from PIL import Image

        frames = self._load_frames(video_path, frame_indices)
        pil_frames = [Image.fromarray(frame) for frame in frames]

        per_frame_latents: list[torch.Tensor] = []
        for start in range(0, len(pil_frames), self.vae_batch_size):
            batch = pil_frames[start : start + self.vae_batch_size]
            latent = self._encode_frame_batch(batch)
            # latent: [B, C, 1, h, w] for per-frame encoding or [1, C, B, h, w] for clip encoding.
            per_frame_latents.extend(self._split_batch(latent))

        full_latent = torch.stack(per_frame_latents, dim=0)  # [T, C, h, w]

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
        self._vae = AutoencoderKLWan.from_pretrained(
            str(model_path),
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)
        self._video_processor = VideoProcessor(vae_scale_factor=2 ** len(self._vae.temperal_downsample))
        self._torch_dtype = dtype

    def _load_frames(self, source_path: Path, frame_indices: tuple[int, ...]) -> list[Any]:
        if self.video_backend == "decord":
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
        if self.video_backend != "opencv":
            raise NotImplementedError(
                f"Unsupported video_backend={self.video_backend!r}; only 'opencv' and 'decord' are available."
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("MoWA Wan encoder requires opencv-python.") from exc

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source_path}")
        frames: list[Any] = []
        try:
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"Unable to read frame at index {frame_index} from {source_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def _encode_frame_batch(self, pil_frames: list[Any]) -> torch.Tensor:
        assert self._vae is not None and self._video_processor is not None and self._torch_dtype is not None
        video_tensor = self._video_processor.preprocess_video(
            pil_frames,
            height=self.height,
            width=self.width,
        )
        device = next(self._vae.parameters()).device
        video_tensor = video_tensor.to(device=device, dtype=self._torch_dtype)
        with torch.inference_mode():
            latents = self._vae.encode(video_tensor).latent_dist.sample()
        return latents.to(dtype=torch.float32)

    def _split_batch(self, latent: torch.Tensor) -> list[torch.Tensor]:
        """Split a VAE batch into per-frame latents.

        This first version supports only per-frame encoding (vae_batch_size=1),
        which avoids temporal downsampling ambiguities across frame batches.
        Expected shape: ``[1, C, 1, h, w]``.
        """
        if latent.dim() != 5:
            raise ValueError(f"Expected 5D VAE latent, got {latent.dim()}D with shape {tuple(latent.shape)}.")
        if latent.shape[0] != 1 or latent.shape[2] != 1:
            raise NotImplementedError(
                "Only vae_batch_size=1 per-frame encoding is supported in this version. "
                f"Got latent shape {tuple(latent.shape)}. Set vae_batch_size=1."
            )
        return [latent.squeeze(0).squeeze(1)]  # [C, h, w]

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

    def validate(self) -> None:
        if not self.video_keys:
            raise ValueError("video_keys must be non-empty.")
        if self.time_order != _TIME_ORDER:
            raise ValueError(f"time_order must be {_TIME_ORDER!r}, got {self.time_order!r}.")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
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
            return int(dataset.shape[0])

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
                array = dataset[list(frame_indices), ...]
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

    details: list[dict[str, Any]] = []
    written_count = 0
    skipped_count = 0
    failed_count = 0

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
            continue

        if config.dry_run:
            detail["status"] = "planned"
            details.append(detail)
            continue

        try:
            row_count = _read_parquet_row_count(parquet_path)
            frame_indices = tuple(range(row_count))

            with h5py.File(store_path, "w") as file:
                _write_metadata(file, config, episode_index, row_count)
                _write_indices(file, row_count)
                _write_robot_and_language(file, dataset_path, episode_index, row_count)

                for video_key in config.video_keys:
                    video_path = _resolve_video_path(dataset_path, video_key, episode_index)
                    if not video_path.is_file():
                        detail.setdefault("warnings", []).append(f"missing video: {video_path}")
                        continue
                    latents = encoder.encode_video_clip(
                        video_path=video_path,
                        frame_indices=frame_indices,
                        video_key=video_key,
                    )
                    _write_video_latent(file, video_key, latents)

            written_count += 1
            detail["status"] = "written"
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            detail["status"] = "failed"
            detail["error"] = str(exc)
            if store_path.exists():
                store_path.unlink()

        details.append(detail)

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
                num_frames = store.num_frames(video_key)
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
) -> None:
    file.attrs["episode_id"] = f"ep_{episode_index:06d}"
    file.attrs["episode_index"] = episode_index
    file.attrs["frame_count"] = row_count
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


def _write_video_latent(file: h5py.File, video_key: str, latents: np.ndarray) -> None:
    file.create_dataset(f"/latents/{video_key}", data=latents, compression="gzip")
    valid = np.ones(latents.shape[0], dtype=bool)
    file.create_dataset(f"/valid/{video_key}", data=valid)


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
    if state.shape != (row_count, -1):
        state = state.reshape(row_count, -1)
    if action.shape != (row_count, -1):
        action = action.reshape(row_count, -1)
    return state, action


def _read_instruction(dataset_path: Path, episode_index: int) -> str:
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
