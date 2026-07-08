"""MoWA window manifest builder.

A window manifest is a per-sample slice catalogue that references episode-level
latent stores.  Rebuilding the manifest (e.g. with a different history window or
future horizon) does not require re-encoding visual latents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from starVLA.dataloader.mowa.episode_latent_store import MoWAEpisodeLatentStore
from starVLA.dataloader.mowa.schema import MoWAWindowConfig


@dataclass(frozen=True)
class MoWAWindowManifestEntry:
    """One training/evaluation window sample."""

    sample_id: str
    episode_id: str
    episode_latent_path: str
    task_name: str
    anchor_index: int
    anchor_timestamp: float
    video_key: str
    history_indices: tuple[int, ...]
    current_index: int
    future_indices: tuple[int, ...]
    robot_state_indices: tuple[int, ...]
    history_action_indices: tuple[int, ...]
    action_chunk_indices: tuple[int, ...]
    label_sidecar_path: str | None
    label_index: int
    history_seconds: float
    future_seconds: float
    wam_hz: float
    history_stride: int
    split: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "episode_latent_path": self.episode_latent_path,
            "task_name": self.task_name,
            "anchor_index": self.anchor_index,
            "anchor_timestamp": self.anchor_timestamp,
            "video_key": self.video_key,
            "history_indices": list(self.history_indices),
            "current_index": self.current_index,
            "future_indices": list(self.future_indices),
            "robot_state_indices": list(self.robot_state_indices),
            "history_action_indices": list(self.history_action_indices),
            "action_chunk_indices": list(self.action_chunk_indices),
            "label_sidecar_path": self.label_sidecar_path,
            "label_index": self.label_index,
            "history_seconds": self.history_seconds,
            "future_seconds": self.future_seconds,
            "wam_hz": self.wam_hz,
            "history_stride": self.history_stride,
            "split": self.split,
            "status": self.status,
        }


@dataclass(frozen=True)
class MoWAWindowManifestConfig:
    """Configuration for building a window manifest from episode latent stores."""

    cache_root: Path
    output_path: Path
    window_config: MoWAWindowConfig
    video_keys: tuple[str, ...] = (
        "observation.images.robot0_agentview_left",
    )
    history_stride: int = 1
    wam_hz: float = 4.0
    obs_fps: float | str = "Data Gate"
    action_hz: float | str = "Data Gate"
    split: str = "train"
    task_name: str = "TBD"
    label_sidecar_root: Path | None = None
    episode_indices: tuple[int, ...] | None = None
    allow_partial_windows: bool = False

    def validate(self) -> None:
        self.window_config.validate()
        if self.history_stride <= 0:
            raise ValueError("history_stride must be positive.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {self.split!r}.")


@dataclass(frozen=True)
class MoWAWindowManifestBuildReport:
    cache_root: str
    output_path: str
    episode_count: int
    window_count: int
    skipped_boundary_count: int
    skipped_missing_store_count: int
    go_no_go: str
    notes: tuple[str, ...] = (
        "Window manifest references episode latent stores; no visual re-encoding.",
        "History/future/stride parameters live in the manifest, not the cache.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_root": self.cache_root,
            "output_path": self.output_path,
            "episode_count": self.episode_count,
            "window_count": self.window_count,
            "skipped_boundary_count": self.skipped_boundary_count,
            "skipped_missing_store_count": self.skipped_missing_store_count,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_window_manifest(
    config: MoWAWindowManifestConfig,
) -> MoWAWindowManifestBuildReport:
    """Build a window manifest parquet from episode latent stores."""

    config.validate()
    cache_root = Path(config.cache_root)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store_paths = sorted(cache_root.glob("ep_*.h5"))
    store_paths = _filter_store_paths(store_paths, config.episode_indices)

    entries: list[MoWAWindowManifestEntry] = []
    skipped_boundary_count = 0
    skipped_missing_store_count = 0

    for store_path in store_paths:
        try:
            store = MoWAEpisodeLatentStore(store_path)
        except FileNotFoundError:
            skipped_missing_store_count += 1
            continue

        attrs = store.attrs
        episode_id = attrs.get("episode_id", store_path.stem)
        episode_index = int(attrs.get("episode_index", 0))
        frame_count = int(attrs.get("frame_count", store.num_frames(config.video_keys[0])))
        obs_fps = attrs.get("obs_fps", config.obs_fps)
        action_hz = attrs.get("action_hz", config.action_hz)

        for anchor_index in range(frame_count):
            window = _resolve_window_indices(
                anchor_index=anchor_index,
                frame_count=frame_count,
                window_config=config.window_config,
                history_stride=config.history_stride,
                allow_partial_windows=config.allow_partial_windows,
            )
            if window is None:
                skipped_boundary_count += 1
                continue

            history_indices, future_indices, action_chunk_indices = window
            current_index = anchor_index

            # Synchronous 1:1 indices for the first iteration; Hz-aware resampling
            # should be added once G0 temporal profile is measured.
            robot_state_indices = tuple(range(anchor_index, anchor_index + 1)) + history_indices
            robot_state_indices = tuple(sorted(set(robot_state_indices)))
            history_action_indices = history_indices

            for video_key in config.video_keys:
                if video_key not in store.list_video_keys():
                    continue
                entry = MoWAWindowManifestEntry(
                    sample_id=f"{episode_id}_a{anchor_index:06d}_{video_key}",
                    episode_id=episode_id,
                    episode_latent_path=str(store_path),
                    task_name=config.task_name,
                    anchor_index=anchor_index,
                    anchor_timestamp=_anchor_timestamp(anchor_index, obs_fps),
                    video_key=video_key,
                    history_indices=history_indices,
                    current_index=current_index,
                    future_indices=future_indices,
                    robot_state_indices=robot_state_indices,
                    history_action_indices=history_action_indices,
                    action_chunk_indices=action_chunk_indices,
                    label_sidecar_path=_label_sidecar_path(config.label_sidecar_root, episode_id),
                    label_index=anchor_index,
                    history_seconds=_seconds(len(history_indices), config.wam_hz),
                    future_seconds=_seconds(len(future_indices), config.wam_hz),
                    wam_hz=config.wam_hz,
                    history_stride=config.history_stride,
                    split=config.split,
                    status="target",
                )
                entries.append(entry)

    _write_manifest_parquet(output_path, entries)

    go_no_go = (
        "TBD: window manifest built; validate samples before training"
        if entries
        else "No-Go: no window samples were generated"
    )

    return MoWAWindowManifestBuildReport(
        cache_root=str(cache_root),
        output_path=str(output_path),
        episode_count=len(store_paths),
        window_count=len(entries),
        skipped_boundary_count=skipped_boundary_count,
        skipped_missing_store_count=skipped_missing_store_count,
        go_no_go=go_no_go,
    )


def load_mowa_window_manifest(path: Path | str) -> tuple[MoWAWindowManifestEntry, ...]:
    """Load a window manifest parquet into immutable entries."""

    table = pq.read_table(str(path))
    entries = []
    for row in table.to_pylist():
        entries.append(
            MoWAWindowManifestEntry(
                sample_id=row["sample_id"],
                episode_id=row["episode_id"],
                episode_latent_path=row["episode_latent_path"],
                task_name=row["task_name"],
                anchor_index=int(row["anchor_index"]),
                anchor_timestamp=float(row["anchor_timestamp"]),
                video_key=row["video_key"],
                history_indices=tuple(int(value) for value in row["history_indices"]),
                current_index=int(row["current_index"]),
                future_indices=tuple(int(value) for value in row["future_indices"]),
                robot_state_indices=tuple(int(value) for value in row["robot_state_indices"]),
                history_action_indices=tuple(int(value) for value in row["history_action_indices"]),
                action_chunk_indices=tuple(int(value) for value in row["action_chunk_indices"]),
                label_sidecar_path=row.get("label_sidecar_path"),
                label_index=int(row["label_index"]),
                history_seconds=float(row["history_seconds"]),
                future_seconds=float(row["future_seconds"]),
                wam_hz=float(row["wam_hz"]),
                history_stride=int(row["history_stride"]),
                split=row["split"],
                status=row["status"],
            )
        )
    return tuple(entries)


def _resolve_window_indices(
    *,
    anchor_index: int,
    frame_count: int,
    window_config: MoWAWindowConfig,
    history_stride: int,
    allow_partial_windows: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    if anchor_index < 0 or anchor_index >= frame_count:
        return None

    # History: inclusive of anchor by default; stride-aware.
    hist_start = anchor_index - (window_config.history_steps - 1) * history_stride
    if hist_start < 0:
        if not allow_partial_windows:
            return None
        hist_start = 0
    history_indices = tuple(
        idx
        for idx in range(hist_start, anchor_index + 1, history_stride)
        if idx <= anchor_index
    )
    if not history_indices:
        return None

    # Future: strictly after anchor.
    future_end = min(
        frame_count,
        anchor_index + 1 + window_config.future_steps,
    )
    future_indices = tuple(range(anchor_index + 1, future_end))
    if not future_indices:
        return None
    if not allow_partial_windows and len(future_indices) < window_config.future_steps:
        return None

    # Action chunk: starting at anchor, inclusive.
    action_end = min(frame_count, anchor_index + window_config.action_chunk_steps)
    action_chunk_indices = tuple(range(anchor_index, action_end))
    if not action_chunk_indices:
        return None
    if not allow_partial_windows and len(action_chunk_indices) < window_config.action_chunk_steps:
        return None

    return history_indices, future_indices, action_chunk_indices


def _filter_store_paths(
    store_paths: tuple[Path, ...],
    episode_indices: tuple[int, ...] | None,
) -> tuple[Path, ...]:
    if episode_indices is None:
        return store_paths
    allowed = set(episode_indices)
    return tuple(
        store_path
        for store_path in store_paths
        if _episode_index_from_store_path(store_path) in allowed
    )


def _episode_index_from_store_path(store_path: Path) -> int:
    stem = store_path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unable to parse episode index from {store_path}") from exc


def _anchor_timestamp(anchor_index: int, obs_fps: Any) -> float:
    fps = float(obs_fps) if isinstance(obs_fps, (int, float)) else 1.0
    if fps <= 0:
        fps = 1.0
    return anchor_index / fps


def _seconds(steps: int, wam_hz: float) -> float:
    return steps / wam_hz if wam_hz > 0 else 0.0


def _label_sidecar_path(label_sidecar_root: Path | None, episode_id: str) -> str | None:
    if label_sidecar_root is None:
        return None
    return str(label_sidecar_root / f"{episode_id}.jsonl")


def _write_manifest_parquet(
    output_path: Path,
    entries: list[MoWAWindowManifestEntry],
) -> None:
    if not entries:
        # Write an empty table with the expected schema.
        schema = pa.schema(
            [
                ("sample_id", pa.string()),
                ("episode_id", pa.string()),
                ("episode_latent_path", pa.string()),
                ("task_name", pa.string()),
                ("anchor_index", pa.int64()),
                ("anchor_timestamp", pa.float64()),
                ("video_key", pa.string()),
                ("history_indices", pa.list_(pa.int64())),
                ("current_index", pa.int64()),
                ("future_indices", pa.list_(pa.int64())),
                ("robot_state_indices", pa.list_(pa.int64())),
                ("history_action_indices", pa.list_(pa.int64())),
                ("action_chunk_indices", pa.list_(pa.int64())),
                ("label_sidecar_path", pa.string()),
                ("label_index", pa.int64()),
                ("history_seconds", pa.float64()),
                ("future_seconds", pa.float64()),
                ("wam_hz", pa.float64()),
                ("history_stride", pa.int64()),
                ("split", pa.string()),
                ("status", pa.string()),
            ]
        )
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        rows = [entry.to_dict() for entry in entries]
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
