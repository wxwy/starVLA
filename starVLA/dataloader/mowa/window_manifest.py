"""MoWA window manifest builder.

A window manifest is a per-sample slice catalogue that references episode-level
latent stores.  Rebuilding the manifest (e.g. with a different history window or
future horizon) does not require re-encoding visual latents.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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
    anchor_video_key: str
    history_indices: tuple[int, ...]
    current_index: int
    future_indices: tuple[int, ...]
    history_state_indices: tuple[int, ...]
    current_state_index: int
    future_state_indices: tuple[int, ...]
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
    source_dataset_path: str | None = None
    source_episode_path: str | None = None
    history_valid_mask: tuple[bool, ...] = ()
    future_valid_mask: tuple[bool, ...] = ()
    action_valid_mask: tuple[bool, ...] = ()
    history_action_valid_mask: tuple[bool, ...] = ()
    history_state_valid_mask: tuple[bool, ...] = ()
    future_state_valid_mask: tuple[bool, ...] = ()
    action_representation: str = "absolute"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "episode_latent_path": self.episode_latent_path,
            "task_name": self.task_name,
            "anchor_index": self.anchor_index,
            "anchor_timestamp": self.anchor_timestamp,
            "anchor_video_key": self.anchor_video_key,
            "history_indices": list(self.history_indices),
            "current_index": self.current_index,
            "future_indices": list(self.future_indices),
            "history_state_indices": list(self.history_state_indices),
            "current_state_index": self.current_state_index,
            "future_state_indices": list(self.future_state_indices),
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
            "source_dataset_path": self.source_dataset_path,
            "source_episode_path": self.source_episode_path,
            "history_valid_mask": list(self.history_valid_mask),
            "future_valid_mask": list(self.future_valid_mask),
            "action_valid_mask": list(self.action_valid_mask),
            "history_action_valid_mask": list(self.history_action_valid_mask),
            "history_state_valid_mask": list(self.history_state_valid_mask),
            "future_state_valid_mask": list(self.future_state_valid_mask),
            "action_representation": self.action_representation,
        }


@dataclass(frozen=True)
class MoWAWindowManifestConfig:
    """Configuration for building a window manifest from episode latent stores."""

    cache_root: Path
    output_path: Path
    window_config: MoWAWindowConfig
    anchor_video_key: str | None = None
    # Kept only so existing Python callers can migrate without changing window
    # semantics.  New callers must use ``anchor_video_key``.
    video_keys: tuple[str, ...] = ()
    history_stride: int = 1
    wam_hz: float | None = None
    obs_fps: float | str = "Data Gate"
    action_hz: float | str = "Data Gate"
    split: str = "train"
    task_name: str = "TBD"
    label_sidecar_root: Path | None = None
    episode_indices: tuple[int, ...] | None = None
    allow_partial_windows: bool = False
    boundary_padding: bool = True
    action_representation: str = "absolute"
    recursive_cache_search: bool = False

    def validate(self) -> None:
        self.window_config.validate()
        if self.history_stride <= 0:
            raise ValueError("history_stride must be positive.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {self.split!r}.")
        if self.action_representation not in {"absolute", "delta"}:
            raise ValueError("action_representation must be 'absolute' or 'delta'.")
        if self.wam_hz is not None and self.wam_hz <= 0:
            raise ValueError("wam_hz must be positive when set.")
        if not self.resolved_anchor_video_key:
            raise ValueError("anchor_video_key must be non-empty.")

    @property
    def resolved_anchor_video_key(self) -> str:
        if self.anchor_video_key:
            return self.anchor_video_key
        if self.video_keys:
            return self.video_keys[0]
        return "observation.images.robot0_agentview_left"


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


def default_mowa_window_manifest_dir(cache_root: Path | str) -> Path:
    cache_root = Path(cache_root)
    return cache_root.parent / f"{cache_root.name}_manifests"


def default_mowa_window_manifest_path(
    cache_root: Path | str,
    filename: str = "window_manifest.parquet",
) -> Path:
    return default_mowa_window_manifest_dir(cache_root) / filename


def build_mowa_window_manifest(
    config: MoWAWindowManifestConfig,
) -> MoWAWindowManifestBuildReport:
    """Build a window manifest parquet from episode latent stores."""

    config.validate()
    cache_root = Path(config.cache_root)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store_paths = sorted(
        cache_root.rglob("ep_*.h5") if config.recursive_cache_search else cache_root.glob("ep_*.h5")
    )
    store_paths = _filter_store_paths(store_paths, config.episode_indices)

    entries: list[MoWAWindowManifestEntry] = []
    skipped_boundary_count = 0
    skipped_missing_store_count = 0
    print("[window-manifest] configuration", flush=True)
    print(f"  cache_root: {cache_root}", flush=True)
    print(f"  output_path: {output_path}", flush=True)
    print(
        f"  anchor_video_key: {config.resolved_anchor_video_key} | "
        f"history/future: {config.window_config.history_steps}/{config.window_config.future_steps} | "
        f"action_chunk_steps: {config.window_config.resolved_action_chunk_steps}",
        flush=True,
    )
    print(
        f"  recursive_cache_search: {config.recursive_cache_search} | stores: {len(store_paths)}",
        flush=True,
    )
    progress_bar = tqdm(total=len(store_paths), desc="Building manifest", unit="ep", dynamic_ncols=True)

    for store_path in store_paths:
        episode_windows_before = len(entries)
        try:
            store = MoWAEpisodeLatentStore(store_path)
        except FileNotFoundError:
            skipped_missing_store_count += 1
            progress_bar.set_postfix(
                ep=store_path.stem,
                windows=len(entries),
                boundary_skips=skipped_boundary_count,
                missing=skipped_missing_store_count,
                refresh=False,
            )
            progress_bar.update(1)
            continue

        attrs = store.attrs
        episode_id = attrs.get("episode_id", store_path.stem)
        episode_index = int(attrs.get("episode_index", 0))
        anchor_video_key = config.resolved_anchor_video_key
        if anchor_video_key not in store.list_video_keys():
            skipped_missing_store_count += 1
            progress_bar.set_postfix(
                ep=episode_id,
                windows=len(entries),
                boundary_skips=skipped_boundary_count,
                missing=skipped_missing_store_count,
                refresh=False,
            )
            progress_bar.update(1)
            continue
        frame_count = int(attrs.get("frame_count", store.num_frames(anchor_video_key)))
        source_dataset_path = attrs.get("dataset_path")
        source_episode_path = attrs.get("source_episode_path")

        source_indices = store.get_latent_source_frame_indices(anchor_video_key)
        is_wan_regular_grid = (
            source_indices is not None
            and int(attrs.get("temporal_compression_factor", 1)) == 4
            and len(source_indices) >= 2
            and int(source_indices[0]) == 0
        )
        try:
            obs_fps = _resolve_obs_fps(
                attrs.get("obs_fps", config.obs_fps),
                source_dataset_path,
                source_episode_path,
            )
        except ValueError:
            if config.wam_hz is None or not is_wan_regular_grid:
                raise
            obs_fps = config.wam_hz * int(attrs.get("temporal_compression_factor", 1))
        wam_hz = _resolve_wam_hz(
            configured_wam_hz=config.wam_hz,
            obs_fps=obs_fps,
            temporal_compression_factor=int(attrs.get("temporal_compression_factor", 1)),
            is_wan_regular_grid=is_wan_regular_grid,
        )
        # Wan's physical z_0 is the causal one-frame chunk.  It is retained in
        # HDF5 but excluded from the trainable, uniform 4-frame latent grid.
        if is_wan_regular_grid:
            latent_anchor_indices = range(1, len(source_indices))
        else:
            latent_anchor_indices = range(frame_count)

        for anchor_index in latent_anchor_indices:
            window = _resolve_window_indices(
                anchor_index=anchor_index,
                frame_count=(len(source_indices) if is_wan_regular_grid else frame_count),
                window_config=config.window_config,
                history_stride=config.history_stride,
                allow_partial_windows=config.allow_partial_windows,
                boundary_padding=config.boundary_padding and is_wan_regular_grid,
            )
            if window is None:
                skipped_boundary_count += 1
                continue

            history_indices, future_indices, history_valid_mask, future_valid_mask = window
            current_index = anchor_index
            source_anchor_index = int(source_indices[anchor_index]) if is_wan_regular_grid else anchor_index
            if is_wan_regular_grid:
                history_state_indices = tuple(int(source_indices[index]) for index in history_indices)
                current_state_index = source_anchor_index
                future_state_indices = tuple(int(source_indices[index]) for index in future_indices)
                history_action_indices, history_action_valid_mask = _padded_history_action_indices(
                    source_anchor_index,
                    config.window_config.resolved_history_action_steps,
                )
                action_chunk_indices, action_valid_mask = _padded_action_indices(
                    source_anchor_index,
                    frame_count,
                    config.window_config.resolved_action_chunk_steps,
                )
            else:
                history_state_indices = history_indices
                current_state_index = anchor_index
                future_state_indices = future_indices
                history_action_indices = history_indices
                history_action_valid_mask = ()
                action_chunk_indices = tuple(
                    range(anchor_index, min(frame_count, anchor_index + config.window_config.resolved_action_chunk_steps))
                )
                action_valid_mask = (True,) * len(action_chunk_indices)

            entry = MoWAWindowManifestEntry(
                sample_id=_sample_id(store_path, cache_root, episode_id, source_anchor_index),
                episode_id=episode_id,
                episode_latent_path=str(store_path),
                task_name=config.task_name,
                anchor_index=source_anchor_index,
                anchor_timestamp=_anchor_timestamp(source_anchor_index, obs_fps),
                anchor_video_key=anchor_video_key,
                history_indices=history_indices,
                current_index=current_index,
                future_indices=future_indices,
                history_state_indices=history_state_indices,
                current_state_index=current_state_index,
                future_state_indices=future_state_indices,
                history_action_indices=history_action_indices,
                action_chunk_indices=action_chunk_indices,
                label_sidecar_path=_label_sidecar_path(config.label_sidecar_root, episode_id),
                label_index=source_anchor_index,
                history_seconds=_seconds(len(history_indices), wam_hz),
                future_seconds=_seconds(len(future_indices), wam_hz),
                wam_hz=wam_hz,
                history_stride=config.history_stride,
                split=config.split,
                status="target",
                source_dataset_path=str(source_dataset_path) if source_dataset_path else None,
                source_episode_path=str(source_episode_path) if source_episode_path else None,
                history_valid_mask=history_valid_mask if is_wan_regular_grid else (),
                future_valid_mask=future_valid_mask if is_wan_regular_grid else (),
                action_valid_mask=action_valid_mask if is_wan_regular_grid else (),
                history_action_valid_mask=history_action_valid_mask,
                history_state_valid_mask=history_valid_mask if is_wan_regular_grid else (),
                future_state_valid_mask=future_valid_mask if is_wan_regular_grid else (),
                action_representation=config.action_representation,
            )
            entries.append(entry)

        progress_bar.set_postfix(
            ep=episode_id,
            windows=len(entries),
            added=len(entries) - episode_windows_before,
            boundary_skips=skipped_boundary_count,
            missing=skipped_missing_store_count,
            refresh=False,
        )
        progress_bar.update(1)

    progress_bar.close()

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


def load_mowa_window_manifest(
    path: Path | str,
    *,
    episode_latent_path_prefix: Path | str | None = None,
) -> tuple[MoWAWindowManifestEntry, ...]:
    """Load a window manifest parquet into immutable entries.

    ``episode_latent_path_prefix`` allows a task-local dataloader to read only
    its own rows from a global manifest, before constructing Python entries.
    """

    return tuple(
        load_mowa_window_manifest_table(
            path,
            episode_latent_path_prefix=episode_latent_path_prefix,
        )
    )


class MoWAWindowManifestTable(Sequence[MoWAWindowManifestEntry]):
    """列式持有 manifest，仅在索引访问时解析单行 entry。"""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _manifest_entry_from_row(self._table.slice(index, 1).to_pylist()[0])

    def sample_keys(self) -> tuple[tuple[int, int, str], ...]:
        """从必要列构造轻量查找 key，不展开完整窗口字段。"""

        episode_ids = self._table.column("episode_id").to_pylist()
        anchor_indices = self._table.column("anchor_index").to_pylist()
        anchor_key_column = "anchor_video_key" if "anchor_video_key" in self._table.column_names else "video_key"
        anchor_video_keys = self._table.column(anchor_key_column).to_pylist()
        return tuple(
            (
                int(str(episode_id).split("_")[-1]),
                int(anchor_index),
                str(anchor_video_key),
            )
            for episode_id, anchor_index, anchor_video_key in zip(
                episode_ids,
                anchor_indices,
                anchor_video_keys,
            )
        )

    def iter_sampling_rows(self):
        """只展开 sampler 分类所需列，避免解析完整 entry。"""

        column_names = (
            "task_name",
            "episode_id",
            "anchor_index",
            "history_indices",
            "history_valid_mask",
            "history_action_valid_mask",
            "future_valid_mask",
            "action_valid_mask",
        )
        sampling_table = self._table.select(column_names)
        for batch in sampling_table.to_batches(max_chunksize=65_536):
            columns = [batch.column(index).to_pylist() for index in range(len(column_names))]
            for row_index in range(batch.num_rows):
                yield tuple(column[row_index] for column in columns)


def load_mowa_window_manifest_table(
    path: Path | str,
    *,
    episode_latent_path_prefix: Path | str | None = None,
) -> MoWAWindowManifestTable:
    """读取列式 manifest，避免启动时创建全部 Python entry。"""

    if episode_latent_path_prefix is None:
        table = pq.read_table(str(path))
    else:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="parquet")
        raw_prefix = str(Path(episode_latent_path_prefix)).rstrip("/")
        resolved_prefix = str(Path(episode_latent_path_prefix).resolve()).rstrip("/")
        table = dataset.to_table(
            filter=pc.starts_with(ds.field("episode_latent_path"), raw_prefix)
        )
        if table.num_rows == 0 and resolved_prefix != raw_prefix:
            table = dataset.to_table(
                filter=pc.starts_with(ds.field("episode_latent_path"), resolved_prefix)
            )
    return MoWAWindowManifestTable(table)


def _manifest_entry_from_row(row: dict[str, Any]) -> MoWAWindowManifestEntry:
    return MoWAWindowManifestEntry(
                sample_id=row["sample_id"],
                episode_id=row["episode_id"],
                episode_latent_path=row["episode_latent_path"],
                task_name=row["task_name"],
                anchor_index=int(row["anchor_index"]),
                anchor_timestamp=float(row["anchor_timestamp"]),
                anchor_video_key=row.get("anchor_video_key") or row["video_key"],
                history_indices=tuple(int(value) for value in row["history_indices"]),
                current_index=int(row["current_index"]),
                future_indices=tuple(int(value) for value in row["future_indices"]),
                history_state_indices=tuple(
                    int(value)
                    for value in (
                        row["history_state_indices"]
                        if "history_state_indices" in row
                        else row["robot_state_indices"]
                    )
                ),
                current_state_index=int(row.get("current_state_index", row["anchor_index"])),
                future_state_indices=tuple(
                    int(value)
                    for value in (
                        row["future_state_indices"]
                        if "future_state_indices" in row
                        else row["future_indices"]
                    )
                ),
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
                source_dataset_path=row.get("source_dataset_path"),
                source_episode_path=row.get("source_episode_path"),
                history_valid_mask=tuple(bool(value) for value in row.get("history_valid_mask") or ()),
                future_valid_mask=tuple(bool(value) for value in row.get("future_valid_mask") or ()),
                action_valid_mask=tuple(bool(value) for value in row.get("action_valid_mask") or ()),
                history_action_valid_mask=tuple(bool(value) for value in row.get("history_action_valid_mask") or ()),
                history_state_valid_mask=tuple(
                    bool(value) for value in (row.get("history_state_valid_mask") or row.get("history_valid_mask") or ())
                ),
                future_state_valid_mask=tuple(
                    bool(value) for value in (row.get("future_state_valid_mask") or row.get("future_valid_mask") or ())
                ),
                action_representation=row.get("action_representation") or "absolute",
            )


def slice_mowa_window_manifest_entry(
    entry: MoWAWindowManifestEntry,
    *,
    history_steps: int | None = None,
    future_steps: int | None = None,
    action_chunk_steps: int | None = None,
) -> MoWAWindowManifestEntry:
    """裁切最大窗口 manifest，供更短的训练窗口复用。

    history 始终取紧邻 current 的末尾，future/action 始终从 current 后的
    第一个位置开始取。请求的长度不能超过 manifest 已提供的最大长度。
    """

    resolved_history_steps = len(entry.history_indices) if history_steps is None else int(history_steps)
    resolved_future_steps = len(entry.future_indices) if future_steps is None else int(future_steps)
    resolved_action_chunk_steps = (
        len(entry.action_chunk_indices)
        if action_chunk_steps is None
        else int(action_chunk_steps)
    )
    if resolved_history_steps < 0 or resolved_future_steps <= 0 or resolved_action_chunk_steps <= 0:
        raise ValueError("history_steps must be non-negative; future/action steps must be positive.")
    available = {
        "history": len(entry.history_indices),
        "future": len(entry.future_indices),
        "history_state": len(entry.history_state_indices),
        "future_state": len(entry.future_state_indices),
        "history_action": len(entry.history_action_indices),
        "action": len(entry.action_chunk_indices),
    }
    required_history_actions = min(
        len(entry.history_action_indices),
        resolved_history_steps * max(1, len(entry.history_action_indices) // max(1, len(entry.history_indices))),
    )
    if (
        resolved_history_steps > available["history"]
        or resolved_history_steps > available["history_state"]
        or resolved_future_steps > available["future"]
        or resolved_future_steps > available["future_state"]
        or resolved_action_chunk_steps > available["action"]
    ):
        raise ValueError(
            "Requested MoWA window exceeds manifest capacity: "
            f"requested H/F/A={resolved_history_steps}/{resolved_future_steps}/{resolved_action_chunk_steps}, "
            f"available H/F/A={available['history']}/{available['future']}/{available['action']}."
        )

    def _tail(values: tuple[Any, ...], count: int) -> tuple[Any, ...]:
        return values[-count:] if count else ()

    return replace(
        entry,
        history_indices=_tail(entry.history_indices, resolved_history_steps),
        future_indices=entry.future_indices[:resolved_future_steps],
        history_state_indices=_tail(entry.history_state_indices, resolved_history_steps),
        future_state_indices=entry.future_state_indices[:resolved_future_steps],
        history_action_indices=_tail(entry.history_action_indices, required_history_actions),
        action_chunk_indices=entry.action_chunk_indices[:resolved_action_chunk_steps],
        history_seconds=resolved_history_steps / entry.wam_hz if entry.wam_hz else 0.0,
        future_seconds=resolved_future_steps / entry.wam_hz if entry.wam_hz else 0.0,
        history_valid_mask=_tail(entry.history_valid_mask, resolved_history_steps),
        future_valid_mask=entry.future_valid_mask[:resolved_future_steps],
        action_valid_mask=entry.action_valid_mask[:resolved_action_chunk_steps],
        history_action_valid_mask=_tail(entry.history_action_valid_mask, required_history_actions),
        history_state_valid_mask=_tail(entry.history_state_valid_mask, resolved_history_steps),
        future_state_valid_mask=entry.future_state_valid_mask[:resolved_future_steps],
    )


def _resolve_window_indices(
    *,
    anchor_index: int,
    frame_count: int,
    window_config: MoWAWindowConfig,
    history_stride: int,
    allow_partial_windows: bool,
    boundary_padding: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...], tuple[bool, ...]] | None:
    if anchor_index < 0 or anchor_index >= frame_count:
        return None

    if boundary_padding:
        history_raw = tuple(anchor_index - history_stride * offset for offset in range(window_config.history_steps, 0, -1))
        history_valid_mask = tuple(index >= 1 for index in history_raw)
        history_indices = tuple(max(1, index) for index in history_raw)
        future_raw = tuple(anchor_index + offset for offset in range(1, window_config.future_steps + 1))
        future_valid_mask = tuple(index < frame_count for index in future_raw)
        future_indices = tuple(min(frame_count - 1, index) for index in future_raw)
        return history_indices, future_indices, history_valid_mask, future_valid_mask

    # Legacy frame-grid behavior.
    hist_start = anchor_index - (window_config.history_steps - 1) * history_stride
    if hist_start < 0:
        if not allow_partial_windows:
            return None
        hist_start = 0
    history_indices = tuple(range(hist_start, anchor_index + 1, history_stride))
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

    action_end = min(frame_count, anchor_index + window_config.resolved_action_chunk_steps)
    if not allow_partial_windows and action_end - anchor_index < window_config.resolved_action_chunk_steps:
        return None

    return history_indices, future_indices, (True,) * len(history_indices), (True,) * len(future_indices)


def _padded_action_indices(anchor_index: int, frame_count: int, steps: int) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Return future raw actions, right-padded at terminal with a validity mask."""
    raw_indices = tuple(anchor_index + offset for offset in range(1, steps + 1))
    valid_mask = tuple(index < frame_count for index in raw_indices)
    return tuple(min(frame_count - 1, index) for index in raw_indices), valid_mask


def _padded_history_action_indices(anchor_index: int, steps: int) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Return contiguous past raw actions, left-padded at episode start."""
    raw_indices = tuple(anchor_index - steps + 1 + offset for offset in range(steps))
    valid_mask = tuple(index >= 0 for index in raw_indices)
    return tuple(max(0, index) for index in raw_indices), valid_mask


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


def _sample_id(store_path: Path, cache_root: Path, episode_id: str, anchor_index: int) -> str:
    """Use the relative store path so a global manifest has no cross-task ID collision."""
    try:
        scope = store_path.parent.relative_to(cache_root).as_posix().replace("/", "__")
    except ValueError:
        scope = store_path.parent.name
    prefix = f"{scope}__" if scope and scope != "." else ""
    return f"{prefix}{episode_id}_a{anchor_index:06d}"


def _anchor_timestamp(anchor_index: int, obs_fps: Any) -> float:
    fps = float(obs_fps) if isinstance(obs_fps, (int, float)) else 1.0
    if fps <= 0:
        fps = 1.0
    return anchor_index / fps


def _seconds(steps: int, wam_hz: float) -> float:
    return steps / wam_hz if wam_hz > 0 else 0.0


def _resolve_obs_fps(
    stored_obs_fps: Any,
    source_dataset_path: Any,
    source_episode_path: Any,
) -> float:
    """Resolve original observation frequency from cache metadata or source data."""
    if isinstance(stored_obs_fps, (int, float)) and stored_obs_fps > 0:
        return float(stored_obs_fps)

    if source_dataset_path:
        info_path = Path(str(source_dataset_path)) / "meta" / "info.json"
        if info_path.is_file():
            try:
                fps = json.loads(info_path.read_text(encoding="utf-8")).get("fps")
                if isinstance(fps, (int, float)) and fps > 0:
                    return float(fps)
            except (OSError, ValueError, TypeError):
                pass

    if source_episode_path:
        try:
            import pyarrow.parquet as pq

            timestamps = np.asarray(
                pq.read_table(str(source_episode_path), columns=["timestamp"])
                .column("timestamp")
                .to_numpy(),
                dtype=np.float64,
            )
            deltas = np.diff(timestamps)
            positive_deltas = deltas[deltas > 0]
            if positive_deltas.size:
                return float(1.0 / np.median(positive_deltas))
        except Exception:  # noqa: BLE001
            pass

    raise ValueError(
        "Unable to infer original observation frequency. Set --wam-hz explicitly "
        "or provide source meta/info.json with a positive fps."
    )


def _resolve_wam_hz(
    *,
    configured_wam_hz: float | None,
    obs_fps: float,
    temporal_compression_factor: int,
    is_wan_regular_grid: bool,
) -> float:
    if configured_wam_hz is not None:
        return configured_wam_hz
    if is_wan_regular_grid:
        if temporal_compression_factor <= 0:
            raise ValueError("Wan temporal_compression_factor must be positive.")
        return obs_fps / temporal_compression_factor
    return obs_fps


def _label_sidecar_path(label_sidecar_root: Path | None, episode_id: str) -> str | None:
    if label_sidecar_root is None:
        return None
    try:
        episode_index = int(episode_id.split("_")[-1])
    except ValueError:
        return str(label_sidecar_root / f"{episode_id}.parquet")
    return str(label_sidecar_root / f"episode_{episode_index:06d}.parquet")


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
                ("anchor_video_key", pa.string()),
                ("history_indices", pa.list_(pa.int64())),
                ("current_index", pa.int64()),
                ("future_indices", pa.list_(pa.int64())),
                ("history_state_indices", pa.list_(pa.int64())),
                ("current_state_index", pa.int64()),
                ("future_state_indices", pa.list_(pa.int64())),
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
                ("source_dataset_path", pa.string()),
                ("source_episode_path", pa.string()),
                ("history_valid_mask", pa.list_(pa.bool_())),
                ("future_valid_mask", pa.list_(pa.bool_())),
                ("action_valid_mask", pa.list_(pa.bool_())),
                ("history_action_valid_mask", pa.list_(pa.bool_())),
                ("history_state_valid_mask", pa.list_(pa.bool_())),
                ("future_state_valid_mask", pa.list_(pa.bool_())),
                ("action_representation", pa.string()),
            ]
        )
        table = pa.Table.from_pydict({name: [] for name in schema.names}, schema=schema)
    else:
        rows = [entry.to_dict() for entry in entries]
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
