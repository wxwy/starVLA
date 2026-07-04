"""MoWA full-recipe temporal profile utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.robocasa365_adapter import fixed_size_list_shape
from starVLA.dataloader.mowa.robocasa365_recipe import (
    MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
)
from starVLA.dataloader.mowa.schema import DATA_GATE


@dataclass(frozen=True)
class MoWATemporalProfileTask:
    task: str
    relative_path: str
    episode_count: int
    parquet_count: int
    metadata_total_frames: int
    parquet_total_rows: int
    min_episode_rows: int | str
    max_episode_rows: int | str
    timestamp_monotonic: bool
    frame_index_monotonic: bool
    timestamp_delta_values: tuple[float, ...]
    state_shapes: tuple[tuple[int, ...] | str, ...]
    action_shapes: tuple[tuple[int, ...] | str, ...]
    reward_signal_seen: bool
    next_done_seen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "relative_path": self.relative_path,
            "episode_count": self.episode_count,
            "parquet_count": self.parquet_count,
            "metadata_total_frames": self.metadata_total_frames,
            "parquet_total_rows": self.parquet_total_rows,
            "min_episode_rows": self.min_episode_rows,
            "max_episode_rows": self.max_episode_rows,
            "timestamp_monotonic": self.timestamp_monotonic,
            "frame_index_monotonic": self.frame_index_monotonic,
            "timestamp_delta_values": self.timestamp_delta_values,
            "state_shapes": self.state_shapes,
            "action_shapes": self.action_shapes,
            "reward_signal_seen": self.reward_signal_seen,
            "next_done_seen": self.next_done_seen,
        }


@dataclass(frozen=True)
class MoWATemporalProfileReport:
    recipe_name: str
    data_root: str
    task_count: int
    episode_count: int
    parquet_count: int
    metadata_total_frames: int
    parquet_total_rows: int
    timestamp_monotonic: bool
    frame_index_monotonic: bool
    timestamp_delta_values: tuple[float, ...]
    state_shapes: tuple[tuple[int, ...] | str, ...]
    action_shapes: tuple[tuple[int, ...] | str, ...]
    reward_signal_seen: bool
    next_done_seen: bool
    tasks: tuple[MoWATemporalProfileTask, ...]
    obs_fps_status: str
    action_hz_status: str
    history_window_status: str
    future_window_status: str
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "task_count": self.task_count,
            "episode_count": self.episode_count,
            "parquet_count": self.parquet_count,
            "metadata_total_frames": self.metadata_total_frames,
            "parquet_total_rows": self.parquet_total_rows,
            "timestamp_monotonic": self.timestamp_monotonic,
            "frame_index_monotonic": self.frame_index_monotonic,
            "timestamp_delta_values": self.timestamp_delta_values,
            "state_shapes": self.state_shapes,
            "action_shapes": self.action_shapes,
            "reward_signal_seen": self.reward_signal_seen,
            "next_done_seen": self.next_done_seen,
            "tasks": [task.to_dict() for task in self.tasks],
            "obs_fps_status": self.obs_fps_status,
            "action_hz_status": self.action_hz_status,
            "history_window_status": self.history_window_status,
            "future_window_status": self.future_window_status,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_atomic_core_temporal_profile(
    data_root: Path | str,
    max_episodes_per_task: int | None = None,
) -> MoWATemporalProfileReport:
    """Profile scalar temporal fields across the fixed recipe.

    该函数读取 parquet scalar columns。它给出数据轮廓，但不自动冻结
    WAM Hz、history window、future horizon 或 action chunk。
    """

    root = Path(data_root)
    task_profiles = tuple(
        _profile_task(root, task, relative_path, max_episodes_per_task=max_episodes_per_task)
        for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items()
    )
    timestamp_values = tuple(
        sorted({value for profile in task_profiles for value in profile.timestamp_delta_values})
    )
    state_shapes = tuple(sorted({shape for profile in task_profiles for shape in profile.state_shapes}, key=str))
    action_shapes = tuple(sorted({shape for profile in task_profiles for shape in profile.action_shapes}, key=str))
    ok = (
        all(profile.timestamp_monotonic for profile in task_profiles)
        and all(profile.frame_index_monotonic for profile in task_profiles)
        and len(state_shapes) == 1
        and len(action_shapes) == 1
    )
    return MoWATemporalProfileReport(
        recipe_name="mowa_robocasa365_target_human_atomic_core_v1",
        data_root=str(root),
        task_count=len(task_profiles),
        episode_count=sum(profile.episode_count for profile in task_profiles),
        parquet_count=sum(profile.parquet_count for profile in task_profiles),
        metadata_total_frames=sum(profile.metadata_total_frames for profile in task_profiles),
        parquet_total_rows=sum(profile.parquet_total_rows for profile in task_profiles),
        timestamp_monotonic=all(profile.timestamp_monotonic for profile in task_profiles),
        frame_index_monotonic=all(profile.frame_index_monotonic for profile in task_profiles),
        timestamp_delta_values=timestamp_values,
        state_shapes=state_shapes,
        action_shapes=action_shapes,
        reward_signal_seen=any(profile.reward_signal_seen for profile in task_profiles),
        next_done_seen=any(profile.next_done_seen for profile in task_profiles),
        tasks=task_profiles,
        obs_fps_status=DATA_GATE,
        action_hz_status=DATA_GATE,
        history_window_status=DATA_GATE,
        future_window_status=DATA_GATE,
        go_no_go=(
            "TBD: temporal profile passed; production WAM Hz/window remain Data Gate"
            if ok
            else "No-Go: temporal profile inconsistency detected"
        ),
        notes=(
            "Timestamp deltas are computed from parquet scalar columns.",
            "This report does not freeze production WAM Hz, history window, future horizon or action chunk.",
            "Video decode throughput and training dataloader performance are not measured here.",
        ),
    )


def _profile_task(
    data_root: Path,
    task: str,
    relative_path: str,
    max_episodes_per_task: int | None,
) -> MoWATemporalProfileTask:
    dataset_path = data_root / relative_path
    rows = _read_episode_rows(dataset_path / "meta" / "episodes.jsonl")
    if max_episodes_per_task is not None:
        rows = rows[:max_episodes_per_task]
    metadata_lengths = tuple(int(row.get("length", 0)) for row in rows)
    episode_indices = tuple(int(row.get("episode_index", 0)) for row in rows)

    row_counts = []
    timestamp_flags = []
    frame_flags = []
    timestamp_deltas = set()
    state_shapes = set()
    action_shapes = set()
    reward_signal_seen = False
    next_done_seen = False
    for episode_index in episode_indices:
        item = _profile_parquet(
            dataset_path / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        row_counts.append(item["row_count"])
        timestamp_flags.append(item["timestamp_monotonic"])
        frame_flags.append(item["frame_index_monotonic"])
        timestamp_deltas.update(item["timestamp_delta_values"])
        state_shapes.add(item["state_shape"])
        action_shapes.add(item["action_shape"])
        reward_signal_seen = reward_signal_seen or item["reward_signal_seen"]
        next_done_seen = next_done_seen or item["next_done_seen"]

    return MoWATemporalProfileTask(
        task=task,
        relative_path=relative_path,
        episode_count=len(rows),
        parquet_count=len(row_counts),
        metadata_total_frames=sum(metadata_lengths),
        parquet_total_rows=sum(row_counts),
        min_episode_rows=min(row_counts) if row_counts else DATA_GATE,
        max_episode_rows=max(row_counts) if row_counts else DATA_GATE,
        timestamp_monotonic=all(timestamp_flags) if timestamp_flags else False,
        frame_index_monotonic=all(frame_flags) if frame_flags else False,
        timestamp_delta_values=tuple(sorted(timestamp_deltas)),
        state_shapes=tuple(sorted(state_shapes, key=str)),
        action_shapes=tuple(sorted(action_shapes, key=str)),
        reward_signal_seen=reward_signal_seen,
        next_done_seen=next_done_seen,
    )


def _profile_parquet(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"MoWA temporal profile parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA temporal profile requires pyarrow.") from exc

    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    read_columns = [
        column
        for column in ("timestamp", "frame_index", "next.reward", "next.done")
        if column in columns
    ]
    data = parquet_file.read(columns=read_columns).to_pydict()
    timestamps = tuple(float(value) for value in data.get("timestamp", ()))
    frame_indices = tuple(int(value) for value in data.get("frame_index", ()))
    rewards = tuple(float(value) for value in data.get("next.reward", ()))
    dones = tuple(bool(value) for value in data.get("next.done", ()))
    return {
        "row_count": parquet_file.metadata.num_rows,
        "timestamp_monotonic": _is_monotonic(timestamps),
        "frame_index_monotonic": _is_monotonic(frame_indices),
        "timestamp_delta_values": _rounded_deltas(timestamps),
        "state_shape": fixed_size_list_shape(parquet_file.schema_arrow.field("observation.state"))
        if "observation.state" in columns
        else DATA_GATE,
        "action_shape": fixed_size_list_shape(parquet_file.schema_arrow.field("action"))
        if "action" in columns
        else DATA_GATE,
        "reward_signal_seen": any(value != 0.0 for value in rewards),
        "next_done_seen": any(dones),
    }


def _read_episode_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return tuple(rows)


def _is_monotonic(values: tuple[float | int, ...]) -> bool:
    return all(curr >= prev for prev, curr in zip(values, values[1:]))


def _rounded_deltas(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted({round(curr - prev, 6) for prev, curr in zip(values, values[1:])}))
