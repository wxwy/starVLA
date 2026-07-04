"""MoWA RoboCasa365 Lerobot read-only schema adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa.sampler import (
    MoWAEpisodeToWindowSampler,
    select_mowa_smoke_anchor_index,
)
from starVLA.dataloader.mowa.schema import DATA_GATE, TBD, MoWAUnifiedEpisode, MoWAWindowConfig
from starVLA.mowa_constants import MOWA_P0_FULL_HEADS


ROBOCASA365_REQUIRED_PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)
MOWA_P0_HEADS = MOWA_P0_FULL_HEADS


@dataclass(frozen=True)
class MoWARoboCasa365EpisodeSchema:
    dataset_path: str
    parquet_path: str
    episode_index: int
    row_count: int
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    timestamp_preview: tuple[float, ...]
    frame_index_preview: tuple[int, ...]
    task_index_preview: tuple[int, ...]
    action_shape: tuple[int, ...] | str
    state_shape: tuple[int, ...] | str
    instruction: str
    unified_episode: MoWAUnifiedEpisode

    @property
    def schema_available(self) -> bool:
        return not self.missing_columns and self.row_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "parquet_path": self.parquet_path,
            "episode_index": self.episode_index,
            "row_count": self.row_count,
            "columns": self.columns,
            "missing_columns": self.missing_columns,
            "timestamp_preview": self.timestamp_preview,
            "frame_index_preview": self.frame_index_preview,
            "task_index_preview": self.task_index_preview,
            "action_shape": self.action_shape,
            "state_shape": self.state_shape,
            "instruction": self.instruction,
            "schema_available": self.schema_available,
            "unified_episode": {
                "episode_id": self.unified_episode.episode_id,
                "dataset_source": self.unified_episode.dataset_source,
                "split": self.unified_episode.split,
                "instruction": self.unified_episode.instruction,
                "num_steps": self.unified_episode.num_steps,
                "observations": dict(self.unified_episode.observations),
                "actions": dict(self.unified_episode.actions),
                "wam_targets": dict(self.unified_episode.wam_targets),
                "metadata": dict(self.unified_episode.metadata),
            },
        }


@dataclass(frozen=True)
class MoWARoboCasa365DatasetSmoke:
    dataset_path: str
    episode_count: int
    sampled_episode_indices: tuple[int, ...]
    sampled_row_counts: tuple[int, ...]
    min_episode_length: int | str
    max_episode_length: int | str
    boundary_smoke: dict[str, Any]
    p0_label_coverage: dict[str, str]
    unresolved_items: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "episode_count": self.episode_count,
            "sampled_episode_indices": self.sampled_episode_indices,
            "sampled_row_counts": self.sampled_row_counts,
            "min_episode_length": self.min_episode_length,
            "max_episode_length": self.max_episode_length,
            "boundary_smoke": self.boundary_smoke,
            "p0_label_coverage": self.p0_label_coverage,
            "unresolved_items": self.unresolved_items,
        }


@dataclass(frozen=True)
class MoWARoboCasa365ProfileSmoke:
    dataset_path: str
    sampled_episode_indices: tuple[int, ...]
    sampled_row_counts: tuple[int, ...]
    timestamp_monotonic: bool
    frame_index_monotonic: bool
    timestamp_delta_preview: tuple[float, ...]
    action_shape_consistent: bool
    state_shape_consistent: bool
    action_shapes: tuple[tuple[int, ...] | str, ...]
    state_shapes: tuple[tuple[int, ...] | str, ...]
    next_done_seen: bool
    reward_signal_seen: bool
    obs_fps_status: str = DATA_GATE
    action_hz_status: str = DATA_GATE
    history_window_status: str = DATA_GATE
    future_window_status: str = DATA_GATE
    notes: tuple[str, ...] = (
        "Timestamp deltas are parquet previews for G0 only; they are not MoWA production profile results.",
        "Production WAM Hz, history window, future window and action chunk remain Data Gate.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "sampled_episode_indices": self.sampled_episode_indices,
            "sampled_row_counts": self.sampled_row_counts,
            "timestamp_monotonic": self.timestamp_monotonic,
            "frame_index_monotonic": self.frame_index_monotonic,
            "timestamp_delta_preview": self.timestamp_delta_preview,
            "action_shape_consistent": self.action_shape_consistent,
            "state_shape_consistent": self.state_shape_consistent,
            "action_shapes": self.action_shapes,
            "state_shapes": self.state_shapes,
            "next_done_seen": self.next_done_seen,
            "reward_signal_seen": self.reward_signal_seen,
            "obs_fps_status": self.obs_fps_status,
            "action_hz_status": self.action_hz_status,
            "history_window_status": self.history_window_status,
            "future_window_status": self.future_window_status,
            "notes": list(self.notes),
        }


def inspect_robocasa365_lerobot_episode_schema(
    dataset_path: Path | str,
    episode_index: int = 0,
    preview_rows: int = 8,
) -> MoWARoboCasa365EpisodeSchema:
    """只读检查一个 RoboCasa365 Lerobot episode 的 parquet schema。

    该函数只读取 parquet schema 和少量标量列预览，不读取 video 内容，
    不计算 fps/Hz/window 的正式数值结论。
    """

    root = Path(dataset_path)
    parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"MoWA RoboCasa365 parquet not found: {parquet_path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA RoboCasa365 schema smoke requires pyarrow.") from exc

    parquet_file = pq.ParquetFile(parquet_path)
    columns = tuple(parquet_file.schema_arrow.names)
    missing_columns = tuple(
        name for name in ROBOCASA365_REQUIRED_PARQUET_COLUMNS if name not in columns
    )

    preview_columns = [
        name
        for name in ("timestamp", "frame_index", "episode_index", "task_index", "action", "observation.state")
        if name in columns
    ]
    preview = parquet_file.read(columns=preview_columns).slice(0, preview_rows).to_pydict()
    timestamp_preview = tuple(float(value) for value in preview.get("timestamp", ()))
    frame_index_preview = tuple(int(value) for value in preview.get("frame_index", ()))
    task_index_preview = tuple(int(value) for value in preview.get("task_index", ()))

    instruction = _resolve_instruction(root, episode_index, task_index_preview)
    episode = MoWAUnifiedEpisode(
        episode_id=f"robocasa365_open_drawer_episode_{episode_index:06d}",
        dataset_source="robocasa365",
        split="train",
        instruction=instruction,
        timestamps=tuple(range(parquet_file.metadata.num_rows)),
        observations={
            "rgb": {
                "source": "lerobot_video_files",
                "keys": _read_video_keys(root),
                "status": DATA_GATE,
            },
            "robot_state": {
                "source": "observation.state",
                "shape": _fixed_size_list_shape(parquet_file.schema_arrow.field("observation.state"))
                if "observation.state" in columns
                else TBD,
            },
        },
        actions={
            "canonical_action": {
                "source": "action",
                "shape": _fixed_size_list_shape(parquet_file.schema_arrow.field("action"))
                if "action" in columns
                else TBD,
            }
        },
        wam_targets={
            "p0_labels": DATA_GATE,
            "future_wan_latent": DATA_GATE,
        },
        metadata={
            "source_format": "robocasa365_lerobot",
            "parquet_path": str(parquet_path),
            "row_count": parquet_file.metadata.num_rows,
            "columns": columns,
            "missing_columns": missing_columns,
            "obs_fps": DATA_GATE,
            "action_hz": DATA_GATE,
            "history_window": DATA_GATE,
            "future_window": DATA_GATE,
        },
    )
    episode.validate()

    return MoWARoboCasa365EpisodeSchema(
        dataset_path=str(root),
        parquet_path=str(parquet_path),
        episode_index=episode_index,
        row_count=parquet_file.metadata.num_rows,
        columns=columns,
        missing_columns=missing_columns,
        timestamp_preview=timestamp_preview,
        frame_index_preview=frame_index_preview,
        task_index_preview=task_index_preview,
        action_shape=episode.actions["canonical_action"]["shape"],
        state_shape=episode.observations["robot_state"]["shape"],
        instruction=instruction,
        unified_episode=episode,
    )


def inspect_robocasa365_lerobot_profile_smoke(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
    preview_rows: int = 32,
) -> MoWARoboCasa365ProfileSmoke:
    """只读检查 timestamp / frame_index / shape 一致性。

    该函数不读取视频内容，不设置正式 fps/Hz/window 结论。
    """

    root = Path(dataset_path)
    sampled_schemas = []
    timestamp_deltas = []
    timestamp_monotonic_flags = []
    frame_index_monotonic_flags = []
    next_done_seen = False
    reward_signal_seen = False

    for episode_index in episode_indices:
        schema = inspect_robocasa365_lerobot_episode_schema(
            root,
            episode_index=episode_index,
            preview_rows=preview_rows,
        )
        sampled_schemas.append(schema)

        preview = _read_scalar_preview(
            Path(schema.parquet_path),
            columns=("timestamp", "frame_index", "next.done", "next.reward"),
            limit=preview_rows,
        )
        timestamps = tuple(float(value) for value in preview.get("timestamp", ()))
        frame_indices = tuple(int(value) for value in preview.get("frame_index", ()))
        timestamp_monotonic_flags.append(_is_monotonic(timestamps))
        frame_index_monotonic_flags.append(_is_monotonic(frame_indices))
        timestamp_deltas.extend(_rounded_deltas(timestamps))
        next_done_seen = next_done_seen or any(bool(value) for value in preview.get("next.done", ()))
        reward_signal_seen = reward_signal_seen or any(float(value) != 0.0 for value in preview.get("next.reward", ()))

    action_shapes = tuple(schema.action_shape for schema in sampled_schemas)
    state_shapes = tuple(schema.state_shape for schema in sampled_schemas)

    return MoWARoboCasa365ProfileSmoke(
        dataset_path=str(root),
        sampled_episode_indices=tuple(schema.episode_index for schema in sampled_schemas),
        sampled_row_counts=tuple(schema.row_count for schema in sampled_schemas),
        timestamp_monotonic=all(timestamp_monotonic_flags) if timestamp_monotonic_flags else False,
        frame_index_monotonic=all(frame_index_monotonic_flags) if frame_index_monotonic_flags else False,
        timestamp_delta_preview=tuple(sorted(set(timestamp_deltas)))[:8],
        action_shape_consistent=len(set(action_shapes)) <= 1 if action_shapes else False,
        state_shape_consistent=len(set(state_shapes)) <= 1 if state_shapes else False,
        action_shapes=action_shapes,
        state_shapes=state_shapes,
        next_done_seen=next_done_seen,
        reward_signal_seen=reward_signal_seen,
    )


def inspect_robocasa365_lerobot_dataset_smoke(
    dataset_path: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
    window_config: MoWAWindowConfig | None = None,
) -> MoWARoboCasa365DatasetSmoke:
    """只读检查多个 episode 的边界与 P0 label coverage 初判。"""

    root = Path(dataset_path)
    episodes = _read_episode_rows(root / "meta" / "episodes.jsonl")
    episode_lengths = tuple(int(row.get("length", 0)) for row in episodes)
    available_indices = {int(row["episode_index"]) for row in episodes if "episode_index" in row}
    sampled_indices = tuple(index for index in episode_indices if index in available_indices)

    if window_config is None:
        window_config = MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)

    sampler = MoWAEpisodeToWindowSampler(window_config)
    sampled_schemas = tuple(
        inspect_robocasa365_lerobot_episode_schema(root, episode_index=index, preview_rows=8)
        for index in sampled_indices
    )
    sampled_row_counts = tuple(schema.row_count for schema in sampled_schemas)
    boundary_results = []
    for schema in sampled_schemas:
        anchor_index = select_mowa_smoke_anchor_index(
            schema.unified_episode.num_steps,
            window_config,
        )
        sample = sampler.sample(schema.unified_episode, anchor_index=anchor_index)
        boundary_results.append(
            {
                "episode_index": schema.episode_index,
                "row_count": schema.row_count,
                "anchor_index": anchor_index,
                "history_indices": sample.history_indices,
                "future_indices": sample.future_indices,
                "action_target_indices": sample.action_target_indices,
                "future_action_in_inputs": "action_chunk_target" in sample.inputs,
                "full_future": sample.boundary_mask.get("has_full_future", False),
                "full_action_chunk": sample.boundary_mask.get("has_full_action_chunk", False),
            }
        )

    return MoWARoboCasa365DatasetSmoke(
        dataset_path=str(root),
        episode_count=len(episodes),
        sampled_episode_indices=sampled_indices,
        sampled_row_counts=sampled_row_counts,
        min_episode_length=min(episode_lengths) if episode_lengths else TBD,
        max_episode_length=max(episode_lengths) if episode_lengths else TBD,
        boundary_smoke={
            "window_config": {
                "history_steps": window_config.history_steps,
                "future_steps": window_config.future_steps,
                "action_chunk_steps": window_config.action_chunk_steps,
                "status": "smoke_only_target",
            },
            "sampled_windows": tuple(boundary_results),
            "cross_episode_leakage_status": "smoke_passed" if boundary_results else TBD,
            "future_action_leakage_status": (
                "smoke_passed"
                if boundary_results and all(not row["future_action_in_inputs"] for row in boundary_results)
                else TBD
            ),
        },
        p0_label_coverage=_build_p0_label_coverage(),
        unresolved_items=(
            "P0 labels are coverage candidates only; no label builder has been validated.",
            "obs fps / action Hz / training windows remain Data Gate until profiling.",
            "full leakage gate still needs production sampler coverage across train/val splits.",
        ),
    )


def _resolve_instruction(
    dataset_path: Path,
    episode_index: int,
    task_index_preview: tuple[int, ...],
) -> str:
    episode_tasks = _read_episode_tasks(dataset_path / "meta" / "episodes.jsonl", episode_index)
    if episode_tasks:
        return episode_tasks[0]

    tasks = _read_tasks(dataset_path / "meta" / "tasks.jsonl")
    if task_index_preview and task_index_preview[0] in tasks:
        return tasks[task_index_preview[0]]
    return "TBD"


def _read_episode_tasks(path: Path, episode_index: int) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if data.get("episode_index") == episode_index:
                tasks = data.get("tasks", ())
                return tuple(str(task) for task in tasks)
    return ()


def _read_episode_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(data)
    return tuple(rows)


def _read_tasks(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    tasks = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if "task_index" in data and "task" in data:
                tasks[int(data["task_index"])] = str(data["task"])
    return tasks


def _read_video_keys(dataset_path: Path) -> tuple[str, ...]:
    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.is_file():
        return ()
    with modality_path.open("r", encoding="utf-8") as file:
        modality = json.load(file)
    video = modality.get("video", {}) if isinstance(modality, dict) else {}
    return tuple(video.keys())


def _fixed_size_list_shape(field: Any) -> tuple[int, ...] | str:
    list_size = getattr(field.type, "list_size", None)
    if list_size is None:
        return TBD
    return (int(list_size),)


def _read_scalar_preview(path: Path, columns: tuple[str, ...], limit: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MoWA RoboCasa365 profile smoke requires pyarrow.") from exc

    parquet_file = pq.ParquetFile(path)
    existing_columns = [column for column in columns if column in parquet_file.schema_arrow.names]
    if not existing_columns:
        return {}
    return parquet_file.read(columns=existing_columns).slice(0, limit).to_pydict()


def _is_monotonic(values: tuple[float | int, ...]) -> bool:
    if not values:
        return False
    prev = values[0]
    for value in values[1:]:
        if value < prev:
            return False
        prev = value
    return True


def _rounded_deltas(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) < 2:
        return ()
    return tuple(round(values[idx] - values[idx - 1], 6) for idx in range(1, len(values)))


def _build_p0_label_coverage() -> dict[str, str]:
    return {
        "task_progress": "candidate_from_frame_index_and_episode_length; Data Gate",
        "manipulation_readiness": "candidate_from_state_action_reward; Data Gate",
        "failure_risk": "insufficient_without_failure_annotation; Data Gate",
        "next_best_view_score": "requires_view_label_or_proxy_definition; Data Gate",
        "subgoal_feasibility": "candidate_from_reward_done_and_task_progress; Data Gate",
        "object_visibility_future": "requires_video_decode_or_visibility_proxy; Data Gate",
        "action_outcome_class": "candidate_from_next.reward_next.done; Data Gate",
    }
