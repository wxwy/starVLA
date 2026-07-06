"""MoWA full-recipe leakage gate smoke utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from starVLA.dataloader.mowa.robocasa365_recipe import (
    MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
)
from starVLA.dataloader.mowa.sampler import (
    MoWAEpisodeToWindowSampler,
    select_mowa_leakage_anchor_indices,
)
from starVLA.dataloader.mowa.schema import DATA_GATE, MoWAUnifiedEpisode, MoWAWindowConfig


@dataclass(frozen=True)
class MoWALeakageGateTaskSmoke:
    task: str
    relative_path: str
    episode_count: int
    checked_window_count: int
    failed_window_count: int
    min_episode_length: int | str
    max_episode_length: int | str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "relative_path": self.relative_path,
            "episode_count": self.episode_count,
            "checked_window_count": self.checked_window_count,
            "failed_window_count": self.failed_window_count,
            "min_episode_length": self.min_episode_length,
            "max_episode_length": self.max_episode_length,
        }


@dataclass(frozen=True)
class MoWALeakageGateSmoke:
    recipe_name: str
    data_root: str
    task_count: int
    episode_count: int
    checked_window_count: int
    failed_window_count: int
    window_config: Mapping[str, Any]
    tasks: tuple[MoWALeakageGateTaskSmoke, ...]
    future_action_leakage_status: str
    cross_episode_leakage_status: str
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "task_count": self.task_count,
            "episode_count": self.episode_count,
            "checked_window_count": self.checked_window_count,
            "failed_window_count": self.failed_window_count,
            "window_config": dict(self.window_config),
            "tasks": [task.to_dict() for task in self.tasks],
            "future_action_leakage_status": self.future_action_leakage_status,
            "cross_episode_leakage_status": self.cross_episode_leakage_status,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_atomic_core_leakage_gate_smoke(
    data_root: Path | str,
    window_config: MoWAWindowConfig | None = None,
) -> MoWALeakageGateSmoke:
    """Run metadata-level leakage smoke across every episode in the fixed recipe."""

    root = Path(data_root)
    if window_config is None:
        window_config = MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)
    sampler = MoWAEpisodeToWindowSampler(window_config)

    task_reports = []
    total_episodes = 0
    total_checked = 0
    total_failed = 0
    for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items():
        rows = _read_episode_rows(root / relative_path / "meta" / "episodes.jsonl")
        lengths = tuple(int(row.get("length", 0)) for row in rows)
        checked = 0
        failed = 0
        for row in rows:
            episode_index = int(row.get("episode_index", 0))
            length = int(row.get("length", 0))
            if length <= 0:
                failed += 1
                continue
            episode = MoWAUnifiedEpisode(
                episode_id=f"{task}_episode_{episode_index:06d}",
                dataset_source="robocasa365",
                split="train",
                instruction=_first_task(row),
                timestamps=tuple(range(length)),
                observations={"rgb": DATA_GATE, "robot_state": DATA_GATE},
                actions={"canonical_action": DATA_GATE},
                wam_targets={"future_labels": DATA_GATE, "future_wan_latent": DATA_GATE},
                metadata={
                    "obs_fps": DATA_GATE,
                    "action_hz": DATA_GATE,
                    "history_window": DATA_GATE,
                    "future_window": DATA_GATE,
                },
            )
            for anchor_index in select_mowa_leakage_anchor_indices(length, window_config):
                checked += 1
                try:
                    sample = sampler.sample(episode, anchor_index=anchor_index)
                    if "action_chunk_target" in sample.inputs:
                        failed += 1
                except Exception:
                    failed += 1
        task_reports.append(
            MoWALeakageGateTaskSmoke(
                task=task,
                relative_path=relative_path,
                episode_count=len(rows),
                checked_window_count=checked,
                failed_window_count=failed,
                min_episode_length=min(lengths) if lengths else DATA_GATE,
                max_episode_length=max(lengths) if lengths else DATA_GATE,
            )
        )
        total_episodes += len(rows)
        total_checked += checked
        total_failed += failed

    status = "smoke_passed" if total_checked and total_failed == 0 else "failed"
    return MoWALeakageGateSmoke(
        recipe_name="mowa_robocasa365_target_human_atomic_core_v1",
        data_root=str(root),
        task_count=len(task_reports),
        episode_count=total_episodes,
        checked_window_count=total_checked,
        failed_window_count=total_failed,
        window_config={
            "history_steps": window_config.history_steps,
            "future_steps": window_config.future_steps,
            "action_chunk_steps": window_config.action_chunk_steps,
            "status": "metadata_level_smoke",
        },
        tasks=tuple(task_reports),
        future_action_leakage_status=status,
        cross_episode_leakage_status=status,
        go_no_go=(
            "TBD: metadata leakage smoke passed; production dataloader remains Data Gate"
            if status == "smoke_passed"
            else "No-Go: metadata leakage smoke failed"
        ),
        notes=(
            "This smoke checks sampler boundaries over episode metadata, not training dataloader workers.",
            "Future action remains a target-only field; it is not inserted into WAM inputs.",
        ),
    )


def _read_episode_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return tuple(rows)


def _first_task(row: Mapping[str, Any]) -> str:
    tasks = row.get("tasks") or ()
    if tasks:
        return str(tasks[0])
    return "TBD"
