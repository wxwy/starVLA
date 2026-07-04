"""MoWA production dataloader preflight smoke utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

from starVLA.dataloader.mowa.robocasa365_adapter import inspect_robocasa365_lerobot_episode_schema
from starVLA.dataloader.mowa.robocasa365_recipe import (
    MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
)
from starVLA.dataloader.mowa.sampler import (
    MoWAEpisodeToWindowSampler,
    select_mowa_smoke_anchor_index,
)
from starVLA.dataloader.mowa.schema import MoWAWindowConfig


@dataclass(frozen=True)
class MoWAProductionPreflightSample:
    task: str
    relative_path: str
    split: str
    episode_index: int
    anchor_index: int
    worker_slot: int
    rank: int
    future_action_in_inputs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "relative_path": self.relative_path,
            "split": self.split,
            "episode_index": self.episode_index,
            "anchor_index": self.anchor_index,
            "worker_slot": self.worker_slot,
            "rank": self.rank,
            "future_action_in_inputs": self.future_action_in_inputs,
        }


@dataclass(frozen=True)
class MoWAProductionPreflightSmoke:
    recipe_name: str
    data_root: str
    task_count: int
    split_config: Mapping[str, Any]
    window_config: Mapping[str, Any]
    worker_count: int
    rank_count: int
    train_episode_count: int
    val_episode_count: int
    split_overlap_count: int
    distributed_overlap_count: int
    worker_sample_count: int
    failed_sample_count: int
    samples: tuple[MoWAProductionPreflightSample, ...]
    split_status: str
    worker_status: str
    distributed_sampler_status: str
    future_action_leakage_status: str
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "task_count": self.task_count,
            "split_config": dict(self.split_config),
            "window_config": dict(self.window_config),
            "worker_count": self.worker_count,
            "rank_count": self.rank_count,
            "train_episode_count": self.train_episode_count,
            "val_episode_count": self.val_episode_count,
            "split_overlap_count": self.split_overlap_count,
            "distributed_overlap_count": self.distributed_overlap_count,
            "worker_sample_count": self.worker_sample_count,
            "failed_sample_count": self.failed_sample_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "split_status": self.split_status,
            "worker_status": self.worker_status,
            "distributed_sampler_status": self.distributed_sampler_status,
            "future_action_leakage_status": self.future_action_leakage_status,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_atomic_core_production_preflight_smoke(
    data_root: Path | str,
    *,
    val_every: int = 10,
    worker_count: int = 2,
    rank_count: int = 2,
    max_worker_samples: int = 16,
    window_config: MoWAWindowConfig | None = None,
) -> MoWAProductionPreflightSmoke:
    """Run a production-entry preflight without starting training.

    This checks deterministic train/val split, rank partition overlap and
    multi-worker schema/window reads over a small sample set.
    """

    if val_every <= 1:
        raise ValueError("MoWA production preflight requires val_every > 1.")
    if worker_count < 1:
        raise ValueError("MoWA production preflight requires worker_count >= 1.")
    if rank_count < 1:
        raise ValueError("MoWA production preflight requires rank_count >= 1.")
    if max_worker_samples < 1:
        raise ValueError("MoWA production preflight requires max_worker_samples >= 1.")

    root = Path(data_root)
    if window_config is None:
        window_config = MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)

    train_keys: list[tuple[str, str, int, int]] = []
    val_keys: list[tuple[str, str, int, int]] = []
    for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items():
        rows = _read_episode_rows(root / relative_path / "meta" / "episodes.jsonl")
        for row in rows:
            episode_index = int(row.get("episode_index", 0))
            length = int(row.get("length", 0))
            key = (task, relative_path, episode_index, length)
            if episode_index % val_every == 0:
                val_keys.append(key)
            else:
                train_keys.append(key)

    split_overlap_count = len(
        {(task, relative_path, episode_index) for task, relative_path, episode_index, _ in train_keys}
        & {(task, relative_path, episode_index) for task, relative_path, episode_index, _ in val_keys}
    )
    rank_sets = _rank_episode_sets(train_keys, rank_count)
    distributed_overlap_count = _count_pairwise_overlap(rank_sets)

    worker_jobs = []
    for worker_slot, key in enumerate(train_keys[:max_worker_samples]):
        rank = worker_slot % rank_count
        worker_jobs.append(
            (
                str(root),
                key[0],
                key[1],
                key[2],
                key[3],
                worker_slot % worker_count,
                rank,
                window_config.history_steps,
                window_config.future_steps,
                window_config.action_chunk_steps,
            )
        )

    samples = _run_worker_jobs(worker_jobs, worker_count)
    failed_sample_count = sum(1 for sample in samples if sample.future_action_in_inputs)
    split_ok = bool(train_keys) and bool(val_keys) and split_overlap_count == 0
    worker_ok = len(samples) == len(worker_jobs) and failed_sample_count == 0
    distributed_ok = distributed_overlap_count == 0
    future_ok = failed_sample_count == 0
    all_ok = split_ok and worker_ok and distributed_ok and future_ok

    return MoWAProductionPreflightSmoke(
        recipe_name="mowa_robocasa365_target_human_atomic_core_v1",
        data_root=str(root),
        task_count=len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS),
        split_config={
            "strategy": "episode_index_modulo",
            "val_every": val_every,
            "status": "smoke_only_target",
        },
        window_config={
            "history_steps": window_config.history_steps,
            "future_steps": window_config.future_steps,
            "action_chunk_steps": window_config.action_chunk_steps,
            "status": "smoke_only_target",
        },
        worker_count=worker_count,
        rank_count=rank_count,
        train_episode_count=len(train_keys),
        val_episode_count=len(val_keys),
        split_overlap_count=split_overlap_count,
        distributed_overlap_count=distributed_overlap_count,
        worker_sample_count=len(samples),
        failed_sample_count=failed_sample_count,
        samples=tuple(samples),
        split_status="smoke_passed" if split_ok else "failed",
        worker_status="smoke_passed" if worker_ok else "failed",
        distributed_sampler_status="smoke_passed" if distributed_ok else "failed",
        future_action_leakage_status="smoke_passed" if future_ok else "failed",
        go_no_go=(
            "TBD: production preflight smoke passed; main training still requires explicit E-001 launch"
            if all_ok
            else "No-Go: production preflight smoke failed"
        ),
        notes=(
            "This smoke does not instantiate the StarVLA training loop or start E-001.",
            "Split/rank/window parameters are smoke-only targets, not frozen production policy.",
            "Future action remains target-only and is checked against WAM inputs.",
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


def _rank_episode_sets(
    train_keys: list[tuple[str, str, int, int]],
    rank_count: int,
) -> tuple[set[tuple[str, str, int]], ...]:
    rank_sets = tuple(set() for _ in range(rank_count))
    for index, (task, relative_path, episode_index, _) in enumerate(train_keys):
        rank_sets[index % rank_count].add((task, relative_path, episode_index))
    return rank_sets


def _count_pairwise_overlap(rank_sets: tuple[set[tuple[str, str, int]], ...]) -> int:
    overlap_count = 0
    for left_index, left in enumerate(rank_sets):
        for right in rank_sets[left_index + 1 :]:
            overlap_count += len(left & right)
    return overlap_count


def _run_worker_jobs(
    worker_jobs: list[tuple[Any, ...]],
    worker_count: int,
) -> tuple[MoWAProductionPreflightSample, ...]:
    if not worker_jobs:
        return ()
    if worker_count == 1:
        return tuple(_inspect_worker_sample(job) for job in worker_jobs)
    context = get_context("spawn")
    with context.Pool(processes=worker_count) as pool:
        return tuple(pool.map(_inspect_worker_sample, worker_jobs))


def _inspect_worker_sample(job: tuple[Any, ...]) -> MoWAProductionPreflightSample:
    (
        root,
        task,
        relative_path,
        episode_index,
        row_count,
        worker_slot,
        rank,
        history_steps,
        future_steps,
        action_chunk_steps,
    ) = job
    window_config = MoWAWindowConfig(
        history_steps=history_steps,
        future_steps=future_steps,
        action_chunk_steps=action_chunk_steps,
    )
    schema = inspect_robocasa365_lerobot_episode_schema(
        Path(root) / relative_path,
        episode_index=episode_index,
        preview_rows=8,
    )
    anchor_index = select_mowa_smoke_anchor_index(row_count or schema.row_count, window_config)
    sample = MoWAEpisodeToWindowSampler(window_config).sample(
        schema.unified_episode,
        anchor_index=anchor_index,
    )
    return MoWAProductionPreflightSample(
        task=task,
        relative_path=relative_path,
        split="train",
        episode_index=episode_index,
        anchor_index=anchor_index,
        worker_slot=worker_slot,
        rank=rank,
        future_action_in_inputs="action_chunk_target" in sample.inputs,
    )
