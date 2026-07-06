"""MoWA batch-level dataloader smoke utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from starVLA.dataloader.mowa.full_head_label_builder import build_mowa_future_label_smoke_sample
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
class MoWABatchSmokeSample:
    task: str
    relative_path: str
    episode_index: int
    row_count: int
    anchor_index: int
    history_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    action_target_indices: tuple[int, ...]
    input_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    future_label_keys: tuple[str, ...]
    future_mask_true_heads: tuple[str, ...]
    future_mask_false_heads: tuple[str, ...]
    boundary_mask: Mapping[str, bool]
    future_action_in_inputs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "relative_path": self.relative_path,
            "episode_index": self.episode_index,
            "row_count": self.row_count,
            "anchor_index": self.anchor_index,
            "history_indices": self.history_indices,
            "future_indices": self.future_indices,
            "action_target_indices": self.action_target_indices,
            "input_keys": self.input_keys,
            "target_keys": self.target_keys,
            "future_label_keys": self.future_label_keys,
            "future_mask_true_heads": self.future_mask_true_heads,
            "future_mask_false_heads": self.future_mask_false_heads,
            "boundary_mask": dict(self.boundary_mask),
            "future_action_in_inputs": self.future_action_in_inputs,
        }


@dataclass(frozen=True)
class MoWABatchDataloaderSmoke:
    recipe_name: str
    data_root: str
    task_count: int
    sampled_episode_indices: tuple[int, ...]
    window_config: Mapping[str, Any]
    sample_count: int
    samples: tuple[MoWABatchSmokeSample, ...]
    future_action_leakage_status: str
    constructible_label_status: str
    go_no_go: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "data_root": self.data_root,
            "task_count": self.task_count,
            "sampled_episode_indices": self.sampled_episode_indices,
            "window_config": dict(self.window_config),
            "sample_count": self.sample_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "future_action_leakage_status": self.future_action_leakage_status,
            "constructible_label_status": self.constructible_label_status,
            "go_no_go": self.go_no_go,
            "notes": list(self.notes),
        }


def build_mowa_atomic_core_batch_dataloader_smoke(
    data_root: Path | str,
    episode_indices: tuple[int, ...] = (0, 1, 4),
    window_config: MoWAWindowConfig | None = None,
) -> MoWABatchDataloaderSmoke:
    """Build batch-level smoke samples across the fixed atomic core recipe.

    该函数不实例化训练 dataloader，不读取视频内容，不启动训练。
    """

    root = Path(data_root)
    if window_config is None:
        window_config = MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)

    sampler = MoWAEpisodeToWindowSampler(window_config)
    samples: list[MoWABatchSmokeSample] = []
    for task, relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items():
        dataset_path = root / relative_path
        for episode_index in episode_indices:
            schema = inspect_robocasa365_lerobot_episode_schema(
                dataset_path,
                episode_index=episode_index,
                preview_rows=8,
            )
            anchor_index = select_mowa_smoke_anchor_index(schema.row_count, window_config)
            window_sample = sampler.sample(schema.unified_episode, anchor_index=anchor_index)
            label_sample = build_mowa_future_label_smoke_sample(
                dataset_path,
                episode_index=episode_index,
                row_index=anchor_index,
            )
            true_heads = tuple(head for head, enabled in label_sample.masks.items() if enabled)
            false_heads = tuple(head for head, enabled in label_sample.masks.items() if not enabled)
            samples.append(
                MoWABatchSmokeSample(
                    task=task,
                    relative_path=relative_path,
                    episode_index=episode_index,
                    row_count=schema.row_count,
                    anchor_index=anchor_index,
                    history_indices=window_sample.history_indices,
                    future_indices=window_sample.future_indices,
                    action_target_indices=window_sample.action_target_indices,
                    input_keys=tuple(sorted(window_sample.inputs.keys())),
                    target_keys=tuple(sorted(window_sample.targets.keys())),
                    future_label_keys=tuple(sorted(label_sample.labels.keys())),
                    future_mask_true_heads=true_heads,
                    future_mask_false_heads=false_heads,
                    boundary_mask=window_sample.boundary_mask,
                    future_action_in_inputs="action_chunk_target" in window_sample.inputs,
                )
            )

    future_action_ok = all(not sample.future_action_in_inputs for sample in samples)
    label_ok = all(
        sample.future_label_keys == ("action_outcome_class", "task_progress")
        and sample.future_mask_true_heads == ("task_progress", "action_outcome_class")
        for sample in samples
    )
    return MoWABatchDataloaderSmoke(
        recipe_name="mowa_robocasa365_target_human_atomic_core_v1",
        data_root=str(root),
        task_count=len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS),
        sampled_episode_indices=episode_indices,
        window_config={
            "history_steps": window_config.history_steps,
            "future_steps": window_config.future_steps,
            "action_chunk_steps": window_config.action_chunk_steps,
            "status": "smoke_only_target",
        },
        sample_count=len(samples),
        samples=tuple(samples),
        future_action_leakage_status="smoke_passed" if future_action_ok else "failed",
        constructible_label_status="smoke_passed" if label_ok else "failed",
        go_no_go=(
            "TBD: batch dataloader smoke passed; production dataloader remains Data Gate"
            if future_action_ok and label_ok
            else "No-Go: batch dataloader smoke failed"
        ),
        notes=(
            "Smoke samples combine WindowSample boundaries with ConstructibleHeads dry-run targets.",
            "No production dataloader, video decode, latent encoder or training loop is executed.",
        ),
    )
