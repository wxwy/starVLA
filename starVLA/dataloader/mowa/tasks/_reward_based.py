"""Reward-based task builder for atomic tasks without a reliable joint progress signal.

Some RoboCasa atomic tasks (e.g. pressing a microwave button, navigating to a
fixture) do not expose a task-specific joint whose qpos directly indicates
progress.  For these tasks we fall back to the binary ``next.reward`` signal:
progress is 0 until the task succeeds and 1 afterwards.

This is a coarse-grained proxy, but it is sufficient for producing future-label
sidecars for spot-checking and for training the subgoal / readiness / failure-risk
heads in the absence of kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import (
    AtomicTaskLabelBuilder,
    compute_subgoal_feasibility,
    failure_risk_label_from_window,
)


@dataclass(frozen=True)
class RewardBasedTaskSchema:
    """Schema for a reward-based atomic task."""

    task_name: str
    subgoal_id: str = "complete"
    completion_threshold: float = 0.95
    schema_version: str = "reward_based_v1"


class RewardBasedTaskBuilder(AtomicTaskLabelBuilder):
    """Builder that derives progress purely from the binary reward signal."""

    def __init__(self, schema: RewardBasedTaskSchema):
        self._schema = schema

    @property
    def task_name(self) -> str:
        return self._schema.task_name

    @property
    def schema(self) -> RewardBasedTaskSchema:
        return self._schema

    def build_cache_for_episode(
        self,
        parquet_path: Path,
        states_path: Path,
        model_path: Path | None = None,
        ep_meta_path: Path | None = None,
        *,
        failure_risk_horizon: int = 10,
        subgoal_horizon: int = 20,
        readiness_horizon: int = 5,
        readiness_progress_delta: float = 0.1,
        readiness_distance_threshold: float = 0.05,
        enable_kinematics: bool = True,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reward-based label cache requires pyarrow.") from exc

        table = pq.read_table(parquet_path, columns=["next.reward", "next.done", "frame_index"])
        data = table.to_pydict()
        rewards = np.asarray(data["next.reward"], dtype=np.float64)
        dones = np.asarray(data["next.done"], dtype=bool)
        frame_indices = np.asarray(data.get("frame_index", range(len(rewards))), dtype=np.int64)

        # Progress is a step function: 0 before success, 1 after.
        # We use cumulative max so a single positive reward propagates forward.
        progress = np.maximum.accumulate(rewards)
        row_count = int(progress.shape[0])

        failure_risk_values = np.zeros(row_count, dtype=np.float64)
        failure_risk_masks = np.ones(row_count, dtype=bool)
        subgoal_feasibility_values, subgoal_feasibility_masks = compute_subgoal_feasibility(
            progress=progress,
            completion_threshold=self.schema.completion_threshold,
            horizon=subgoal_horizon,
        )
        manipulation_readiness_values = np.zeros(row_count, dtype=np.float64)
        manipulation_readiness_masks = np.ones(row_count, dtype=bool)

        for timestep in range(row_count):
            current_progress = float(progress[timestep])
            completed_now = current_progress >= self.schema.completion_threshold

            failure_risk = failure_risk_label_from_window(
                rewards=rewards.tolist(),
                dones=dones.tolist(),
                anchor=timestep,
                horizon=failure_risk_horizon,
            )
            failure_risk_masks[timestep] = failure_risk is None
            failure_risk_values[timestep] = float(failure_risk) if failure_risk is not None else 0.0

            subgoal_feasibility_masks[timestep] = completed_now
            if not completed_now and subgoal_feasibility_values[timestep] > 0.0:
                subgoal_feasibility_values[timestep] = 1.0

            manipulation_readiness_masks[timestep] = completed_now
            if not completed_now:
                future = progress[timestep + 1 : timestep + 1 + readiness_horizon]
                if future.size == 0:
                    manipulation_readiness_masks[timestep] = True
                elif np.any(future - current_progress >= readiness_progress_delta):
                    manipulation_readiness_values[timestep] = 1.0

        return {
            "frame_index": frame_indices,
            "task_progress": progress.astype(np.float64),
            "failure_risk": failure_risk_values,
            "failure_risk_mask": failure_risk_masks,
            "subgoal_feasibility": subgoal_feasibility_values,
            "subgoal_feasibility_mask": subgoal_feasibility_masks,
            "manipulation_readiness": manipulation_readiness_values,
            "manipulation_readiness_mask": manipulation_readiness_masks,
            "row_count": row_count,
            "schema_version": self.schema.schema_version,
            "readiness_schema_version": f"{self.task_name.lower()}_reward_imminence_v1",
            "readiness_predicate": f"future reward progress >= {readiness_progress_delta}",
            "kinematics_available": False,
            "kinematics_message": "reward-based builder does not use kinematics",
        }
