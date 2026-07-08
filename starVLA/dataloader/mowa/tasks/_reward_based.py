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
from starVLA.dataloader.mowa.mujoco_state_utils import build_mujoco_model_from_episode
from starVLA.dataloader.mowa.visual_head_utils import compute_visual_head_labels


@dataclass(frozen=True)
class RewardBasedTaskSchema:
    """Schema for a reward-based atomic task."""

    task_name: str
    subgoal_id: str = "complete"
    completion_threshold: float = 0.95
    # Optional MuJoCo site/body name used as the 3D visual target.  If both are
    # None, visual heads are skipped (all masked).
    visual_target_site_name: str | None = None
    visual_target_body_name: str | None = None
    # Fallback candidate site names; the first one that exists in the model is used.
    visual_target_site_candidates: tuple[str, ...] = ()
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
        readiness_horizon: int = 10,
        readiness_progress_delta: float | None = None,
        readiness_distance_threshold: float | None = None,
        enable_kinematics: bool = True,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        if readiness_horizon is None:
            readiness_horizon = 10
        if readiness_progress_delta is None:
            readiness_progress_delta = 0.05
        if readiness_distance_threshold is None:
            readiness_distance_threshold = 0.05

        if model_path is None:
            model_path = states_path.parent / "model.xml.gz"
        if ep_meta_path is None:
            ep_meta_path = states_path.parent / "ep_meta.json"

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

        visual_labels = _compute_visual_labels_for_reward_based(
            model_path=model_path,
            states_path=states_path,
            ep_meta_path=ep_meta_path,
            schema=self.schema,
            progress=progress,
            completion_threshold=self.schema.completion_threshold,
            repo_root=repo_root,
        )
        if visual_labels is None:
            row_count = int(progress.shape[0])
            visual_labels = {
                "object_visibility_future": np.zeros(row_count, dtype=np.float64),
                "object_visibility_future_mask": np.ones(row_count, dtype=bool),
                "next_best_view_score": np.zeros(row_count, dtype=np.float64),
                "next_best_view_score_mask": np.ones(row_count, dtype=bool),
                "schema_version": "mowa_visual_proxy_skipped_v1",
            }

        result = {
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

        if visual_labels is not None:
            result["object_visibility_future"] = visual_labels["object_visibility_future"]
            result["object_visibility_future_mask"] = visual_labels["object_visibility_future_mask"]
            result["next_best_view_score"] = visual_labels["next_best_view_score"]
            result["next_best_view_score_mask"] = visual_labels["next_best_view_score_mask"]
            result["visual_head_schema_version"] = visual_labels.get("schema_version", "mowa_visual_proxy_v1")

        return result


def _resolve_site_id_from_candidates(model: Any, candidates: tuple[str, ...]) -> int | None:
    """Return the first existing site id from a list of candidate names."""
    import mujoco

    for name in candidates:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return int(site_id)
    return None


def _compute_visual_labels_for_reward_based(
    model_path: Path,
    states_path: Path,
    ep_meta_path: Path,
    schema: RewardBasedTaskSchema,
    progress: np.ndarray,
    completion_threshold: float,
    *,
    repo_root: Path | None = None,
    visibility_horizon: int = 10,
) -> dict[str, Any] | None:
    """Compute visual heads for a reward-based task if a target is configured."""
    from starVLA.dataloader.mowa.mujoco_state_utils import find_site_id, load_ep_meta, load_states

    has_explicit_target = (
        schema.visual_target_site_name is not None or schema.visual_target_body_name is not None
    )
    if not has_explicit_target and not schema.visual_target_site_candidates:
        return None

    try:
        model = build_mujoco_model_from_episode(model_path, repo_root=repo_root)
        states = load_states(states_path)
        ep_meta = load_ep_meta(ep_meta_path)

        site_id = None
        if schema.visual_target_site_name is not None:
            try:
                site_id = find_site_id(model, schema.visual_target_site_name)
            except ValueError:
                site_id = None
        if site_id is None and schema.visual_target_site_candidates:
            site_id = _resolve_site_id_from_candidates(model, schema.visual_target_site_candidates)

        target_body_id: int | None = None
        if site_id is not None:
            target_body_id = int(model.site_bodyid[site_id])

            def target_resolver(data: Any, timestep: int) -> np.ndarray | None:
                del timestep
                return data.site_xpos[site_id].copy()

            def target_body_id_resolver(data: Any, timestep: int) -> int | None:
                del data, timestep
                return target_body_id
        elif schema.visual_target_body_name is not None:
            import mujoco

            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, schema.visual_target_body_name)
            if body_id < 0:
                return None
            target_body_id = int(body_id)

            def target_resolver(data: Any, timestep: int) -> np.ndarray | None:
                del timestep
                return data.xpos[body_id].copy()

            def target_body_id_resolver(data: Any, timestep: int) -> int | None:
                del data, timestep
                return target_body_id
        else:
            return None

        return compute_visual_head_labels(
            model=model,
            states=states,
            ep_meta=ep_meta,
            target_point_resolver=target_resolver,
            target_body_id_resolver=target_body_id_resolver,
            progress=progress,
            completion_threshold=completion_threshold,
            visibility_horizon=visibility_horizon,
        )
    except Exception:  # pragma: no cover - runtime fallback
        return None
