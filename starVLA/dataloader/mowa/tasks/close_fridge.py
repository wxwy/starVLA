"""MoWA future-label builder for the CloseFridge atomic task.

CloseFridge may feature a single fridge door or a pair of French doors.  The
task succeeds only when all fridge doors are closed, so progress is defined as
the minimum progress across all detected fridge door joints.
"""

from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import (
    AtomicTaskLabelBuilder,
    compute_subgoal_feasibility,
    failure_risk_label_from_window,
    register_atomic_task_label_builder,
)
from starVLA.dataloader.mowa.mujoco_state_utils import build_mujoco_model_from_episode, load_states
from starVLA.dataloader.mowa.visual_head_utils import compute_visual_head_labels


def _hinge_progress(raw_qpos: np.ndarray, joint_range: tuple[float, ...]) -> np.ndarray:
    """Normalize a hinge joint so that closed = 1 and open = 0."""
    if not joint_range:
        return raw_qpos
    lo, hi = float(joint_range[0]), float(joint_range[1])
    denom = abs(hi - lo)
    if denom <= 0:
        return raw_qpos
    # For fridge doors: lo=0 (closed), hi>0 (open).  We want closed=1.
    return 1.0 - (raw_qpos - lo) / denom


@register_atomic_task_label_builder
class CloseFridgeLabelBuilder(AtomicTaskLabelBuilder):
    """Build future labels for CloseFridge using total fridge door progress.

    All detected fridge door joints contribute equally to the task progress via
    their mean progress.  This reflects the total closing effort: both doors
    must reach the closed state for the average to hit the completion threshold.
    """

    # Task-specific readiness defaults: fridge doors close slowly, so use a
    # longer horizon and a smaller progress delta than the global fallback.
    default_readiness_horizon: int = 15
    default_readiness_progress_delta: float = 0.03

    @property
    def task_name(self) -> str:
        return "CloseFridge"

    def _find_fridge_door_joints(self, xml_root: ET.Element) -> list[dict[str, Any]]:
        """Return all hinge joints belonging to the target fridge doors."""
        joints: list[dict[str, Any]] = []
        qpos_index = 1
        for joint in xml_root.iter("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
            name = joint.attrib["name"]
            # Target the actual fridge door hinge joints (not freezer/cabinet decoys).
            name_lower = name.lower()
            if (
                joint_type == "hinge"
                and "fridge" in name_lower
                and "door_joint" in name_lower
                and "freezer" not in name_lower
            ):
                range_text = joint.attrib.get("range", "")
                range_value = tuple(float(v) for v in range_text.split()) if range_text else ()
                joints.append({
                    "name": name,
                    "qpos_index": qpos_index,
                    "range": range_value,
                })
            qpos_index += nq
        return joints

    def build_cache_for_episode(
        self,
        parquet_path: Path,
        states_path: Path,
        model_path: Path | None = None,
        ep_meta_path: Path | None = None,
        *,
        failure_risk_horizon: int = 10,
        subgoal_horizon: int = 20,
        readiness_horizon: int | None = None,
        readiness_progress_delta: float | None = None,
        readiness_distance_threshold: float | None = None,
        enable_kinematics: bool = True,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        if readiness_horizon is None:
            readiness_horizon = self.default_readiness_horizon
        if readiness_progress_delta is None:
            readiness_progress_delta = self.default_readiness_progress_delta
        if readiness_distance_threshold is None:
            readiness_distance_threshold = 0.05

        if model_path is None:
            model_path = states_path.parent / "model.xml.gz"
        if ep_meta_path is None:
            ep_meta_path = states_path.parent / "ep_meta.json"

        xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
        states = load_states(states_path)
        model = build_mujoco_model_from_episode(model_path, repo_root=repo_root)
        door_joints = self._find_fridge_door_joints(xml_root)
        if not door_joints:
            raise ValueError(f"No fridge door joints found for {self.task_name}")

        door_body_ids: list[int] = []
        for joint in door_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if joint_id >= 0:
                door_body_ids.append(int(model.jnt_bodyid[joint_id]))

        # Compute per-door progress and aggregate by mean (total closing effort).
        per_door_progress: list[np.ndarray] = []
        for joint in door_joints:
            raw_qpos = states[:, joint["qpos_index"]]
            progress = _hinge_progress(raw_qpos, joint["range"])
            per_door_progress.append(np.asarray(progress, dtype=np.float64))
        overall_progress = np.mean(np.stack(per_door_progress, axis=0), axis=0)

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("CloseFridge label cache requires pyarrow.") from exc

        table = pq.read_table(parquet_path, columns=["next.reward", "next.done", "frame_index"])
        data = table.to_pydict()
        rewards = np.asarray(data["next.reward"], dtype=np.float64)
        dones = np.asarray(data["next.done"], dtype=bool)
        frame_indices = np.asarray(data.get("frame_index", range(len(rewards))), dtype=np.int64)

        row_count = min(len(rewards), len(dones), len(frame_indices), int(states.shape[0]))
        rewards = rewards[:row_count]
        dones = dones[:row_count]
        frame_indices = frame_indices[:row_count]
        overall_progress = overall_progress[:row_count]

        completion_threshold = 0.95
        failure_risk_values = np.zeros(row_count, dtype=np.float64)
        failure_risk_masks = np.ones(row_count, dtype=bool)
        subgoal_feasibility_values, subgoal_feasibility_masks = compute_subgoal_feasibility(
            progress=overall_progress,
            completion_threshold=completion_threshold,
            horizon=subgoal_horizon,
        )
        manipulation_readiness_values = np.zeros(row_count, dtype=np.float64)
        manipulation_readiness_masks = np.ones(row_count, dtype=bool)

        for timestep in range(row_count):
            current_progress = float(overall_progress[timestep])
            completed_now = current_progress >= completion_threshold

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
                future = overall_progress[timestep + 1 : timestep + 1 + readiness_horizon]
                if future.size == 0:
                    manipulation_readiness_masks[timestep] = True
                elif np.any(future - current_progress >= readiness_progress_delta):
                    manipulation_readiness_values[timestep] = 1.0

        ep_meta = json.loads(ep_meta_path.read_text())

        def _target_resolver(data: Any, timestep: int) -> np.ndarray | None:
            del timestep
            if not door_body_ids:
                return None
            positions = np.stack([data.xpos[bid].copy() for bid in door_body_ids], axis=0)
            return np.mean(positions, axis=0)

        visual_labels = compute_visual_head_labels(
            model=model,
            states=states,
            ep_meta=ep_meta,
            target_point_resolver=_target_resolver,
            progress=overall_progress,
            completion_threshold=completion_threshold,
            visibility_horizon=10,
        )

        return {
            "frame_index": frame_indices,
            "task_progress": overall_progress.astype(np.float64),
            "failure_risk": failure_risk_values,
            "failure_risk_mask": failure_risk_masks,
            "subgoal_feasibility": subgoal_feasibility_values,
            "subgoal_feasibility_mask": subgoal_feasibility_masks,
            "manipulation_readiness": manipulation_readiness_values,
            "manipulation_readiness_mask": manipulation_readiness_masks,
            "row_count": row_count,
            "door_joint_names": [j["name"] for j in door_joints],
            "door_progress": np.stack(per_door_progress, axis=1).astype(np.float64),
            "schema_version": "close_fridge_v3",
            "readiness_schema_version": "close_fridge_progress_imminence_v1",
            "readiness_predicate": f"future progress gain >= {readiness_progress_delta}",
            "kinematics_available": False,
            "kinematics_message": "progress-imminence proxy for multi-door fridge",
            "object_visibility_future": visual_labels["object_visibility_future"],
            "object_visibility_future_mask": visual_labels["object_visibility_future_mask"],
            "next_best_view_score": visual_labels["next_best_view_score"],
            "next_best_view_score_mask": visual_labels["next_best_view_score_mask"],
            "visual_head_schema_version": visual_labels.get("schema_version", "mowa_visual_proxy_v1"),
        }
