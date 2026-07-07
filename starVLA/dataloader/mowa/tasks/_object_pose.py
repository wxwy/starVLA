"""Object-pose task builder for PickPlace-style atomic tasks.

These tasks do not have a single task-specific joint.  Instead, success is
defined by the pose of a manipulated object (e.g. a mug moved from counter to
under the coffee dispenser).  We reconstruct the object trajectory from its
freejoint qpos and define progress as normalized displacement from the start
pose toward the final pose.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import (
    AtomicTaskLabelBuilder,
    compute_subgoal_feasibility,
    failure_risk_label_from_window,
)


@dataclass(frozen=True)
class ObjectPoseTaskSchema:
    """Schema for an object-pose atomic task."""

    task_name: str
    # Callable to select the object freejoint name from (xml_root, states).
    object_joint_selector: Callable[[ET.Element, np.ndarray], str | None]
    subgoal_id: str = "complete"
    completion_threshold: float = 0.95
    schema_version: str = "object_pose_v1"


def select_most_displaced_freejoint(
    predicate: Callable[[str], bool] | None = None,
) -> Callable[[ET.Element, np.ndarray], str | None]:
    """Return a selector that picks the freejoint with largest 3D displacement."""

    def _layout(xml_root: ET.Element) -> list[dict[str, Any]]:
        qpos_index = 1
        entries = []
        for joint in xml_root.iter("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
            entries.append({
                "name": joint.attrib["name"],
                "type": joint_type,
                "nq": nq,
                "qpos_index": qpos_index,
            })
            qpos_index += nq
        return entries

    def _selector(xml_root: ET.Element, states: np.ndarray) -> str | None:
        layout = _layout(xml_root)
        candidates = [e for e in layout if e["type"] == "free"]
        if predicate is not None:
            candidates = [e for e in candidates if predicate(e["name"])]
        if not candidates:
            return None
        best = None
        best_disp = -1.0
        for entry in candidates:
            qpos_idx = entry["qpos_index"]
            pos = states[:, qpos_idx:qpos_idx + 3]
            disp = float(np.linalg.norm(pos[-1] - pos[0]))
            if disp > best_disp:
                best_disp = disp
                best = entry["name"]
        return best

    return _selector


class ObjectPoseTaskBuilder(AtomicTaskLabelBuilder):
    """Builder for tasks whose progress is the 3D displacement of a manipulated object."""

    def __init__(self, schema: ObjectPoseTaskSchema):
        self._schema = schema

    @property
    def task_name(self) -> str:
        return self._schema.task_name

    @property
    def schema(self) -> ObjectPoseTaskSchema:
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
        if model_path is None:
            model_path = states_path.parent / "model.xml.gz"
        if ep_meta_path is None:
            ep_meta_path = states_path.parent / "ep_meta.json"

        xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
        states = np.load(states_path, allow_pickle=True)["states"]

        joint_name = self.schema.object_joint_selector(xml_root, states)
        if joint_name is None:
            raise ValueError(f"Could not resolve object freejoint for task {self.task_name}")

        # Locate the freejoint's qpos index.
        qpos_index = 1
        found = False
        for joint in xml_root.iter("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
            if joint.attrib["name"] == joint_name:
                found = True
                break
            qpos_index += nq
        if not found:
            raise ValueError(f"Joint {joint_name} not found in XML")

        positions = states[:, qpos_index:qpos_index + 3].astype(np.float64)
        start_pos = positions[0]
        end_pos = positions[-1]
        total_disp = float(np.linalg.norm(end_pos - start_pos))
        if total_disp <= 1e-6:
            progress = np.zeros(positions.shape[0], dtype=np.float64)
        else:
            disp_from_start = np.linalg.norm(positions - start_pos, axis=1)
            progress = np.clip(disp_from_start / total_disp, 0.0, 1.0)
            # Cumulative max makes progress monotonic.
            progress = np.maximum.accumulate(progress)

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Object-pose label cache requires pyarrow.") from exc

        table = pq.read_table(parquet_path, columns=["next.reward", "next.done", "frame_index"])
        data = table.to_pydict()
        rewards = np.asarray(data["next.reward"], dtype=np.float64)
        dones = np.asarray(data["next.done"], dtype=bool)
        frame_indices = np.asarray(data.get("frame_index", range(len(rewards))), dtype=np.int64)

        row_count = min(len(rewards), len(dones), len(frame_indices), int(states.shape[0]))
        rewards = rewards[:row_count]
        dones = dones[:row_count]
        frame_indices = frame_indices[:row_count]
        progress = progress[:row_count]
        positions = positions[:row_count]

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
            "object_position": positions.astype(np.float64),
            "failure_risk": failure_risk_values,
            "failure_risk_mask": failure_risk_masks,
            "subgoal_feasibility": subgoal_feasibility_values,
            "subgoal_feasibility_mask": subgoal_feasibility_masks,
            "manipulation_readiness": manipulation_readiness_values,
            "manipulation_readiness_mask": manipulation_readiness_masks,
            "row_count": row_count,
            "object_joint_name": joint_name,
            "schema_version": self.schema.schema_version,
            "readiness_schema_version": f"{self.task_name.lower()}_object_displacement_imminence_v1",
            "readiness_predicate": f"future object displacement progress >= {readiness_progress_delta}",
            "kinematics_available": False,
            "kinematics_message": "object-pose builder uses freejoint qpos only",
        }
