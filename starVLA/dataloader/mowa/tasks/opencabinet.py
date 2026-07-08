"""MoWA future-label builder for the OpenCabinet atomic task.

OpenCabinet episodes may feature a single cabinet door or a pair of doors, and
the fixture reference varies by episode.  We resolve the fixture ref from
``ep_meta.json`` and aggregate progress over all matching door-hinge joints.
"""

from __future__ import annotations

import gzip
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
from starVLA.dataloader.mowa.mujoco_state_utils import build_mujoco_model_from_episode, load_ep_meta, load_states
from starVLA.dataloader.mowa.visual_head_utils import compute_visual_head_labels


def _hinge_progress_open(raw_qpos: np.ndarray, joint_range: tuple[float, ...]) -> np.ndarray:
    """Normalize a hinge joint so that closed = 0 and open = 1."""
    if not joint_range:
        return raw_qpos
    lo, hi = float(joint_range[0]), float(joint_range[1])
    denom = abs(hi - lo)
    if denom <= 0:
        return raw_qpos
    return (raw_qpos - lo) / denom


@register_atomic_task_label_builder
class OpenCabinetLabelBuilder(AtomicTaskLabelBuilder):
    """Build future labels for OpenCabinet using per-episode cabinet door progress."""

    default_readiness_horizon: int = 10
    default_readiness_progress_delta: float = 0.05

    @property
    def task_name(self) -> str:
        return "OpenCabinet"

    def _find_cabinet_door_joints(
        self,
        xml_root: ET.Element,
        fixture_ref: str,
    ) -> list[dict[str, Any]]:
        """Return all hinge joints belonging to the target cabinet doors."""
        joints: list[dict[str, Any]] = []
        qpos_index = 1
        fixture_ref_lower = fixture_ref.lower()
        for joint in xml_root.iter("joint"):
            joint_type = joint.attrib.get("type", "hinge")
            nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
            name = joint.attrib.get("name", "")
            name_lower = name.lower()
            if (
                joint_type == "hinge"
                and name_lower.startswith(fixture_ref_lower)
                and "doorhinge" in name_lower
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

        ep_meta = load_ep_meta(ep_meta_path)
        fixture_ref = ep_meta.get("fixture_refs", {}).get("fxtr")
        if not fixture_ref:
            raise ValueError(f"OpenCabinet missing fixture_refs.fxtr in {ep_meta_path}")

        xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
        states = load_states(states_path)
        model = build_mujoco_model_from_episode(model_path, repo_root=repo_root)
        door_joints = self._find_cabinet_door_joints(xml_root, fixture_ref)
        if not door_joints:
            raise ValueError(
                f"No cabinet door joints found for {self.task_name} "
                f"(fixture_ref={fixture_ref})"
            )
        # Resolve door body ids for visual target computation.
        door_body_ids: list[int] = []
        for joint in door_joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if joint_id >= 0:
                door_body_ids.append(int(model.jnt_bodyid[joint_id]))

        per_door_progress: list[np.ndarray] = []
        for joint in door_joints:
            raw_qpos = states[:, joint["qpos_index"]]
            progress = _hinge_progress_open(raw_qpos, joint["range"])
            per_door_progress.append(np.asarray(progress, dtype=np.float64))
        overall_progress = np.mean(np.stack(per_door_progress, axis=0), axis=0)

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("OpenCabinet label cache requires pyarrow.") from exc

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

        def _target_resolver(data: Any, timestep: int) -> np.ndarray | None:
            del timestep
            # Prefer the door handle site; it is a much better visual proxy than
            # the hinge pivot body center, which can sit outside the camera view.
            handle_site_names = [
                f"{joint['name'].replace('doorhinge', 'door_handle_default_site')}"
                for joint in door_joints
            ]
            handle_positions: list[np.ndarray] = []
            for site_name in handle_site_names:
                try:
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        handle_positions.append(data.site_xpos[site_id].copy())
                except Exception:
                    continue
            if handle_positions:
                return np.mean(np.stack(handle_positions, axis=0), axis=0)
            if not door_body_ids:
                return None
            positions = np.stack([data.xpos[bid].copy() for bid in door_body_ids], axis=0)
            return np.mean(positions, axis=0)

        def _target_body_id_resolver(data: Any, timestep: int) -> int | None:
            del data, timestep
            # Exclude the first door body from occlusion checks.  Multi-door cabinets
            # are handled approximately; the other door body remains a valid occluder.
            return door_body_ids[0] if door_body_ids else None

        visual_labels = compute_visual_head_labels(
            model=model,
            states=states,
            ep_meta=ep_meta,
            target_point_resolver=_target_resolver,
            target_body_id_resolver=_target_body_id_resolver,
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
            "schema_version": "opencabinet_open_door_v2",
            "readiness_schema_version": "opencabinet_progress_imminence_v1",
            "readiness_predicate": f"future progress gain >= {readiness_progress_delta}",
            "kinematics_available": False,
            "kinematics_message": "progress-imminence proxy for multi-door cabinet",
            "object_visibility_future": visual_labels["object_visibility_future"],
            "object_visibility_future_mask": visual_labels["object_visibility_future_mask"],
            "next_best_view_score": visual_labels["next_best_view_score"],
            "next_best_view_score_mask": visual_labels["next_best_view_score_mask"],
            "visual_head_schema_version": visual_labels.get("schema_version", "mowa_visual_proxy_v1"),
        }
