"""Reusable single-DOF task builder for hinge/slide/knob atomic tasks.

This module provides ``SingleDofTaskLabelBuilder``, a configurable subclass of
``AtomicTaskLabelBuilder`` that covers the common case where a task's success
predicate is defined by a single joint angle/position (door, drawer, lid, rack,
knob, etc.) and readiness is defined by EEF proximity to a handle/site.
"""

from __future__ import annotations

import gzip
import json
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
from starVLA.dataloader.mowa.mujoco_state_utils import (
    build_mujoco_model_from_episode,
    compute_site_distances,
    find_body_id,
    find_site_id,
    import_mujoco,
    load_ep_meta,
    load_states,
    ordered_joint_state_layout,
    resolve_joint_state_indices,
)


@dataclass(frozen=True)
class SingleDofTaskSchema:
    """Schema describing a single-DOF atomic task.

    Fields ending in ``_template`` may use ``{fixture_ref}`` interpolation.
    """

    task_name: str
    # How to find the fixture reference in ep_meta.json.
    fixture_ref_key: str | None = None
    # Callable taking ep_meta and returning fixture_ref string (overrides key).
    fixture_resolver: Callable[[dict[str, Any]], str | None] | None = None
    # Joint name template, e.g. "{fixture_ref}_doorhinge".
    joint_name_template: str | None = None
    # Callable to dynamically pick a joint name from the XML, given states.
    joint_selector: Callable[[ET.Element, np.ndarray], str] | None = None
    # Handle/site for proximity-based readiness.
    handle_site_template: str | None = None
    # Body that counts as the manipulated part for contact debug (optional).
    manipulated_body_template: str | None = None
    # Normalize raw qpos to [0, 1] progress.  If None, use raw qpos directly.
    progress_normalizer: Callable[[np.ndarray, tuple[float, ...]], np.ndarray] | None = None
    # If True, flip the normalized progress so that the task goal is 1.0.
    # Use this for "closing / turning off / sliding in" tasks where the default
    # normalizer would make the goal state 0.0.
    invert_progress: bool = False
    # Completion threshold in normalized (or raw) progress units.
    completion_threshold: float = 0.95
    # Readiness proximity threshold in meters.
    readiness_distance_threshold: float = 0.05
    # Fallback progress-imminence delta when kinematics unavailable.
    readiness_progress_delta: float = 0.1
    # Schema version string for audit metadata.
    schema_version: str = "single_dof_v1"
    # Subgoal id.
    subgoal_id: str = "complete"


def _default_slide_normalizer(raw_qpos: np.ndarray, joint_range: tuple[float, ...]) -> np.ndarray:
    """Normalize a slide joint so that fully closed = 0 and fully open = 1."""
    if not joint_range:
        return raw_qpos
    lo, hi = float(joint_range[0]), float(joint_range[1])
    denom = abs(hi - lo)
    if denom <= 0:
        return raw_qpos
    # RoboCasa drawers: qpos is negative when open.  Use same formula as OpenDrawer.
    return (-raw_qpos) / (0.55 * abs(lo) / 2)


def _default_hinge_normalizer(raw_qpos: np.ndarray, joint_range: tuple[float, ...]) -> np.ndarray:
    """Normalize a hinge joint so that closed = 0 and open = 1."""
    if not joint_range:
        return raw_qpos
    lo, hi = float(joint_range[0]), float(joint_range[1])
    denom = abs(hi - lo)
    if denom <= 0:
        return raw_qpos
    return (raw_qpos - lo) / denom


class SingleDofTaskBuilder(AtomicTaskLabelBuilder):
    """Configurable builder for single-DOF atomic tasks."""

    def __init__(self, schema: SingleDofTaskSchema):
        self._schema = schema

    @property
    def task_name(self) -> str:
        return self._schema.task_name

    @property
    def schema(self) -> SingleDofTaskSchema:
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

        if not ep_meta_path.is_file():
            raise FileNotFoundError(f"Episode extras missing: {ep_meta_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Episode extras missing: {model_path}")

        ep_meta = load_ep_meta(ep_meta_path)
        xml_root = ET.fromstring(gzip.decompress(model_path.read_bytes()))
        states = load_states(states_path)

        fixture_ref = self._resolve_fixture_ref(ep_meta)
        joint_name = self._resolve_joint_name(xml_root, states, fixture_ref)
        if joint_name is None:
            raise ValueError(
                f"Could not resolve joint name for task {self.task_name} "
                f"(fixture_ref={fixture_ref})"
            )

        indices = resolve_joint_state_indices(xml_root, joint_name)
        joint_entry = next(
            entry for entry in ordered_joint_state_layout(xml_root) if entry["name"] == joint_name
        )
        joint_range = tuple(float(value) for value in joint_entry["range"])
        raw_qpos = states[:, int(indices["qpos_index"])]

        normalizer = self.schema.progress_normalizer
        if normalizer is None:
            normalizer = _default_hinge_normalizer
        progress = normalizer(raw_qpos, joint_range)
        progress = np.asarray(progress, dtype=np.float64)
        if self.schema.invert_progress:
            progress = 1.0 - progress

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Single-DOF label cache requires pyarrow.") from exc

        table = pq.read_table(parquet_path, columns=["next.reward", "next.done", "frame_index"])
        data = table.to_pydict()
        rewards = [float(value) for value in data["next.reward"]]
        dones = [bool(value) for value in data["next.done"]]
        frame_indices = [int(value) for value in data.get("frame_index", range(len(rewards)))]

        row_count = min(len(rewards), len(dones), len(frame_indices), int(states.shape[0]))
        rewards = rewards[:row_count]
        dones = dones[:row_count]
        frame_indices = frame_indices[:row_count]
        progress = progress[:row_count]
        raw_qpos = raw_qpos[:row_count]

        kinematics = _maybe_extract_kinematics(
            model_path=model_path,
            states_path=states_path,
            fixture_ref=fixture_ref,
            handle_site_template=self.schema.handle_site_template,
            manipulated_body_template=self.schema.manipulated_body_template,
            enable=enable_kinematics,
            row_count=row_count,
            repo_root=repo_root,
        )

        failure_risk_values = np.zeros(row_count, dtype=np.float64)
        failure_risk_masks = np.ones(row_count, dtype=bool)
        subgoal_feasibility_values, subgoal_feasibility_masks = compute_subgoal_feasibility(
            progress=progress,
            completion_threshold=self.schema.completion_threshold,
            horizon=subgoal_horizon,
        )
        manipulation_readiness_values = np.zeros(row_count, dtype=np.float64)
        manipulation_readiness_masks = np.zeros(row_count, dtype=bool)

        for timestep in range(row_count):
            current_progress = float(progress[timestep])
            completed_now = current_progress >= self.schema.completion_threshold

            failure_risk = failure_risk_label_from_window(
                rewards=rewards,
                dones=dones,
                anchor=timestep,
                horizon=failure_risk_horizon,
            )
            failure_risk_masks[timestep] = failure_risk is None
            failure_risk_values[timestep] = float(failure_risk) if failure_risk is not None else 0.0

            # Override subgoal mask logic: if already completed, mask.
            subgoal_feasibility_masks[timestep] = completed_now
            if not completed_now and subgoal_feasibility_values[timestep] > 0.0:
                subgoal_feasibility_values[timestep] = 1.0

            manipulation_readiness_masks[timestep] = completed_now
            if not completed_now:
                if kinematics["available"]:
                    dist = float(kinematics["eef_to_handle_distance"][timestep])
                    manipulation_readiness_values[timestep] = 1.0 if dist <= readiness_distance_threshold else 0.0
                else:
                    readiness_future = progress[timestep + 1 : timestep + 1 + readiness_horizon]
                    if readiness_future.size == 0:
                        manipulation_readiness_masks[timestep] = True
                    else:
                        future_gain = readiness_future - current_progress
                        if np.any(future_gain >= readiness_progress_delta):
                            manipulation_readiness_values[timestep] = 1.0

        result: dict[str, Any] = {
            "frame_index": np.asarray(frame_indices, dtype=np.int64),
            "task_progress": progress.astype(np.float64),
            "raw_joint_qpos": raw_qpos.astype(np.float64),
            "failure_risk": failure_risk_values,
            "failure_risk_mask": failure_risk_masks,
            "subgoal_feasibility": subgoal_feasibility_values,
            "subgoal_feasibility_mask": subgoal_feasibility_masks,
            "manipulation_readiness": manipulation_readiness_values,
            "manipulation_readiness_mask": manipulation_readiness_masks,
            "row_count": row_count,
            "joint_name": joint_name,
            "joint_range": joint_range,
            "schema_version": self.schema.schema_version,
            "readiness_schema_version": (
                f"{self.task_name.lower()}_proximity_v1"
                if kinematics["available"]
                else f"{self.task_name.lower()}_progress_imminence_v1"
            ),
            "readiness_predicate": (
                f"eef_to_handle_distance <= {readiness_distance_threshold}"
                if kinematics["available"]
                else f"future progress gain >= {readiness_progress_delta}"
            ),
            "kinematics_available": kinematics["available"],
            "kinematics_message": kinematics["message"],
        }
        if kinematics["available"]:
            result["eef_to_handle_distance"] = kinematics["eef_to_handle_distance"].astype(np.float64)
        return result

    def _resolve_fixture_ref(self, ep_meta: dict[str, Any]) -> str | None:
        if self.schema.fixture_resolver is not None:
            return self.schema.fixture_resolver(ep_meta)
        if self.schema.fixture_ref_key is not None:
            return ep_meta.get("fixture_refs", {}).get(self.schema.fixture_ref_key)
        return None

    def _resolve_joint_name(
        self,
        xml_root: ET.Element,
        states: np.ndarray,
        fixture_ref: str | None,
    ) -> str | None:
        if self.schema.joint_selector is not None:
            return self.schema.joint_selector(xml_root, states)
        if self.schema.joint_name_template is not None and fixture_ref is not None:
            return self.schema.joint_name_template.format(fixture_ref=fixture_ref)
        return None


def select_most_displaced_joint(pattern: str) -> Callable[[ET.Element, np.ndarray], str | None]:
    """Return a joint selector that picks the matching joint with largest qpos displacement.

    ``pattern`` is matched case-insensitively against joint names.  If no joint
    matches, returns None.
    """
    return select_most_displaced_joint_matching_all((pattern,))


def select_most_displaced_joint_matching_all(
    patterns: tuple[str, ...],
) -> Callable[[ET.Element, np.ndarray], str | None]:
    """Return a joint selector that picks the joint matching ALL patterns with
    the largest qpos displacement."""

    def _selector(xml_root: ET.Element, states: np.ndarray) -> str | None:
        layout = ordered_joint_state_layout(xml_root)
        candidates = [
            entry for entry in layout
            if all(pattern in entry["name"].lower() for pattern in patterns)
        ]
        if not candidates:
            return None
        best_joint = None
        best_disp = -1.0
        for entry in candidates:
            qpos = states[:, int(entry["qpos_index"])]
            disp = float(np.max(np.abs(qpos - qpos[0])))
            if disp > best_disp:
                best_disp = disp
                best_joint = entry["name"]
        return best_joint

    return _selector


def _maybe_extract_kinematics(
    model_path: Path,
    states_path: Path,
    fixture_ref: str | None,
    handle_site_template: str | None,
    manipulated_body_template: str | None,
    enable: bool,
    row_count: int,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Return EEF-to-handle distance if MuJoCo and site names are available."""
    if not enable or handle_site_template is None or fixture_ref is None:
        return {
            "available": False,
            "message": "kinematics disabled or no handle site",
            "eef_to_handle_distance": np.zeros(row_count, dtype=np.float64),
        }

    if not model_path.is_file():
        return {
            "available": False,
            "message": f"missing {model_path.name}",
            "eef_to_handle_distance": np.zeros(row_count, dtype=np.float64),
        }

    try:
        model = build_mujoco_model_from_episode(model_path, repo_root=repo_root)
        states = load_states(states_path)
        handle_site_name = handle_site_template.format(fixture_ref=fixture_ref)
        distances = compute_site_distances(
            model=model,
            states=states,
            site_a_name=handle_site_name,
            site_b_name="gripper0_right_grip_site",
        )
        return {
            "available": True,
            "message": "ok",
            "eef_to_handle_distance": distances,
        }
    except Exception as exc:  # pragma: no cover - runtime fallback
        return {
            "available": False,
            "message": f"kinematics failed: {exc}",
            "eef_to_handle_distance": np.zeros(row_count, dtype=np.float64),
        }
