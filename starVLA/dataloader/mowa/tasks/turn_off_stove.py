"""MoWA future-label builder for the TurnOffStove atomic task."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.mujoco_state_utils import ordered_joint_state_layout
from starVLA.dataloader.mowa.tasks._single_dof import SingleDofTaskBuilder, SingleDofTaskSchema


def _select_active_burner_knob(xml_root: ET.Element, states: np.ndarray) -> str | None:
    """Select the stove burner knob with the largest qpos displacement.

    RoboCasa episode XMLs use inconsistent naming conventions:
    ``stove_main_group_knob_*``, ``stovetop_main_group_knob_*``,
    ``stove_right_group_knob_*``, etc.  We match any hinge joint whose name
    contains ``stove`` or ``stovetop`` plus ``knob``, then exclude the
    temperature / timer / function knobs and pick the burner knob that moves
    the most.
    """

    layout = ordered_joint_state_layout(xml_root)
    excluded = ("temp", "time", "doneness", "function")
    candidates = [
        entry
        for entry in layout
        if entry["type"] == "hinge"
        and ("stove" in entry["name"].lower() or "stovetop" in entry["name"].lower())
        and "knob" in entry["name"].lower()
        and not any(excl in entry["name"].lower() for excl in excluded)
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


@register_atomic_task_label_builder
class TurnOffStoveLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for TurnOffStove using the active burner knob progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="TurnOffStove",
                fixture_ref_key=None,
                joint_selector=_select_active_burner_knob,
                handle_site_template=None,
                manipulated_body_template=None,
                invert_progress=True,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="turn_off_stove_v2",
                subgoal_id="turn_off_stove",
            )
        )
