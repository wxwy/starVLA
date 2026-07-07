"""MoWA future-label builder for the TurnOnSinkFaucet atomic task."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from starVLA.dataloader.mowa.atomic_task_label_builder import register_atomic_task_label_builder
from starVLA.dataloader.mowa.mujoco_state_utils import ordered_joint_state_layout
from starVLA.dataloader.mowa.tasks._single_dof import SingleDofTaskBuilder, SingleDofTaskSchema


def _select_sink_handle_joint(xml_root: ET.Element, states: np.ndarray) -> str | None:
    """Select the sink faucet handle joint (not the temperature handle)."""
    layout = ordered_joint_state_layout(xml_root)
    candidates = [
        entry for entry in layout
        if "sink" in entry["name"].lower()
        and "handle_joint" in entry["name"].lower()
        and "temp" not in entry["name"].lower()
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
class TurnOnSinkFaucetLabelBuilder(SingleDofTaskBuilder):
    """Build future labels for TurnOnSinkFaucet using sink handle joint progress."""

    def __init__(self) -> None:
        super().__init__(
            SingleDofTaskSchema(
                task_name="TurnOnSinkFaucet",
                fixture_ref_key=None,
                joint_selector=_select_sink_handle_joint,
                handle_site_template=None,
                manipulated_body_template=None,
                completion_threshold=0.95,
                readiness_distance_threshold=0.05,
                readiness_progress_delta=0.05,
                schema_version="turn_on_sink_faucet_v1",
                subgoal_id="turn_on_sink_faucet",
            )
        )
