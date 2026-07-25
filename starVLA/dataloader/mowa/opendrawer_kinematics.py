"""MuJoCo forward-kinematics utilities for OpenDrawer handle/EEF extraction.

.. deprecated::
    The reusable helpers have moved to ``mujoco_state_utils.py``.  This module
    is kept as a thin backward-compatibility wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.mowa.mujoco_state_utils import (
    build_mujoco_model_from_episode,
    collect_gripper_body_ids,
    compute_site_distances,
    find_body_id,
    find_site_id,
    import_mujoco,
    load_ep_meta,
    load_states,
    patch_model_xml_paths,
    resolve_asset_paths_for_episode,
)


def _import_mujoco() -> Any:
    """Backward-compatible alias for ``import_mujoco``."""
    return import_mujoco()


def _patch_model_xml_paths(xml_text: str, repo_root: Path | None = None) -> str:
    """Backward-compatible alias for ``patch_model_xml_paths``."""
    return patch_model_xml_paths(xml_text, repo_root=repo_root)


def _resolve_asset_paths_for_episode(
    model_path: Path,
    repo_root: Path | None = None,
) -> str:
    """Backward-compatible alias for ``resolve_asset_paths_for_episode``."""
    return resolve_asset_paths_for_episode(model_path, repo_root=repo_root)


def extract_opendrawer_kinematics(
    model_path: Path,
    states_path: Path,
    drawer_ref: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run MuJoCo FK and extract handle/EEF kinematics + contact flags.

    Deprecated: this logic now lives in ``tasks/opendrawer.py``; the function
    is preserved for existing callers.
    """
    mujoco = import_mujoco()
    model = build_mujoco_model_from_episode(model_path, repo_root=repo_root)
    states = load_states(states_path)
    row_count = int(states.shape[0])

    handle_site_name = f"{drawer_ref}_door_handle_default_site"
    eef_site_name = "gripper0_right_grip_site"
    distances = compute_site_distances(model, states, handle_site_name, eef_site_name)

    handle_site_id = find_site_id(model, handle_site_name)
    handle_body_id = int(model.site_bodyid[handle_site_id])
    handle_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, handle_body_id)
    door_body_name = f"{drawer_ref}_door_main"

    gripper_body_ids = collect_gripper_body_ids(model)

    gripper_handle_contact = np.zeros(row_count, dtype=bool)
    gripper_finger_contact = np.zeros(row_count, dtype=bool)

    from starVLA.dataloader.mowa.mujoco_state_utils import iter_forward_kinematics

    for t, data in enumerate(iter_forward_kinematics(model, states)):
        for c in data.contact[: data.ncon]:
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            pair = {b1, b2}
            if handle_body_id in pair and bool(pair & gripper_body_ids):
                gripper_handle_contact[t] = True
            if door_body_name and find_body_id(model, door_body_name) in pair and bool(pair & gripper_body_ids):
                gripper_finger_contact[t] = True

    return {
        "handle_world_position": None,  # No longer precomputed; caller can use FK directly.
        "eef_world_position": None,
        "eef_to_handle_distance": distances,
        "gripper_handle_contact": gripper_handle_contact,
        "gripper_finger_contact": gripper_finger_contact,
        "states_row_count": row_count,
        "message": "ok",
    }


def _find_body_id(model: Any, body_name: str) -> int | None:
    """Backward-compatible alias for ``find_body_id``."""
    return find_body_id(model, body_name)


def _collect_gripper_body_ids(model: Any) -> set[int]:
    """Backward-compatible alias for ``collect_gripper_body_ids``."""
    return collect_gripper_body_ids(model)
