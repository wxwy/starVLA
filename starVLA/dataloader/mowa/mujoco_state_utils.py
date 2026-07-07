"""Shared MuJoCo state-loading and forward-kinematics utilities for MoWA.

These functions are task-agnostic: they know how to parse a RoboCasa episode's
``model.xml.gz`` / ``states.npz`` / ``ep_meta.json`` and run deterministic
forward kinematics, but they do not encode any task-specific joint/site naming
rules or success predicates.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np


def import_mujoco() -> Any:
    """Lazily import mujoco so that non-robocasa interpreters can import this module."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo utilities require mujoco.") from exc
    return mujoco


def patch_model_xml_paths(xml_text: str, repo_root: Path | None = None) -> str:
    """Patch absolute asset paths embedded in RoboCasa episode XML.

    Saved episode XMLs contain absolute paths from the machine that generated
    them.  We rewrite the two known prefixes to the current environment.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    try:
        import robosuite

        robosuite_root = Path(robosuite.__file__).parent
    except Exception:  # pragma: no cover - fallback to relative search
        robosuite_root = repo_root / ".robocase" / "lib" / "python3.11" / "site-packages" / "robosuite"

    robocasa_assets = (repo_root / "playground" / "Code" / "robocasa365" / "robocasa" / "models" / "assets").resolve()

    patched = xml_text.replace(
        "/opt/conda/envs/robocasa/lib/python3.10/site-packages/robosuite",
        str(robosuite_root),
    ).replace(
        "/root/robocasa/robocasa/models/assets",
        str(robocasa_assets),
    )

    # If any absolute file references are still missing, raise early so that
    # the caller knows which prefix needs patching.
    missing: list[str] = []
    for match in re.finditer(r'file=["\']([^"\']+)["\']', patched):
        path_str = match.group(1)
        if path_str.startswith("/") and not Path(path_str).exists():
            missing.append(path_str)
    if missing:
        sample = "\n  ".join(sorted(set(missing))[:8])
        raise FileNotFoundError(
            f"MuJoCo XML still has {len(set(missing))} missing absolute asset paths, e.g.:\n  {sample}"
        )

    return patched


def resolve_asset_paths_for_episode(
    model_path: Path,
    repo_root: Path | None = None,
) -> str:
    """Read and patch a single episode model.xml.gz."""
    xml_text = gzip.decompress(model_path.read_bytes()).decode("utf-8", errors="ignore")
    return patch_model_xml_paths(xml_text, repo_root=repo_root)


def build_mujoco_model_from_episode(
    model_path: Path,
    repo_root: Path | None = None,
) -> Any:
    """Build an MjModel from a patched episode model.xml.gz."""
    mujoco = import_mujoco()
    patched_xml = resolve_asset_paths_for_episode(model_path, repo_root=repo_root)
    return mujoco.MjModel.from_xml_string(patched_xml)


def ordered_joint_state_layout(xml_root: ET.Element) -> list[dict[str, Any]]:
    """Reconstruct the flattened [time | qpos | qvel] layout from MuJoCo XML.

    Returns a list of joint entries with qpos/qvel start indices (1-based,
    because column 0 is time).  This mirrors the convention used by
    ``states.npz`` saved by RoboCasa.
    """
    qpos_index = 1
    qvel_index = 1
    entries = []
    for joint in xml_root.iter("joint"):
        joint_type = joint.attrib.get("type", "hinge")
        nq = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
        nv = 6 if joint_type == "free" else 3 if joint_type == "ball" else 1
        range_text = joint.attrib.get("range", "")
        range_value = tuple(float(value) for value in range_text.split()) if range_text else ()
        entries.append(
            {
                "name": joint.attrib["name"],
                "type": joint_type,
                "nq": nq,
                "nv": nv,
                "qpos_index": qpos_index,
                "qvel_index": qvel_index,
                "range": range_value,
            }
        )
        qpos_index += nq
        qvel_index += nv
    return entries


def resolve_joint_state_indices(
    xml_root: ET.Element,
    joint_name: str,
) -> dict[str, int]:
    """Return the qpos/qvel column indices in ``states.npz`` for a named joint.

    The returned indices are 0-based with respect to the full ``states`` array,
    i.e. they already account for the leading time column.
    """
    layout = ordered_joint_state_layout(xml_root)
    for entry in layout:
        if entry["name"] == joint_name:
            inferred_nq = sum(int(e["nq"]) for e in layout)
            return {
                "qpos_index": int(entry["qpos_index"]),
                "qvel_index": 1 + inferred_nq + int(entry["qvel_index"]) - 1,
            }
    raise ValueError(f"Joint {joint_name} not found in XML")


def find_body_id(model: Any, body_name: str) -> int | None:
    """Safe wrapper around mj_name2id for bodies."""
    mujoco = import_mujoco()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return int(bid) if bid >= 0 else None


def find_site_id(model: Any, site_name: str) -> int:
    """Resolve a site name to a MuJoCo site id; raise if missing."""
    mujoco = import_mujoco()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site not found in XML: {site_name}")
    return int(site_id)


def collect_gripper_body_ids(model: Any) -> set[int]:
    """Return body IDs whose names contain 'gripper' or 'finger'."""
    mujoco = import_mujoco()
    ids: set[int] = set()
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name and ("gripper" in name.lower() or "finger" in name.lower()):
            ids.add(bid)
    return ids


def iter_forward_kinematics(
    model: Any,
    states: np.ndarray,
) -> Generator[Any, None, None]:
    """Yield per-row MjData after running mj_forward.

    ``states`` must have shape (T, 1 + nq + nv) where column 0 is time,
    columns [1, 1+nq) are qpos, and columns [1+nq, 1+nq+nv) are qvel.
    """
    mujoco = import_mujoco()
    nq = model.nq
    nv = model.nv
    data = mujoco.MjData(model)
    for t in range(int(states.shape[0])):
        data.time = float(states[t, 0])
        data.qpos[:] = states[t, 1 : 1 + nq]
        data.qvel[:] = states[t, 1 + nq : 1 + nq + nv]
        mujoco.mj_forward(model, data)
        yield data


def compute_site_distances(
    model: Any,
    states: np.ndarray,
    site_a_name: str,
    site_b_name: str,
) -> np.ndarray:
    """Return array of Euclidean distances between two sites over time."""
    site_a_id = find_site_id(model, site_a_name)
    site_b_id = find_site_id(model, site_b_name)
    row_count = int(states.shape[0])
    distances = np.zeros(row_count, dtype=np.float64)
    for t, data in enumerate(iter_forward_kinematics(model, states)):
        pos_a = data.site_xpos[site_a_id]
        pos_b = data.site_xpos[site_b_id]
        distances[t] = float(np.linalg.norm(pos_a - pos_b))
    return distances


def detect_body_contact(
    model: Any,
    states: np.ndarray,
    body_a_predicate: Callable[[str], bool],
    body_b_predicate: Callable[[str], bool],
) -> np.ndarray:
    """Return bool array indicating contact between bodies matching predicates.

    A body name matches if the callable returns True.  The contact is recorded
    when one body in the pair matches ``body_a_predicate`` and the other matches
    ``body_b_predicate``.
    """
    body_a_ids = {bid for bid in range(model.nbody) if body_a_predicate(_body_name(model, bid))}
    body_b_ids = {bid for bid in range(model.nbody) if body_b_predicate(_body_name(model, bid))}
    if not body_a_ids or not body_b_ids:
        return np.zeros(int(states.shape[0]), dtype=bool)

    row_count = int(states.shape[0])
    contact_flags = np.zeros(row_count, dtype=bool)
    for t, data in enumerate(iter_forward_kinematics(model, states)):
        for c in data.contact[: data.ncon]:
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            pair = {b1, b2}
            if (pair & body_a_ids) and (pair & body_b_ids):
                contact_flags[t] = True
                break
    return contact_flags


def _body_name(model: Any, body_id: int) -> str | None:
    """Return the name of a body id, or None if invalid."""
    mujoco = import_mujoco()
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def load_ep_meta(ep_meta_path: Path) -> dict[str, Any]:
    """Load and parse ep_meta.json."""
    return json.loads(ep_meta_path.read_text(encoding="utf-8"))


def load_states(states_path: Path) -> np.ndarray:
    """Load the ``states`` array from a ``states.npz`` file."""
    return np.load(states_path, allow_pickle=True)["states"]
