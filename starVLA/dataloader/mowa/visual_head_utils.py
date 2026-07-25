"""MuJoCo-based visual future-head utilities for MoWA.

These functions compute proxy labels for ``object_visibility_future`` and
``next_best_view_score`` by projecting a task-specific 3D target point into the
available camera views.  They do not decode video frames; all visibility metrics
are derived from the episode ``model.xml.gz`` and ``states.npz`` via forward
kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Camera intrinsics / extrinsics resolved for a single timestep."""

    name: str
    cam_id: int | None  # MuJoCo camera id, or None for virtual cameras from ep_meta
    width: int
    height: int
    fovy: float
    focal_length: float
    principal_point: tuple[float, float]


def quat_to_rotation_matrix(quat: tuple[float, float, float, float]) -> np.ndarray:
    """Convert a quaternion ``(w, x, y, z)`` to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def project_world_to_camera(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
) -> np.ndarray:
    """Transform a world point to the camera frame.

    MuJoCo / OpenGL cameras look down the negative Z axis, so visible points have
    ``p_cam[2] < 0``.
    """
    return cam_mat.T @ (np.asarray(point_world) - np.asarray(cam_pos))


def project_camera_to_image(
    p_cam: np.ndarray,
    camera: CameraConfig,
) -> tuple[float, float]:
    """Project a camera-frame point to pixel coordinates."""
    p_cam = np.asarray(p_cam)
    z = float(-p_cam[2])
    if z <= 0:
        raise ValueError("Point is behind the camera")
    u = camera.focal_length * float(p_cam[0]) / z + camera.principal_point[0]
    v = camera.focal_length * float(p_cam[1]) / z + camera.principal_point[1]
    return u, v


def compute_visibility_score(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    camera: CameraConfig,
    max_distance: float = 2.0,
    *,
    model: Any | None = None,
    data: Any | None = None,
    target_body_id: int | None = None,
    check_occlusion: bool = False,
) -> float:
    """Return a continuous visibility score in ``[0, 1]`` for one camera.

    The score is ``1 - distance/max_distance`` when the projected point lies
    inside the image and the camera-to-target distance is below ``max_distance``;
    otherwise it is ``0``.

    If ``check_occlusion`` is True and ``target_body_id`` is provided, a MuJoCo
    ray is cast from the camera toward the target point (excluding the target
    body).  If another geom is hit before the target distance, the point is
    considered occluded and the score is ``0``.
    """
    p_cam = project_world_to_camera(point_world, cam_pos, cam_mat)
    if p_cam[2] >= 0:
        return 0.0

    try:
        u, v = project_camera_to_image(p_cam, camera)
    except ValueError:
        return 0.0

    if not (0.0 <= u < camera.width and 0.0 <= v < camera.height):
        return 0.0

    distance = float(np.linalg.norm(point_world - cam_pos))
    if distance >= max_distance:
        return 0.0

    if check_occlusion and target_body_id is not None and model is not None and data is not None:
        import mujoco

        ray_vec = np.asarray(point_world - cam_pos, dtype=np.float64)
        hit_geom_id = np.array([-1], dtype=np.int32)
        # mj_ray returns distance along ray to the nearest surface, or -1 if no hit.
        hit_dist = mujoco.mj_ray(
            model,
            data,
            np.asarray(cam_pos, dtype=np.float64),
            ray_vec,
            None,
            True,
            -1,
            hit_geom_id,
            None,
        )
        if 0.0 <= hit_dist < distance - 1e-4:
            # Something is in front of the target point.
            hit_body_id = int(model.geom_bodyid[hit_geom_id[0]])
            if hit_body_id != int(target_body_id):
                return 0.0
            # If the target body itself is hit well before the target point,
            # the point lies behind the visible surface of the target object.
            # A 5cm tolerance is used because site targets (e.g. drawer handles)
            # can sit slightly in front of the underlying body surface.
            if hit_dist < distance - 0.05:
                return 0.0

    return max(0.0, 1.0 - distance / max_distance)


def build_camera_configs(
    model: Any,
    ep_meta: dict[str, Any],
    image_height: int = 256,
    image_width: int = 256,
) -> list[CameraConfig]:
    """Build camera configs from MuJoCo model cameras and ep_meta virtual cameras.

    MuJoCo model cameras are preferred when present; otherwise we fall back to
    the ``cam_configs`` entries in ``ep_meta.json`` (agentview_left/right,
    frontview, etc.).  Only cameras that have a matching video stream in the
    LeRobot dataset are useful, but this function returns all resolvable cameras
    so callers can choose which ones to evaluate.
    """
    import mujoco

    configs: list[CameraConfig] = []

    # 1) MuJoCo native cameras.
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        fovy = float(model.cam_fovy[cam_id])
        focal = (image_height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        configs.append(
            CameraConfig(
                name=name,
                cam_id=cam_id,
                width=image_width,
                height=image_height,
                fovy=fovy,
                focal_length=focal,
                principal_point=(image_width / 2.0, image_height / 2.0),
            )
        )

    # 2) Virtual cameras declared in ep_meta.  Skip if a native camera with the
    #    same name already exists.
    existing_names = {cfg.name for cfg in configs}
    cam_cfgs = ep_meta.get("cam_configs", {})
    for name, cam_cfg in cam_cfgs.items():
        if name in existing_names:
            continue
        parent_body = cam_cfg.get("parent_body")
        if parent_body is None:
            continue
        fovy = float(cam_cfg.get("camera_attribs", {}).get("fovy", 60))
        focal = (image_height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        configs.append(
            CameraConfig(
                name=name,
                cam_id=None,
                width=image_width,
                height=image_height,
                fovy=fovy,
                focal_length=focal,
                principal_point=(image_width / 2.0, image_height / 2.0),
            )
        )

    return configs


def resolve_virtual_camera_pose(
    data: Any,
    cam_cfg: dict[str, Any],
    model: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve world position and orientation of an ep_meta virtual camera."""
    import mujoco

    parent_body = cam_cfg.get("parent_body")
    if parent_body is None:
        return None
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_body)
    if parent_id < 0:
        return None

    rel_pos = np.array(cam_cfg.get("pos", [0.0, 0.0, 0.0]))
    rel_quat = tuple(cam_cfg.get("quat", [1.0, 0.0, 0.0, 0.0]))
    rel_mat = quat_to_rotation_matrix(rel_quat)

    parent_pos = data.xpos[parent_id].copy()
    parent_mat = data.xmat[parent_id].reshape(3, 3).copy()

    cam_pos = parent_pos + parent_mat @ rel_pos
    cam_mat = parent_mat @ rel_mat
    return cam_pos, cam_mat


def resolve_camera_pose(
    data: Any,
    camera: CameraConfig,
    ep_meta: dict[str, Any],
    model: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(cam_pos, cam_mat)`` for a camera config at the current data state."""
    if camera.cam_id is not None:
        cam_pos = data.cam_xpos[camera.cam_id].copy()
        cam_mat = data.cam_xmat[camera.cam_id].reshape(3, 3).copy()
        return cam_pos, cam_mat

    cam_cfgs = ep_meta.get("cam_configs", {})
    if camera.name not in cam_cfgs:
        return None
    return resolve_virtual_camera_pose(data, cam_cfgs[camera.name], model)


def compute_visual_head_labels(
    model: Any,
    states: np.ndarray,
    ep_meta: dict[str, Any],
    target_point_resolver: Callable[[Any, int], np.ndarray | None],
    progress: np.ndarray,
    completion_threshold: float,
    *,
    visibility_horizon: int = 10,
    main_camera_names: tuple[str, ...] = (
        "robot0_eye_in_hand",
    ),
    alternative_camera_names: tuple[str, ...] = (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_frontview",
        "robot0_robotview",
        "robot0_eye_in_hand",
    ),
    image_height: int = 256,
    image_width: int = 256,
    max_distance: float = 2.0,
    target_body_id_resolver: Callable[[Any, int], int | None] | None = None,
    occlusion_camera_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compute ``object_visibility_future`` and ``next_best_view_score`` labels.

    Args:
        model: Loaded MuJoCo MjModel.
        states: ``states.npz`` array of shape ``(T, 1 + nq + nv)``.
        ep_meta: Parsed ``ep_meta.json``.
        target_point_resolver: Callable taking ``(data, timestep)`` and returning
            the 3D world position of the visual target, or ``None`` if unavailable.
        progress: Per-timestep task progress in ``[0, 1]``; used to mask completed
            anchors the same way as ``subgoal_feasibility``.
        completion_threshold: Progress value above which the task is considered
            already completed.
        visibility_horizon: Number of future steps to look ahead.
        main_camera_names: Cameras used for ``object_visibility_future``; the target
            is considered visible if any of these cameras sees it.  Defaults to the
            eye-in-hand camera because it produces more discriminative visibility
            variation during manipulation than fixed workspace cameras.
        alternative_camera_names: Cameras considered for ``next_best_view_score``.
        image_height: Video frame height in pixels.
        image_width: Video frame width in pixels.
        max_distance: Maximum camera-to-target distance for a non-zero score.
        target_body_id_resolver: Optional callable returning the MuJoCo body id of
            the target object, used to exclude the target body from occlusion checks.
        occlusion_camera_names: Cameras for which occlusion is checked.  Defaults to
            ``main_camera_names``.  ``next_best_view_score`` uses unoccluded scores
            so that its continuous signal is not zeroed out by occlusion.

    Returns:
        Dict with arrays:
        - ``object_visibility_future`` (binary)
        - ``object_visibility_future_mask`` (True = masked)
        - ``next_best_view_score`` (continuous in ``[0, 1]``)
        - ``next_best_view_score_mask`` (True = masked)
    """
    row_count = int(states.shape[0])
    cameras = build_camera_configs(model, ep_meta, image_height, image_width)
    _occlusion_camera_names = (
        set(occlusion_camera_names) if occlusion_camera_names is not None else set(main_camera_names)
    )

    # Per-timestep visibility scores for each camera.  We do a single position-only
    # forward kinematics pass per timestep and evaluate all cameras at once.
    per_camera_scores: dict[str, np.ndarray] = {
        camera.name: np.zeros(row_count, dtype=np.float64) for camera in cameras
    }
    for t, target, target_body_id, camera_poses, data in _iter_forward_positions(
        model, states, ep_meta, cameras, target_point_resolver, target_body_id_resolver
    ):
        if target is None:
            continue
        for camera in cameras:
            pose = camera_poses.get(camera.name)
            if pose is None:
                continue
            cam_pos, cam_mat = pose
            per_camera_scores[camera.name][t] = compute_visibility_score(
                target,
                cam_pos,
                cam_mat,
                camera,
                max_distance,
                model=model,
                data=data,
                target_body_id=target_body_id,
                check_occlusion=camera.name in _occlusion_camera_names,
            )

    object_visibility_values = np.zeros(row_count, dtype=np.float64)
    object_visibility_masks = np.zeros(row_count, dtype=bool)
    next_best_view_values = np.zeros(row_count, dtype=np.float64)
    next_best_view_masks = np.zeros(row_count, dtype=bool)

    for t in range(row_count):
        completed_now = float(progress[t]) >= completion_threshold
        object_visibility_masks[t] = completed_now
        next_best_view_masks[t] = completed_now

        if completed_now:
            continue

        # object_visibility_future: binary, any main camera sees target within horizon.
        for name in main_camera_names:
            if name not in per_camera_scores:
                continue
            future_scores = per_camera_scores[name][t + 1 : t + 1 + visibility_horizon]
            if future_scores.size > 0 and np.any(future_scores > 0.0):
                object_visibility_values[t] = 1.0
                break

        # next_best_view_score: max visibility across alternative cameras.
        best_score = 0.0
        for name in alternative_camera_names:
            if name not in per_camera_scores:
                continue
            future_scores = per_camera_scores[name][t + 1 : t + 1 + visibility_horizon]
            if future_scores.size > 0:
                best_score = max(best_score, float(np.max(future_scores)))
        next_best_view_values[t] = best_score

    return {
        "object_visibility_future": object_visibility_values,
        "object_visibility_future_mask": object_visibility_masks,
        "next_best_view_score": next_best_view_values,
        "next_best_view_score_mask": next_best_view_masks,
        "per_camera_visibility": per_camera_scores,
        "main_cameras": list(main_camera_names),
        "alternative_cameras": list(alternative_camera_names),
        "visibility_horizon": visibility_horizon,
        "max_distance": max_distance,
        "schema_version": "mowa_visual_proxy_v2_occlusion",
    }


def _iter_forward_positions(
    model: Any,
    states: np.ndarray,
    ep_meta: dict[str, Any],
    cameras: list[CameraConfig],
    target_point_resolver: Callable[[Any, int], np.ndarray | None],
    target_body_id_resolver: Callable[[Any, int], int | None] | None = None,
):
    """Yield ``(timestep, target_point, target_body_id, camera_poses, data)`` for each state row.

    This uses MuJoCo's position-only forward kinematics (``mj_fwdPosition``),
    which is cheaper than ``mj_forward`` because we only need body/site/camera
    world positions for visibility checks.
    """
    import mujoco

    nq = model.nq
    nv = model.nv
    data = mujoco.MjData(model)
    row_count = int(states.shape[0])

    for t in range(row_count):
        data.time = float(states[t, 0])
        data.qpos[:] = states[t, 1 : 1 + nq]
        data.qvel[:] = states[t, 1 + nq : 1 + nq + nv]
        mujoco.mj_fwdPosition(model, data)

        target = target_point_resolver(data, t)
        target_body_id = None
        if target_body_id_resolver is not None:
            target_body_id = target_body_id_resolver(data, t)
        camera_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for camera in cameras:
            pose = resolve_camera_pose(data, camera, ep_meta, model)
            if pose is not None:
                camera_poses[camera.name] = pose
        yield t, target, target_body_id, camera_poses, data


def _iter_forward_kinematics(model: Any, states: np.ndarray):
    """Re-export of ``mujoco_state_utils.iter_forward_kinematics`` locally."""
    from starVLA.dataloader.mowa.mujoco_state_utils import iter_forward_kinematics

    yield from iter_forward_kinematics(model, states)
