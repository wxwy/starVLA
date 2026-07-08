"""Tests for MoWA visual future-head utilities.

These tests focus on the core projection and visibility-score math, which is
pure NumPy, and on ``compute_visual_head_labels`` using mocked camera poses so
that MuJoCo is not required for the label-composition logic.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from starVLA.dataloader.mowa.visual_head_utils import (
    CameraConfig,
    compute_visibility_score,
    compute_visual_head_labels,
    project_camera_to_image,
    project_world_to_camera,
    quat_to_rotation_matrix,
)


class MoWAVisualHeadUtilsTest(unittest.TestCase):
    def test_quat_to_rotation_matrix_identity(self):
        quat = (1.0, 0.0, 0.0, 0.0)
        mat = quat_to_rotation_matrix(quat)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-10)

    def test_quat_to_rotation_matrix_90_degree_z(self):
        # 90-degree rotation around Z axis: w=cos(45), z=sin(45)
        q = np.sqrt(2.0) / 2.0
        mat = quat_to_rotation_matrix((q, 0.0, 0.0, q))
        expected = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_project_world_to_camera_looks_down_negative_z(self):
        # Camera at origin, looking down -Z.
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, -1.0])
        p_cam = project_world_to_camera(point, cam_pos, cam_mat)
        np.testing.assert_allclose(p_cam, point, atol=1e-10)

    def test_project_world_to_camera_translates(self):
        cam_pos = np.array([1.0, 2.0, 3.0])
        cam_mat = np.eye(3)
        point = np.array([2.0, 4.0, 1.0])
        p_cam = project_world_to_camera(point, cam_pos, cam_mat)
        np.testing.assert_allclose(p_cam, np.array([1.0, 2.0, -2.0]), atol=1e-10)

    def test_project_camera_to_image_center(self):
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=256.0 / (2.0 * np.tan(np.deg2rad(60.0) / 2.0)),
            principal_point=(128.0, 128.0),
        )
        # Point 1m in front of camera, centered.
        p_cam = np.array([0.0, 0.0, -1.0])
        u, v = project_camera_to_image(p_cam, camera)
        self.assertAlmostEqual(u, 128.0, places=6)
        self.assertAlmostEqual(v, 128.0, places=6)

    def test_project_camera_to_image_rejects_behind_camera(self):
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=256.0 / (2.0 * np.tan(np.deg2rad(60.0) / 2.0)),
            principal_point=(128.0, 128.0),
        )
        p_cam = np.array([0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            project_camera_to_image(p_cam, camera)

    def test_compute_visibility_score_centered_point(self):
        # MuJoCo/OpenGL cameras look down the negative Z axis; a visible point
        # must therefore have negative Z in the camera frame.
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=256.0 / (2.0 * np.tan(np.deg2rad(60.0) / 2.0)),
            principal_point=(128.0, 128.0),
        )
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, -1.0])
        score = compute_visibility_score(point, cam_pos, cam_mat, camera, max_distance=2.0)
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_compute_visibility_score_zero_when_behind_camera(self):
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=1.0,
            principal_point=(128.0, 128.0),
        )
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, 1.0])
        score = compute_visibility_score(point, cam_pos, cam_mat, camera, max_distance=2.0)
        self.assertEqual(score, 0.0)

    def test_compute_visibility_score_zero_when_outside_image(self):
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=1.0,
            principal_point=(128.0, 128.0),
        )
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        # Far to the right but still in front of camera.
        point = np.array([10.0, 0.0, 1.0])
        score = compute_visibility_score(point, cam_pos, cam_mat, camera, max_distance=20.0)
        self.assertEqual(score, 0.0)

    def test_compute_visibility_score_zero_when_beyond_max_distance(self):
        camera = CameraConfig(
            name="center",
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=1.0,
            principal_point=(128.0, 128.0),
        )
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, 3.0])
        score = compute_visibility_score(point, cam_pos, cam_mat, camera, max_distance=2.0)
        self.assertEqual(score, 0.0)

    def _make_mock_camera(self, name: str) -> CameraConfig:
        return CameraConfig(
            name=name,
            cam_id=0,
            width=256,
            height=256,
            fovy=60.0,
            focal_length=256.0 / (2.0 * np.tan(np.deg2rad(60.0) / 2.0)),
            principal_point=(128.0, 128.0),
        )

    def test_compute_visual_head_labels_binary_and_continuous(self):
        """Target is visible to main camera at t=0..2 and beyond max distance at t=3."""
        model = MagicMock()
        states = np.zeros((5, 10), dtype=np.float64)
        ep_meta: dict = {"cam_configs": {}}

        def target_resolver(data: object, timestep: int) -> np.ndarray:
            del data
            # Place target at z=-1 (in front of camera) for first three steps,
            # z=-3 (beyond max_distance) afterwards.
            z = -1.0 if timestep < 3 else -3.0
            return np.array([0.0, 0.0, z])

        progress = np.array([0.0, 0.1, 0.2, 0.5, 0.99], dtype=np.float64)

        camera = self._make_mock_camera("robot0_agentview_left")

        def fake_iter_positions(model, states, ep_meta, cameras, target_resolver, target_body_id_resolver):
            del target_body_id_resolver
            for t in range(5):
                yield t, target_resolver(None, t), None, {
                    camera.name: (np.array([0.0, 0.0, 0.0]), np.eye(3))
                    for camera in cameras
                }, None

        with patch(
            "starVLA.dataloader.mowa.visual_head_utils.build_camera_configs",
            return_value=[camera],
        ), patch(
            "starVLA.dataloader.mowa.visual_head_utils._iter_forward_positions",
            side_effect=fake_iter_positions,
        ):
            labels = compute_visual_head_labels(
                model=model,
                states=states,
                ep_meta=ep_meta,
                target_point_resolver=target_resolver,
                progress=progress,
                completion_threshold=0.95,
                visibility_horizon=2,
                main_camera_names=("robot0_agentview_left",),
                alternative_camera_names=("robot0_agentview_left",),
            )

        # At t=0, future t=1,2 both visible -> object_visibility_future=1.
        self.assertEqual(labels["object_visibility_future"][0], 1.0)
        # score = 1 - 1/2 = 0.5 for visible future steps.
        self.assertAlmostEqual(labels["next_best_view_score"][0], 0.5, places=6)

        # At t=3, future t=4 has target at z=-3 (beyond max_distance) and t=5 does not exist.
        self.assertEqual(labels["object_visibility_future"][3], 0.0)
        self.assertEqual(labels["next_best_view_score"][3], 0.0)

        # Last step is near completion (progress=0.99 >= 0.95) -> masked.
        self.assertTrue(labels["object_visibility_future_mask"][-1])
        self.assertTrue(labels["next_best_view_score_mask"][-1])

        # Earlier steps should be unmasked.
        self.assertFalse(labels["object_visibility_future_mask"][0])
        self.assertFalse(labels["next_best_view_score_mask"][0])

    def test_compute_visual_head_labels_uses_max_across_cameras(self):
        """next_best_view_score picks the best camera; object_visibility uses OR."""
        model = MagicMock()
        states = np.zeros((3, 10), dtype=np.float64)
        ep_meta: dict = {"cam_configs": {}}

        def target_resolver(data: object, timestep: int) -> np.ndarray:
            del data
            return np.array([0.0, 0.0, -1.0])

        progress = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        left_cam = self._make_mock_camera("robot0_agentview_left")
        right_cam = self._make_mock_camera("robot0_agentview_right")

        # left camera sees target at distance 1 -> score 0.5.
        # right camera is farther: distance 2 -> score 0.0 (at max_distance boundary).
        def resolve_pose(data: object, camera: CameraConfig, ep_meta: dict, model: object):
            del data, ep_meta, model
            if camera.name == "robot0_agentview_left":
                return (np.array([0.0, 0.0, 0.0]), np.eye(3))
            return (np.array([0.0, 0.0, -1.0]), np.eye(3))

        def fake_iter_positions(model, states, ep_meta, cameras, target_resolver, target_body_id_resolver):
            del target_body_id_resolver
            for t in range(3):
                poses = {}
                for camera in cameras:
                    poses[camera.name] = resolve_pose(None, camera, ep_meta, model)
                yield t, target_resolver(None, t), None, poses, None

        with patch(
            "starVLA.dataloader.mowa.visual_head_utils.build_camera_configs",
            return_value=[left_cam, right_cam],
        ), patch(
            "starVLA.dataloader.mowa.visual_head_utils._iter_forward_positions",
            side_effect=fake_iter_positions,
        ):
            labels = compute_visual_head_labels(
                model=model,
                states=states,
                ep_meta=ep_meta,
                target_point_resolver=target_resolver,
                progress=progress,
                completion_threshold=0.95,
                visibility_horizon=1,
                main_camera_names=("robot0_agentview_left",),
                alternative_camera_names=(
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                ),
            )

        # Object visibility: left sees it -> 1.
        self.assertEqual(labels["object_visibility_future"][0], 1.0)
        # Next best view: left gives 0.5, right gives 0.0 -> max is 0.5.
        self.assertAlmostEqual(labels["next_best_view_score"][0], 0.5, places=6)

    def test_compute_visual_head_labels_missing_target_masks(self):
        """If target resolver returns None, scores are zero but masks stay tied to progress."""
        model = MagicMock()
        states = np.zeros((3, 10), dtype=np.float64)
        ep_meta: dict = {"cam_configs": {}}
        progress = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        camera = self._make_mock_camera("robot0_agentview_left")

        def fake_iter_positions(model, states, ep_meta, cameras, target_resolver, target_body_id_resolver):
            del target_body_id_resolver
            for t in range(3):
                yield t, target_resolver(None, t), None, {
                    camera.name: (np.array([0.0, 0.0, 0.0]), np.eye(3))
                    for camera in cameras
                }, None

        with patch(
            "starVLA.dataloader.mowa.visual_head_utils.build_camera_configs",
            return_value=[camera],
        ), patch(
            "starVLA.dataloader.mowa.visual_head_utils._iter_forward_positions",
            side_effect=fake_iter_positions,
        ):
            labels = compute_visual_head_labels(
                model=model,
                states=states,
                ep_meta=ep_meta,
                target_point_resolver=lambda _data, _t: None,
                progress=progress,
                completion_threshold=0.95,
                visibility_horizon=2,
                main_camera_names=("robot0_agentview_left",),
                alternative_camera_names=("robot0_agentview_left",),
            )

        np.testing.assert_array_equal(labels["object_visibility_future"], np.zeros(3))
        np.testing.assert_array_equal(labels["next_best_view_score"], np.zeros(3))
        self.assertFalse(labels["object_visibility_future_mask"][0])
        self.assertFalse(labels["next_best_view_score_mask"][0])

    def test_compute_visibility_score_occlusion_blocks_when_ray_hits_early(self):
        """When mj_ray reports a hit before the target, the score is zeroed."""
        camera = self._make_mock_camera("robot0_agentview_left")
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, -1.0])

        model = MagicMock()
        data = MagicMock()

        def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, normal):
            del model, data, pnt, vec, geomgroup, flg_static, bodyexclude, normal
            # Hit at 0.3m along a 1m ray, on a different body -> occluded.
            geomid[0] = 0
            return 0.3

        model.geom_bodyid = np.array([999], dtype=np.int32)
        with patch("mujoco.mj_ray", side_effect=fake_ray):
            score = compute_visibility_score(
                point, cam_pos, cam_mat, camera, max_distance=2.0,
                model=model, data=data, target_body_id=1, check_occlusion=True,
            )
        self.assertEqual(score, 0.0)

    def test_compute_visibility_score_occlusion_allows_when_ray_reaches_target(self):
        """When mj_ray reports no hit, the target is considered unoccluded."""
        camera = self._make_mock_camera("robot0_agentview_left")
        cam_pos = np.array([0.0, 0.0, 0.0])
        cam_mat = np.eye(3)
        point = np.array([0.0, 0.0, -1.0])

        model = MagicMock()
        data = MagicMock()

        def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, normal):
            del model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, normal
            return -1.0

        with patch("mujoco.mj_ray", side_effect=fake_ray):
            score = compute_visibility_score(
                point, cam_pos, cam_mat, camera, max_distance=2.0,
                model=model, data=data, target_body_id=1, check_occlusion=True,
            )
        self.assertAlmostEqual(score, 0.5, places=6)

    def test_compute_visual_head_labels_occlusion_uses_target_body_resolver(self):
        """object_visibility_future respects occlusion on main cameras; next_best_view_score does not."""
        model = MagicMock()
        states = np.zeros((3, 10), dtype=np.float64)
        ep_meta: dict = {"cam_configs": {}}
        progress = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        left_cam = self._make_mock_camera("robot0_agentview_left")
        right_cam = self._make_mock_camera("robot0_agentview_right")

        def target_resolver(data: object, timestep: int) -> np.ndarray:
            del data, timestep
            return np.array([0.0, 0.0, -1.0])

        def target_body_id_resolver(data: object, timestep: int) -> int:
            del data, timestep
            return 42

        # Main camera sees the target but mj_ray reports an early hit on another body -> occluded.
        def fake_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid, normal):
            del model, data, pnt, vec, geomgroup, flg_static, bodyexclude, normal
            geomid[0] = 0
            return 0.2

        model.geom_bodyid = np.array([999], dtype=np.int32)

        def fake_iter_positions(model, states, ep_meta, cameras, target_point_resolver, target_body_id_resolver):
            fake_data = MagicMock()
            for t in range(3):
                yield t, target_point_resolver(None, t), target_body_id_resolver(None, t), {
                    camera.name: (np.array([0.0, 0.0, 0.0]), np.eye(3))
                    for camera in cameras
                }, fake_data

        with patch(
            "starVLA.dataloader.mowa.visual_head_utils.build_camera_configs",
            return_value=[left_cam, right_cam],
        ), patch(
            "starVLA.dataloader.mowa.visual_head_utils._iter_forward_positions",
            side_effect=fake_iter_positions,
        ), patch(
            "mujoco.mj_ray",
            side_effect=fake_ray,
        ):
            labels = compute_visual_head_labels(
                model=model,
                states=states,
                ep_meta=ep_meta,
                target_point_resolver=target_resolver,
                target_body_id_resolver=target_body_id_resolver,
                progress=progress,
                completion_threshold=0.95,
                visibility_horizon=1,
                main_camera_names=("robot0_agentview_left",),
                alternative_camera_names=("robot0_agentview_right",),
            )

        # object_visibility_future is occluded -> 0 for all unmasked anchors.
        np.testing.assert_array_equal(labels["object_visibility_future"], np.zeros(3))
        # next_best_view_score uses the alternative camera and ignores occlusion.
        # Horizon=1, so the last anchor has no future step and stays 0.
        np.testing.assert_allclose(labels["next_best_view_score"], np.array([0.5, 0.5, 0.0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
