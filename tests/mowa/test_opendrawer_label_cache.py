"""Tests for OpenDrawer future-label cache and dataloader integration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import _attach_mowa_future_labels
from starVLA.dataloader.mowa import (
    build_mowa_future_constructible_label_smoke,
    build_opendrawer_label_cache_for_episode,
    load_opendrawer_label_cache_for_episode,
    opendrawer_label_cache_available,
    write_opendrawer_label_cache,
)


class MoWAOpenDrawerLabelCacheTest(unittest.TestCase):
    def _write_minimal_robocasa_parquet_dataset(self, dataset_path: Path, length: int = 5):
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta_dir = dataset_path / "meta"
        data_dir = dataset_path / "data" / "chunk-000"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        episode_rows = [
            json.dumps(
                {
                    "episode_index": 0,
                    "tasks": ["Open the right drawer."],
                    "length": length,
                }
            )
        ]
        table = pa.table(
            {
                "annotation.human.task_description": pa.array([0] * length, type=pa.int64()),
                "annotation.human.task_name": pa.array([2] * length, type=pa.int64()),
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * 16)]),
                    16,
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * 12)]),
                    12,
                ),
                "next.reward": pa.array([0.0] * (length - 1) + [1.0], type=pa.float32()),
                "next.done": pa.array([False] * (length - 1) + [True]),
                "timestamp": pa.array([0.05 * idx for idx in range(length)], type=pa.float32()),
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([0] * length, type=pa.int64()),
                "index": pa.array(list(range(length)), type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
            }
        )
        pq.write_table(table, data_dir / "episode_000000.parquet")
        (meta_dir / "episodes.jsonl").write_text("\n".join(episode_rows) + "\n", encoding="utf-8")
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "Open the right drawer."}) + "\n",
            encoding="utf-8",
        )
        (meta_dir / "modality.json").write_text(
            json.dumps(
                {
                    "video": {
                        "robot0_agentview_left": {
                            "original_key": "observation.images.robot0_agentview_left"
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def _write_minimal_opendrawer_extras(self, dataset_path: Path, states: np.ndarray):
        extras_dir = dataset_path / "extras" / "episode_000000"
        extras_dir.mkdir(parents=True)
        (extras_dir / "ep_meta.json").write_text(
            json.dumps({"fixture_refs": {"drawer": "stack_1_left_group_4"}}),
            encoding="utf-8",
        )
        (extras_dir / "model.xml.gz").write_bytes(
            __import__("gzip").compress(
                (
                    "<mujoco><worldbody>"
                    "<body><joint name='robot0_joint1' type='hinge' range='-1 1' />"
                    "<joint name='stack_1_left_group_4_slidejoint' type='slide' range='-0.6 0' />"
                    "</body></worldbody></mujoco>"
                ).encode("utf-8")
            )
        )
        np.savez_compressed(extras_dir / "states.npz", states=states)

    def _write_opendrawer_extras_with_kinematics(
        self, dataset_path: Path, states: np.ndarray
    ):
        extras_dir = dataset_path / "extras" / "episode_000000"
        extras_dir.mkdir(parents=True)
        (extras_dir / "ep_meta.json").write_text(
            json.dumps({"fixture_refs": {"drawer": "stack_1_left_group_4"}}),
            encoding="utf-8",
        )
        # EEF site fixed at (0.1, 0, 0); handle site attached to drawer body which
        # slides along x.  Distance = |drawer_qpos - 0.1|.
        xml = (
            "<mujoco>"
            "<worldbody>"
            "<body name='gripper0_right' pos='0.1 0 0'>"
            "<site name='gripper0_right_grip_site' pos='0 0 0' size='0.01'/>"
            "</body>"
            "<body name='stack_1_left_group_4' pos='0 0 0'>"
            "<joint name='robot0_joint1' type='hinge' axis='0 0 1' range='-1 1'/>"
            "<joint name='stack_1_left_group_4_slidejoint' type='slide' axis='1 0 0' range='-0.6 0.2'/>"
            "<geom name='stack_1_left_group_4_door_main' type='box' size='0.01 0.01 0.01'/>"
            "<body name='stack_1_left_group_4_door_main'>"
            "<site name='stack_1_left_group_4_door_handle_default_site' pos='0 0 0' size='0.01'/>"
            "</body>"
            "</body>"
            "</worldbody>"
            "</mujoco>"
        )
        (extras_dir / "model.xml.gz").write_bytes(__import__("gzip").compress(xml.encode("utf-8")))
        np.savez_compressed(extras_dir / "states.npz", states=states)

    def test_build_label_cache_for_episode_produces_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            # Drawer qpos at column 2; range [-0.6, 0]; open_denominator=0.165.
            # progress = (-qpos) / 0.165.  qpos=-0.2 -> progress ~1.21 (>0.95).
            states = np.array(
                [
                    [0.0, 0.0, -0.00, 0.0, 0.0],
                    [0.0, 0.0, -0.05, 0.0, -0.05],
                    [0.0, 0.0, -0.10, 0.0, -0.05],
                    [0.0, 0.0, -0.20, 0.0, -0.10],
                    [0.0, 0.0, -0.20, 0.0, 0.00],
                ],
                dtype=float,
            )
            self._write_minimal_opendrawer_extras(dataset_path, states)

            cache = build_opendrawer_label_cache_for_episode(
                dataset_path / "data" / "chunk-000" / "episode_000000.parquet",
                dataset_path / "extras" / "episode_000000" / "states.npz",
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
            )

        self.assertEqual(cache["row_count"], 5)
        # Row 0: future [1,2] has progress 0.303, 0.606 -> no subgoal, but readiness delta row2=0.303>0.05 -> ready
        self.assertEqual(cache["subgoal_feasibility"][0], 0.0)
        self.assertEqual(cache["manipulation_readiness"][0], 1.0)
        # Row 1: future [2,3] has progress 0.606, 1.212 -> subgoal yes, readiness yes
        self.assertEqual(cache["subgoal_feasibility"][1], 1.0)
        self.assertEqual(cache["manipulation_readiness"][1], 1.0)
        # Last row is completed -> raw mask is True (masked).
        self.assertTrue(cache["subgoal_feasibility_mask"][-1])
        self.assertTrue(cache["manipulation_readiness_mask"][-1])

    def test_write_and_load_label_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            states = np.array(
                [
                    [0.0, 0.0, -0.00, 0.0, 0.0],
                    [0.0, 0.0, -0.05, 0.0, -0.05],
                    [0.0, 0.0, -0.10, 0.0, -0.05],
                    [0.0, 0.0, -0.20, 0.0, -0.10],
                    [0.0, 0.0, -0.20, 0.0, 0.00],
                ],
                dtype=float,
            )
            self._write_minimal_opendrawer_extras(dataset_path, states)

            manifest = write_opendrawer_label_cache(
                dataset_path,
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
            )
            self.assertEqual(manifest["processed_episode_count"], 1)
            self.assertTrue(opendrawer_label_cache_available(dataset_path))

            cache = load_opendrawer_label_cache_for_episode(dataset_path, 0)
            self.assertIsNotNone(cache)
            self.assertEqual(len(cache["frame_index"]), 5)
            self.assertIn("subgoal_feasibility", cache["labels"])
            self.assertIn("manipulation_readiness", cache["masks"])

    def test_label_builder_smoke_uses_cache_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            states = np.array(
                [
                    [0.0, 0.0, -0.00, 0.0, 0.0],
                    [0.0, 0.0, -0.05, 0.0, -0.05],
                    [0.0, 0.0, -0.10, 0.0, -0.05],
                    [0.0, 0.0, -0.20, 0.0, -0.10],
                    [0.0, 0.0, -0.20, 0.0, 0.00],
                ],
                dtype=float,
            )
            self._write_minimal_opendrawer_extras(dataset_path, states)
            write_opendrawer_label_cache(
                dataset_path,
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
            )

            report = build_mowa_future_constructible_label_smoke(
                dataset_path, episode_indices=(0,), preview_rows=3
            ).to_dict()

        first = report["samples"][0]
        self.assertIn("subgoal_feasibility", first["labels"])
        self.assertIn("manipulation_readiness", first["labels"])
        self.assertTrue(first["masks"]["subgoal_feasibility"])
        self.assertTrue(first["masks"]["manipulation_readiness"])

    def test_attach_mowa_future_labels_merges_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            states = np.array(
                [
                    [0.0, 0.0, -0.00, 0.0, 0.0],
                    [0.0, 0.0, -0.05, 0.0, -0.05],
                    [0.0, 0.0, -0.10, 0.0, -0.05],
                    [0.0, 0.0, -0.20, 0.0, -0.10],
                    [0.0, 0.0, -0.20, 0.0, 0.00],
                ],
                dtype=float,
            )
            self._write_minimal_opendrawer_extras(dataset_path, states)
            write_opendrawer_label_cache(
                dataset_path,
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
            )

            dataset = MagicMock()
            dataset.data_cfg = {"enable_mowa_future_labels": True}
            dataset.dataset_path = dataset_path
            dataset.curr_traj_data = None
            import pandas as pd
            dataset.get_trajectory_data.return_value = pd.read_parquet(
                dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
            )

            sample = {}
            sample = _attach_mowa_future_labels(sample, dataset, trajectory_id=0, base_index=1)

        self.assertIn("mowa_future_targets", sample)
        self.assertIn("subgoal_feasibility", sample["mowa_future_targets"])
        self.assertIn("manipulation_readiness", sample["mowa_future_targets"])
        self.assertTrue(sample["mowa_future_masks"]["subgoal_feasibility"])
        self.assertTrue(sample["mowa_future_masks"]["manipulation_readiness"])
        self.assertEqual(
            sample["mowa_future_metadata"]["label_status"],
            "merged_opendrawer_label_cache",
        )

    def test_failure_risk_stays_masked_when_single_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            states = np.array(
                [
                    [0.0, 0.0, -0.00, 0.0, 0.0],
                    [0.0, 0.0, -0.05, 0.0, -0.05],
                    [0.0, 0.0, -0.10, 0.0, -0.05],
                    [0.0, 0.0, -0.20, 0.0, -0.10],
                    [0.0, 0.0, -0.20, 0.0, 0.00],
                ],
                dtype=float,
            )
            self._write_minimal_opendrawer_extras(dataset_path, states)
            write_opendrawer_label_cache(
                dataset_path,
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
            )

            dataset = MagicMock()
            dataset.data_cfg = {"enable_mowa_future_labels": True}
            dataset.dataset_path = dataset_path
            dataset.curr_traj_data = None
            import pandas as pd
            dataset.get_trajectory_data.return_value = pd.read_parquet(
                dataset_path / "data" / "chunk-000" / "episode_000000.parquet"
            )

            sample = {}
            sample = _attach_mowa_future_labels(sample, dataset, trajectory_id=0, base_index=0)

        # All rewards are 0 except last step; short horizon means no done in window for row 0.
        self.assertFalse(sample["mowa_future_masks"]["failure_risk"])

    def test_kinematics_based_readiness_uses_distance_threshold(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not available in the current interpreter")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            self._write_minimal_robocasa_parquet_dataset(dataset_path, length=5)
            # Drawer qpos at column 2. EEF at x=0.1, handle at x=drawer_qpos.
            # distance = |qpos - 0.1|. Threshold 0.05 -> ready when qpos in [0.05, 0.15].
            states = np.array(
                [
                    [0.0, 0.0, 0.00, 0.0, 0.0],  # dist=0.10 -> not ready
                    [0.0, 0.0, 0.08, 0.0, 0.0],  # dist=0.02 -> ready
                    [0.0, 0.0, 0.12, 0.0, 0.0],  # dist=0.02 -> ready
                    [0.0, 0.0, 0.20, 0.0, 0.0],  # dist=0.10 -> not ready
                    [0.0, 0.0, 0.20, 0.0, 0.0],  # completed -> masked
                ],
                dtype=float,
            )
            self._write_opendrawer_extras_with_kinematics(dataset_path, states)

            cache = build_opendrawer_label_cache_for_episode(
                dataset_path / "data" / "chunk-000" / "episode_000000.parquet",
                dataset_path / "extras" / "episode_000000" / "states.npz",
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.05,
                readiness_distance_threshold=0.05,
                enable_kinematics=True,
            )

        self.assertTrue(cache["kinematics_available"])
        self.assertEqual(cache["readiness_schema_version"], "opendrawer_proximity_v1")
        self.assertEqual(cache["manipulation_readiness"][0], 0.0)
        self.assertEqual(cache["manipulation_readiness"][1], 1.0)
        self.assertEqual(cache["manipulation_readiness"][2], 1.0)
        self.assertEqual(cache["manipulation_readiness"][3], 0.0)
        # Last row is completed -> masked.
        self.assertTrue(cache["manipulation_readiness_mask"][-1])


if __name__ == "__main__":
    unittest.main()
