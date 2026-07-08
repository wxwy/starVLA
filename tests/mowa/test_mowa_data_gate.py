import unittest
import tempfile
import json
import gzip
from pathlib import Path

from starVLA.dataloader.mowa import (
    DATA_GATE,
    MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS,
    MOWA_PRIMARY_CANDIDATE,
    MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH,
    MoWAEpisodeToWindowSampler,
    MoWAUnifiedEpisode,
    MoWAWindowConfig,
    MoWAWindowSample,
    build_mowa_atomic_core_batch_dataloader_smoke,
    build_mowa_atomic_core_leakage_gate_smoke,
    build_mowa_atomic_core_production_preflight_smoke,
    build_mowa_atomic_core_temporal_profile,
    build_mowa_failure_risk_data_gate_smoke,
    build_mowa_future_latent_cache_contract_smoke,
    build_mowa_future_latent_cache_manifest_smoke,
    build_mowa_g0_report_skeleton,
    build_mowa_latent_cache_contract_smoke,
    build_mowa_latent_cache_manifest_smoke,
    build_mowa_opendrawer_future_label_sidecar_smoke,
    build_mowa_future_constructible_label_smoke,
    build_mowa_future_label_source_audit,
    build_mowa_robocasa365_local_smoke_report,
    build_mowa_opendrawer_state_mapping_audit,
    build_mowa_opendrawer_subgoal_data_gate_smoke,
    build_mowa_shuffled_episode_pairs,
    fixed_size_list_shape,
    inspect_mowa_future_label_coverage,
    inspect_mowa_robocasa365_atomic_core_recipe,
    inspect_robocasa365_lerobot_dataset_smoke,
    inspect_robocasa365_lerobot_episode_schema,
    inspect_robocasa365_lerobot_profile_smoke,
    select_mowa_leakage_anchor_indices,
    select_mowa_smoke_anchor_index,
)
from starVLA.dataloader.mowa.tasks._object_pose import ObjectPoseTaskBuilder, ObjectPoseTaskSchema
from starVLA.dataloader.mowa.tasks._reward_based import RewardBasedTaskBuilder, RewardBasedTaskSchema
from starVLA.dataloader.mowa.tasks.close_fridge import CloseFridgeLabelBuilder
from starVLA.mowa_constants import MOWA_ACTION_OUTCOME_CLASS_MAPPING_STATUS


class MoWADataGateTest(unittest.TestCase):
    def test_fixed_size_list_shape_is_public_helper(self):
        try:
            import pyarrow as pa
        except ImportError:
            self.skipTest("pyarrow is not available")

        field = pa.field("action", pa.list_(pa.float32(), 12))

        self.assertEqual(fixed_size_list_shape(field), (12,))

    def _write_minimal_robocasa_parquet_dataset(self, dataset_path: Path, lengths=(6, 7)):
        import pyarrow as pa
        import pyarrow.parquet as pq

        meta_dir = dataset_path / "meta"
        data_dir = dataset_path / "data" / "chunk-000"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        episode_rows = []
        for episode_index, length in enumerate(lengths):
            episode_rows.append(
                json.dumps(
                    {
                        "episode_index": episode_index,
                        "tasks": ["Open the right drawer."],
                        "length": length,
                    }
                )
            )
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
                    "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                    "index": pa.array(list(range(length)), type=pa.int64()),
                    "task_index": pa.array([0] * length, type=pa.int64()),
                }
            )
            pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")

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

    def _episode(self) -> MoWAUnifiedEpisode:
        return MoWAUnifiedEpisode(
            episode_id="ep_001",
            dataset_source="robocasa365",
            split="train",
            instruction="open the drawer",
            timestamps=[0, 1, 2, 3, 4, 5],
            observations={"rgb_front": "Data Gate", "robot_state": "Data Gate"},
            actions={"canonical_action": "Data Gate"},
            wam_targets={"future_labels": "Data Gate", "future_wan_latent": "Data Gate"},
            metadata={"obs_fps": DATA_GATE, "action_hz": DATA_GATE},
        )

    def _write_minimal_opendrawer_extras(self, dataset_path: Path, states):
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
        import numpy as np

        np.savez_compressed(
            extras_dir / "states.npz",
            states=np.array(states, dtype=float),
        )

    def test_unified_episode_rejects_non_monotonic_timestamps(self):
        episode = MoWAUnifiedEpisode(
            episode_id="ep_bad",
            dataset_source="robocasa365",
            split="train",
            instruction="open the drawer",
            timestamps=[0, 2, 1],
        )

        with self.assertRaisesRegex(ValueError, "timestamps must be monotonic"):
            episode.validate()

    def test_episode_to_window_sampler_keeps_future_targets_out_of_inputs(self):
        sampler = MoWAEpisodeToWindowSampler(
            MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)
        )
        sample = sampler.sample(self._episode(), anchor_index=3)

        self.assertEqual(sample.history_indices, (1, 2, 3))
        self.assertEqual(sample.future_indices, (4, 5))
        self.assertEqual(sample.action_target_indices, (3, 4))
        self.assertIn("history_actions", sample.inputs)
        self.assertIn("action_chunk_target", sample.targets)
        self.assertNotIn("action_chunk_target", sample.inputs)
        sample.validate()

    def test_smoke_anchor_selection_uses_shared_first_full_history_policy(self):
        config = MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)

        self.assertEqual(select_mowa_smoke_anchor_index(6, config), 2)
        self.assertEqual(select_mowa_smoke_anchor_index(2, config), 0)
        self.assertEqual(select_mowa_leakage_anchor_indices(6, config), (2, 3))

    def test_window_sample_blocks_future_action_leakage(self):
        sample = MoWAWindowSample(
            episode_id="ep_001",
            dataset_source="robocasa365",
            anchor_index=1,
            history_indices=(0, 1),
            current_index=1,
            future_indices=(2,),
            action_target_indices=(1, 2),
            inputs={
                "future_action_label": "must not enter WAM",
                "mowa_future_latent": "must not enter WAM",
            },
            targets={},
        )

        with self.assertRaisesRegex(ValueError, "forbidden future keys"):
            sample.validate()

    def test_g0_report_skeleton_marks_robocasa365_as_primary_without_profile_values(self):
        report = build_mowa_g0_report_skeleton()
        payload = report.to_dict()

        self.assertEqual(payload["stage"], "G0")
        self.assertEqual(payload["experiment_budget"], "not_counted")
        self.assertEqual(payload["candidates"][0]["dataset"], MOWA_PRIMARY_CANDIDATE)
        self.assertEqual(payload["candidates"][0]["role"], "PrimaryCandidate")
        self.assertEqual(payload["candidates"][0]["temporal_profile_status"], DATA_GATE)
        self.assertEqual(payload["candidates"][0]["temporal_profile_status"], DATA_GATE)

    def test_robocasa365_local_smoke_reports_missing_data_without_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_mowa_robocasa365_local_smoke_report(Path(tmpdir))
        payload = report.to_dict()

        self.assertFalse(payload["local_checks"]["path_exists"])
        self.assertEqual(payload["local_checks"]["profile_status"], DATA_GATE)
        self.assertIn("missing local minimal dataset", payload["go_no_go"])
        self.assertEqual(payload["local_checks"]["profile_status"], DATA_GATE)

    def test_robocasa365_local_smoke_detects_minimal_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH).mkdir(parents=True)
            report = build_mowa_robocasa365_local_smoke_report(root)
        payload = report.to_dict()

        self.assertTrue(payload["local_checks"]["path_exists"])
        self.assertEqual(payload["candidates"][0]["download_status"], "available")
        self.assertEqual(payload["candidates"][0]["temporal_profile_status"], DATA_GATE)

    def test_robocasa365_local_smoke_reads_lerobot_meta_without_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            meta_dir = dataset_path / "meta"
            meta_dir.mkdir(parents=True)
            (dataset_path / "data" / "chunk-000").mkdir(parents=True)
            (dataset_path / "videos" / "chunk-000").mkdir(parents=True)
            (dataset_path / "extras" / "episode_000000").mkdir(parents=True)
            info = {
                "robot_type": "PandaOmron",
                "total_episodes": 1,
                "total_frames": 10,
                "fps": 20,
                "splits": {"train": "0:1"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float64", "shape": [16]},
                    "action": {"dtype": "float64", "shape": [12]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                    "observation.images.robot0_agentview_left": {"dtype": "video", "shape": [256, 256, 3]},
                },
            }
            modality = {
                "state": {"base_position": {"original_key": "observation.state"}},
                "action": {"base_motion": {"original_key": "action"}},
                "video": {"robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"}},
            }
            (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")
            (meta_dir / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
            (meta_dir / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "tasks": ["Open the drawer."], "length": 10}) + "\n",
                encoding="utf-8",
            )
            (meta_dir / "episodes_stats.jsonl").write_text("{}\n", encoding="utf-8")
            (meta_dir / "tasks.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_path / "data" / "chunk-000" / "episode_000000.parquet").write_text("", encoding="utf-8")
            (
                dataset_path
                / "videos"
                / "chunk-000"
                / "observation.images.robot0_agentview_left"
            ).mkdir(parents=True)
            (
                dataset_path
                / "videos"
                / "chunk-000"
                / "observation.images.robot0_agentview_left"
                / "episode_000000.mp4"
            ).write_text("", encoding="utf-8")

            report = build_mowa_robocasa365_local_smoke_report(root)
        payload = report.to_dict()

        self.assertTrue(payload["local_checks"]["schema_smoke_available"])
        self.assertEqual(payload["candidates"][0]["schema_status"], "schema_smoke_available")
        self.assertEqual(payload["local_checks"]["declared_fps"], 20)
        self.assertEqual(payload["local_checks"]["obs_fps_status"], DATA_GATE)
        self.assertEqual(payload["local_checks"]["action_hz_status"], DATA_GATE)
        self.assertEqual(payload["local_checks"]["parquet_file_count"], 1)
        self.assertEqual(payload["local_checks"]["video_file_count"], 1)
        self.assertIn("profile/leakage pending", payload["go_no_go"])
        self.assertEqual(payload["local_checks"]["profile_status"], DATA_GATE)

    def test_e003_history_sampling_consistency_smoke_matches_temporal_policy_from_temp_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa" / "g0_atomic_core_smoke").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "temporal_policy:\n"
                "  raw_action_hz: 20\n"
                "  production_wam_hz: 5\n"
                "  wam_stride: 4\n"
                "  history_steps: 10\n"
                "  future_steps: 5\n"
                "  action_chunk_steps: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_launch_draft.yaml").write_text(
                "training:\n"
                "  raw_action_hz: 20\n"
                "  wam_hz: 5\n"
                "  wam_stride: 4\n"
                "  history_steps: 10\n"
                "  future_steps: 5\n"
                "  action_chunk_steps: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_draft.yaml").write_text(
                "runtime_targets:\n"
                "  raw_action_hz: 20\n"
                "  production_wam_hz: 5\n"
                "  wam_stride: 4\n"
                "  history_steps: 10\n"
                "  future_steps: 5\n"
                "  action_chunk_steps: 10\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_readiness_smoke.json").write_text(
                json.dumps(
                    {
                        "observed": {
                            "raw_action_hz": 20,
                            "production_wam_hz": 5,
                            "wam_stride": 4,
                            "history_steps": 10,
                            "future_steps": 5,
                            "action_chunk_steps": 10,
                        }
                    }
                ),
                encoding="utf-8",
            )
            window_preflight_path = (
                root / "docs_zh" / "mowa" / "g0_atomic_core_smoke"
                / "mowa_g0_atomic_core_production_window_5hz_preflight_smoke.json"
            )
            window_preflight_path.write_text(
                json.dumps(
                    {
                        "window_config": {
                            "history_steps": 10,
                            "future_steps": 5,
                            "action_chunk_steps": 10,
                        },
                        "raw_action_hz": 20,
                        "production_wam_hz": 5,
                        "wam_stride": 4,
                    }
                ),
                encoding="utf-8",
            )
            temporal_profile_path = (
                root / "docs_zh" / "mowa" / "g0_atomic_core_smoke"
                / "mowa_g0_atomic_core_temporal_profile.json"
            )
            temporal_profile_path.write_text(
                json.dumps(
                    {
                        "obs_fps_status": DATA_GATE,
                        "action_hz_status": DATA_GATE,
                        "history_window_status": DATA_GATE,
                        "future_window_status": DATA_GATE,
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e003_history_sampling_consistency_smoke import (
                build_e003_history_sampling_consistency_smoke,
            )

            report = build_e003_history_sampling_consistency_smoke(root)

        self.assertEqual(
            report["go_no_go"],
            "TBD: history sampling consistency smoke passed; latent-cache builder remains gated",
        )
        self.assertTrue(report["checks"]["launch_draft_matches_runtime_policy"])
        self.assertTrue(report["checks"]["training_command_matches_runtime_policy_core"])
        self.assertTrue(report["checks"]["readiness_matches_runtime_policy"])
        self.assertTrue(report["checks"]["readiness_matches_production_window_preflight_core"])
        self.assertTrue(report["checks"]["launch_matches_production_window_preflight_core"])
        self.assertTrue(report["checks"]["readiness_matches_production_window_preflight_core"])
        self.assertTrue(report["checks"]["temporal_profile_reports_data_gate_status"])

    def test_robocasa365_adapter_maps_parquet_to_unified_episode_without_leakage(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6,))

            schema = inspect_robocasa365_lerobot_episode_schema(dataset_path, episode_index=0)
            sample = MoWAEpisodeToWindowSampler(
                MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)
            ).sample(schema.unified_episode, anchor_index=3)

        self.assertTrue(schema.schema_available)
        self.assertEqual(schema.row_count, 6)
        self.assertEqual(schema.instruction, "Open the right drawer.")
        self.assertEqual(schema.action_shape, (12,))
        self.assertEqual(schema.state_shape, (16,))
        self.assertEqual(schema.unified_episode.metadata["obs_fps"], DATA_GATE)
        self.assertEqual(schema.unified_episode.metadata["action_hz"], DATA_GATE)
        self.assertEqual(sample.history_indices, (1, 2, 3))
        self.assertEqual(sample.future_indices, (4, 5))
        self.assertEqual(sample.action_target_indices, (3, 4))
        self.assertNotIn("action_chunk_target", sample.inputs)
        sample.validate()

    def test_robocasa365_dataset_smoke_checks_boundaries_and_future_coverage(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7))
            smoke = inspect_robocasa365_lerobot_dataset_smoke(
                dataset_path,
                episode_indices=(0, 1),
                window_config=MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2),
            )
        payload = smoke.to_dict()

        self.assertEqual(payload["episode_count"], 2)
        self.assertEqual(payload["sampled_episode_indices"], (0, 1))
        self.assertEqual(payload["sampled_row_counts"], (6, 7))
        self.assertEqual(payload["boundary_smoke"]["cross_episode_leakage_status"], "smoke_passed")
        self.assertEqual(payload["boundary_smoke"]["future_action_leakage_status"], "smoke_passed")
        self.assertIn("task_progress", payload["future_label_coverage"])
        self.assertIn("action_outcome_class", payload["future_label_coverage"])
        self.assertEqual(payload["boundary_smoke"]["window_config"]["status"], "smoke_only_target")

    def test_atomic_core_recipe_reports_missing_and_available_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_task, first_relative_path = next(
                iter(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.items())
            )
            task_path = root / first_relative_path
            (task_path / "meta").mkdir(parents=True)
            (task_path / "data").mkdir()
            (task_path / "videos").mkdir()
            (task_path / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            partial = inspect_mowa_robocasa365_atomic_core_recipe(root).to_dict()

            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                task_path = root / relative_path
                (task_path / "meta").mkdir(parents=True, exist_ok=True)
                (task_path / "data").mkdir(exist_ok=True)
                (task_path / "videos").mkdir(exist_ok=True)
                (task_path / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

            full = inspect_mowa_robocasa365_atomic_core_recipe(root).to_dict()

        self.assertEqual(partial["available_task_count"], 1)
        self.assertEqual(partial["missing_task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS) - 1)
        self.assertIn(first_task, str(partial))
        self.assertIn("incomplete", partial["go_no_go"])
        self.assertEqual(full["available_task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(full["missing_task_count"], 0)
        self.assertIn("profile/leakage/labels still Data Gate", full["go_no_go"])
        self.assertIn("Data Gate", " ".join(full["notes"]))

    def test_robocasa365_profile_smoke_keeps_profile_fields_data_gate(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7))
            profile = inspect_robocasa365_lerobot_profile_smoke(
                dataset_path,
                episode_indices=(0, 1),
                preview_rows=6,
            ).to_dict()

        self.assertEqual(profile["sampled_episode_indices"], (0, 1))
        self.assertEqual(profile["sampled_row_counts"], (6, 7))
        self.assertTrue(profile["timestamp_monotonic"])
        self.assertTrue(profile["frame_index_monotonic"])
        self.assertEqual(profile["timestamp_delta_preview"], (0.05,))
        self.assertTrue(profile["action_shape_consistent"])
        self.assertTrue(profile["state_shape_consistent"])
        self.assertTrue(profile["reward_signal_seen"])
        self.assertTrue(profile["next_done_seen"])
        self.assertEqual(profile["obs_fps_status"], DATA_GATE)
        self.assertEqual(profile["action_hz_status"], DATA_GATE)
        self.assertEqual(profile["history_window_status"], DATA_GATE)
        self.assertEqual(profile["future_window_status"], DATA_GATE)

    def test_future_label_coverage_reports_constructible_and_masked_heads(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7))
            report = inspect_mowa_future_label_coverage(dataset_path, episode_indices=(0, 1)).to_dict()

        self.assertEqual(report["sampled_episode_indices"], (0, 1))
        self.assertIn("frame_index", report["available_columns"])
        self.assertIn("action", report["available_columns"])
        self.assertIn("task_progress", report["constructible_heads"])
        self.assertIn("action_outcome_class", report["constructible_heads"])
        self.assertIn("next_best_view_score", report["constructible_heads"])
        self.assertIn("object_visibility_future", report["constructible_heads"])
        self.assertIn("failure_risk", report["masked_heads"])
        self.assertIn("manipulation_readiness", report["masked_heads"])
        self.assertIn("subgoal_feasibility", report["masked_heads"])
        by_head = {item["head"]: item for item in report["head_coverage"]}
        self.assertEqual(by_head["manipulation_readiness"]["status"], DATA_GATE)
        self.assertEqual(by_head["subgoal_feasibility"]["status"], DATA_GATE)
        self.assertTrue(by_head["next_best_view_score"]["status"].startswith("candidate"))
        self.assertTrue(by_head["object_visibility_future"]["status"].startswith("candidate"))

    def test_future_constructible_label_builder_keeps_unvalidated_heads_masked(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6,))
            report = build_mowa_future_constructible_label_smoke(
                dataset_path,
                episode_indices=(0,),
                preview_rows=3,
            ).to_dict()

        self.assertEqual(report["constructible_heads"], ("task_progress", "action_outcome_class"))
        self.assertEqual(report["sample_count"], 3)
        sample = report["samples"][0]
        self.assertIn("task_progress", sample["labels"])
        self.assertIn("action_outcome_class", sample["labels"])
        self.assertTrue(sample["masks"]["task_progress"])
        self.assertTrue(sample["masks"]["action_outcome_class"])
        self.assertFalse(sample["masks"]["manipulation_readiness"])
        self.assertFalse(sample["masks"]["object_visibility_future"])
        self.assertEqual(
            sample["labels"]["action_outcome_class"]["class_mapping_status"],
            MOWA_ACTION_OUTCOME_CLASS_MAPPING_STATUS,
        )

    def test_future_label_source_audit_reports_proxy_blockers(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(12,))
            report = build_mowa_future_label_source_audit(
                root,
                repo_root=Path("."),
                episode_indices=(0,),
            ).to_dict()

        self.assertEqual(report["available_task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["summary"]["failure_risk_candidate_task_count"], len(report["tasks"]))
        first = report["tasks"][0]
        self.assertEqual(first["label_readiness"]["failure_risk"], "candidate_for_data_gate")
        self.assertTrue(first["reward_done_audit"]["reward_is_sparse_0_1_on_sample"])
        self.assertGreater(first["reward_done_audit"]["failure_risk_h10_labeled_count"], 0)
        self.assertTrue(first["label_readiness"]["subgoal_feasibility"].startswith("blocked"))
        self.assertTrue(first["label_readiness"]["manipulation_readiness"].startswith("blocked"))

    def test_failure_risk_data_gate_smoke_reports_single_class_blocker(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(12,))
            report = build_mowa_failure_risk_data_gate_smoke(
                root,
                horizon=10,
                max_episodes_per_task=1,
            ).to_dict()

        self.assertEqual(report["available_task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["summary"]["sparse_reward_task_count"], len(report["tasks"]))
        self.assertEqual(report["summary"]["candidate_for_mask_lift_review_task_count"], 0)
        first = report["tasks"][0]
        self.assertGreater(first["labeled_count"], 0)
        self.assertEqual(first["positive_count"], 0)
        self.assertGreater(first["negative_count"], 0)
        self.assertEqual(first["gate_status"], "blocked_by_single_class_distribution")
        self.assertIsNone(first["next_done_phi_correlation"])
        self.assertTrue(report["go_no_go"].startswith("No-Go: failure_risk proxy remains gated"))

    def test_opendrawer_state_mapping_audit_recovers_drawer_joint_index(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(12,))
            self._write_minimal_opendrawer_extras(
                dataset_path,
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.05, 0.1, -0.35, 0.0, -0.1],
                ],
            )
            report = build_mowa_opendrawer_state_mapping_audit(
                dataset_path,
                episode_indices=(0,),
            ).to_dict()

        self.assertEqual(report["fixture_ref_value"], "stack_1_left_group_4")
        self.assertEqual(report["drawer_joint_name"], "stack_1_left_group_4_slidejoint")
        self.assertEqual(report["state_vector_width"], 5)
        self.assertEqual(report["inferred_nq"], 2)
        self.assertEqual(report["inferred_nv"], 2)
        self.assertEqual(report["drawer_qpos_state_index"], 2)
        self.assertEqual(report["drawer_qvel_state_index"], 4)
        self.assertTrue(report["summary"]["state_layout_matches_time_plus_qpos_plus_qvel"])
        self.assertTrue(report["summary"]["subgoal_open_drawer_predicate_ready"])
        sample = report["sampled_episodes"][0]
        self.assertAlmostEqual(sample["drawer_qpos_last"], -0.35)
        self.assertGreater(sample["drawer_normalized_open_max"], 0.95)

    def test_opendrawer_subgoal_data_gate_smoke_reports_usable_distribution(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(4,))
            self._write_minimal_opendrawer_extras(
                dataset_path,
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.05, 0.1, -0.1, 0.0, -0.1],
                    [0.10, 0.1, -0.34, 0.0, -0.1],
                    [0.15, 0.1, -0.34, 0.0, 0.0],
                ],
            )
            report = build_mowa_opendrawer_subgoal_data_gate_smoke(
                dataset_path,
                horizon=2,
                max_episodes=1,
            ).to_dict()

        self.assertEqual(report["summary"]["state_success_episode_count"], 1)
        self.assertEqual(report["summary"]["terminal_success_episode_count"], 1)
        self.assertEqual(report["summary"]["aligned_success_episode_count"], 1)
        self.assertEqual(report["summary"]["mismatched_episode_count"], 0)
        self.assertGreater(report["summary"]["anchor_positive_count"], 0)
        self.assertGreater(report["summary"]["anchor_negative_count"], 0)
        self.assertTrue(report["go_no_go"].startswith("TBD: OpenDrawer open_drawer subgoal labels look usable"))
        episode = report["sampled_episodes"][0]
        self.assertEqual(episode["alignment_status"], "aligned_success")
        self.assertGreaterEqual(episode["drawer_progress_max"], 0.95)

    def test_opendrawer_future_label_sidecar_smoke_builds_three_head_audit_labels(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(4,))
            self._write_minimal_opendrawer_extras(
                dataset_path,
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.05, 0.1, -0.1, 0.0, -0.1],
                    [0.10, 0.1, -0.22, 0.0, -0.1],
                    [0.15, 0.1, -0.34, 0.0, 0.0],
                ],
            )
            report = build_mowa_opendrawer_future_label_sidecar_smoke(
                dataset_path,
                failure_risk_horizon=2,
                subgoal_horizon=2,
                readiness_horizon=2,
                readiness_progress_delta=0.10,
                max_episodes=1,
            ).to_dict()

        self.assertEqual(report["episode_count"], 1)
        self.assertGreater(report["summary"]["step_count"], 0)
        self.assertGreater(report["summary"]["failure_risk_labeled_count"], 0)
        self.assertGreater(report["summary"]["subgoal_positive_count"], 0)
        self.assertGreater(report["summary"]["manipulation_readiness_positive_count"], 0)
        self.assertTrue(report["go_no_go"].startswith("TBD: OpenDrawer sidecar smoke built"))
        self.assertEqual(report["schema_contract"]["subgoal"]["schema_version"], "opendrawer_open_drawer_v1")
        self.assertEqual(report["schema_contract"]["subgoal"]["completion_threshold"], 0.95)
        self.assertEqual(report["schema_contract"]["readiness"]["schema_version"], "opendrawer_open_proxy_v1")
        self.assertEqual(report["schema_contract"]["readiness"]["threshold_version"], "readiness_progress_delta_v1")
        episode = report["sampled_episodes"][0]
        self.assertEqual(episode["alignment_status"], "aligned_success")
        self.assertEqual(episode["schema_contract"], report["schema_contract"])
        first_step = episode["steps"][0]
        self.assertIn("failure_risk", first_step["labels"])
        self.assertIn("subgoal_feasibility", first_step["labels"])
        self.assertIn("manipulation_readiness", first_step["labels"])
        self.assertEqual(first_step["labels"]["subgoal_schema_version"], "opendrawer_open_drawer_v1")
        self.assertEqual(first_step["labels"]["readiness_schema_version"], "opendrawer_open_proxy_v1")
        self.assertEqual(first_step["labels"]["subgoal_completion_threshold"], 0.95)
        self.assertEqual(first_step["labels"]["threshold_version"], "readiness_progress_delta_v1")
        self.assertTrue(first_step["masks"]["subgoal_feasibility"])
        self.assertTrue(first_step["masks"]["manipulation_readiness"])
        self.assertNotIn("future_action", " ".join(first_step["debug"].keys()))
        self.assertEqual(first_step["debug"]["subgoal_schema_version"], "opendrawer_open_drawer_v1")
        self.assertEqual(first_step["debug"]["readiness_schema_version"], "opendrawer_open_proxy_v1")
        self.assertEqual(first_step["debug"]["readiness_progress_delta"], 0.10)
        self.assertEqual(first_step["debug"]["profile_status"], "measured")

    def test_atomic_core_batch_dataloader_smoke_combines_windows_and_labels(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6,))
            report = build_mowa_atomic_core_batch_dataloader_smoke(
                root,
                episode_indices=(0,),
                window_config=MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2),
            ).to_dict()

        self.assertEqual(report["task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["sample_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["future_action_leakage_status"], "smoke_passed")
        self.assertEqual(report["constructible_label_status"], "smoke_passed")
        first = report["samples"][0]
        self.assertIn("history_actions", first["input_keys"])
        self.assertIn("action_chunk_target", first["target_keys"])
        self.assertFalse(first["future_action_in_inputs"])
        self.assertEqual(first["future_label_keys"], ("action_outcome_class", "task_progress"))

    def test_atomic_core_leakage_gate_scans_episode_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7))
            report = build_mowa_atomic_core_leakage_gate_smoke(
                root,
                window_config=MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2),
            ).to_dict()

        self.assertEqual(report["task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["episode_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS) * 2)
        self.assertGreater(report["checked_window_count"], 0)
        self.assertEqual(report["failed_window_count"], 0)
        self.assertEqual(report["future_action_leakage_status"], "smoke_passed")
        self.assertEqual(report["cross_episode_leakage_status"], "smoke_passed")

    def test_atomic_core_temporal_profile_keeps_window_data_gate(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7))
            report = build_mowa_atomic_core_temporal_profile(root).to_dict()

        self.assertEqual(report["task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertEqual(report["episode_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS) * 2)
        self.assertEqual(report["metadata_total_frames"], report["parquet_total_rows"])
        self.assertTrue(report["timestamp_monotonic"])
        self.assertTrue(report["frame_index_monotonic"])
        self.assertEqual(report["timestamp_delta_values"], (0.05,))
        self.assertEqual(report["state_shapes"], ((16,),))
        self.assertEqual(report["action_shapes"], ((12,),))
        self.assertEqual(report["history_window_status"], DATA_GATE)
        self.assertEqual(report["future_window_status"], DATA_GATE)

    def test_atomic_core_production_preflight_checks_split_workers_and_ranks(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for relative_path in MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS.values():
                dataset_path = root / relative_path
                self._write_minimal_robocasa_parquet_dataset(dataset_path, lengths=(6, 7, 8))
            report = build_mowa_atomic_core_production_preflight_smoke(
                root,
                val_every=2,
                worker_count=1,
                rank_count=2,
                max_worker_samples=4,
                window_config=MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2),
            ).to_dict()

        self.assertEqual(report["task_count"], len(MOWA_ROBOCASA365_TARGET_HUMAN_ATOMIC_CORE_TASK_PATHS))
        self.assertGreater(report["train_episode_count"], 0)
        self.assertGreater(report["val_episode_count"], 0)
        self.assertEqual(report["split_overlap_count"], 0)
        self.assertEqual(report["distributed_overlap_count"], 0)
        self.assertEqual(report["worker_sample_count"], 4)
        self.assertEqual(report["failed_sample_count"], 0)
        self.assertEqual(report["split_status"], "smoke_passed")
        self.assertEqual(report["worker_status"], "smoke_passed")
        self.assertEqual(report["distributed_sampler_status"], "smoke_passed")
        self.assertEqual(report["future_action_leakage_status"], "smoke_passed")
        self.assertIn("explicit E-001", report["go_no_go"])

    def test_latent_cache_manifest_smoke_checks_video_paths_without_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            video_dir = (
                dataset_path
                / "videos"
                / "chunk-000"
                / "observation.images.robot0_agentview_left"
            )
            video_dir.mkdir(parents=True)
            (video_dir / "episode_000000.mp4").write_text("", encoding="utf-8")

            report = build_mowa_latent_cache_manifest_smoke(
                dataset_path,
                episode_indices=(0,),
                video_keys=("observation.images.robot0_agentview_left",),
            ).to_dict()

        self.assertEqual(report["sampled_episode_indices"], (0,))
        self.assertEqual(report["missing_video_count"], 0)
        self.assertEqual(report["latent_shape_status"], DATA_GATE)
        self.assertEqual(report["cache_hash_status"], DATA_GATE)
        self.assertEqual(report["encoder_status"], DATA_GATE)
        self.assertTrue(report["entries"][0]["video_exists"])
        self.assertEqual(len(report["entries"][0]["cache_key"]), 16)
        self.assertTrue(report["entries"][0]["cache_relative_path"].endswith(".pt"))

    def test_latent_cache_contract_smoke_plans_artifacts_without_future_action_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH
            video_dir = (
                dataset_path
                / "videos"
                / "chunk-000"
                / "observation.images.robot0_agentview_left"
            )
            video_dir.mkdir(parents=True)
            (video_dir / "episode_000000.mp4").write_text("", encoding="utf-8")

            report = build_mowa_latent_cache_contract_smoke(
                dataset_path,
                cache_root=root / "cache",
                episode_indices=(0,),
                video_keys=("observation.images.robot0_agentview_left",),
            ).to_dict()

        self.assertEqual(report["missing_video_count"], 0)
        self.assertEqual(report["missing_cache_count"], 1)
        self.assertEqual(report["duplicate_cache_key_count"], 0)
        self.assertEqual(report["latent_shape_status"], DATA_GATE)
        self.assertEqual(report["cache_artifact_status"], DATA_GATE)
        self.assertEqual(report["encoder_status"], DATA_GATE)
        self.assertEqual(report["future_action_input_status"], "not_used_as_input")
        self.assertTrue(report["entries"][0]["cache_path"].endswith(".pt"))
        self.assertFalse(report["entries"][0]["cache_exists"])
        self.assertEqual(report["entries"][0]["cache_status"], "planned")

    def test_future_latent_cache_aliases_preserve_contract_smoke(self):
        self.assertIs(
            build_mowa_future_latent_cache_manifest_smoke,
            build_mowa_latent_cache_manifest_smoke,
        )
        self.assertIs(
            build_mowa_future_latent_cache_contract_smoke,
            build_mowa_latent_cache_contract_smoke,
        )

    def test_reward_based_builder_uses_schema_progress_delta_when_override_missing(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            data_dir = dataset_path / "data" / "chunk-000"
            data_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "next.reward": pa.array([0.0, 0.0, 1.0], type=pa.float32()),
                    "next.done": pa.array([False, False, True]),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                }
            )
            pq.write_table(table, data_dir / "episode_000000.parquet")

            builder = RewardBasedTaskBuilder(
                RewardBasedTaskSchema(
                    task_name="NavigateKitchen",
                    completion_threshold=0.95,
                    schema_version="reward_based_schema_fallback_test_v1",
                )
            )
            schema_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                root / "states.npz",
                readiness_horizon=2,
                readiness_progress_delta=None,
                readiness_distance_threshold=None,
            )
            override_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                root / "states.npz",
                readiness_horizon=2,
                readiness_progress_delta=1.1,
                readiness_distance_threshold=None,
            )

        self.assertEqual(schema_cache["manipulation_readiness"][0], 1.0)
        self.assertEqual(override_cache["manipulation_readiness"][0], 0.0)
        self.assertEqual(schema_cache["readiness_schema_version"], "navigatekitchen_reward_imminence_v1")

    def test_object_pose_builder_uses_schema_progress_delta_when_override_missing(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            data_dir = dataset_path / "data" / "chunk-000"
            extras_dir = dataset_path / "extras" / "episode_000000"
            data_dir.mkdir(parents=True)
            extras_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "next.reward": pa.array([0.0, 0.0, 1.0], type=pa.float32()),
                    "next.done": pa.array([False, False, True]),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                }
            )
            pq.write_table(table, data_dir / "episode_000000.parquet")
            xml = (
                "<mujoco><worldbody><body name='object_body'>"
                "<joint name='object_freejoint' type='free' />"
                "</body></worldbody></mujoco>"
            )
            (extras_dir / "model.xml.gz").write_bytes(gzip.compress(xml.encode("utf-8")))
            (extras_dir / "ep_meta.json").write_text("{}", encoding="utf-8")
            import numpy as np

            states = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            np.savez_compressed(extras_dir / "states.npz", states=states)

            builder = ObjectPoseTaskBuilder(
                ObjectPoseTaskSchema(
                    task_name="PickPlaceTest",
                    object_joint_selector=lambda xml_root, states: "object_freejoint",
                    completion_threshold=0.95,
                    schema_version="object_pose_schema_fallback_test_v1",
                )
            )
            schema_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                extras_dir / "states.npz",
                readiness_horizon=2,
                readiness_progress_delta=None,
                readiness_distance_threshold=None,
            )
            override_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                extras_dir / "states.npz",
                readiness_horizon=2,
                readiness_progress_delta=1.1,
                readiness_distance_threshold=None,
            )

        self.assertEqual(schema_cache["manipulation_readiness"][0], 1.0)
        self.assertEqual(override_cache["manipulation_readiness"][0], 0.0)
        self.assertEqual(schema_cache["readiness_schema_version"], "pickplacetest_object_displacement_imminence_v1")

    def test_close_fridge_builder_uses_class_defaults_when_override_missing(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            data_dir = dataset_path / "data" / "chunk-000"
            extras_dir = dataset_path / "extras" / "episode_000000"
            data_dir.mkdir(parents=True)
            extras_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "next.reward": pa.array([0.0, 0.0, 1.0], type=pa.float32()),
                    "next.done": pa.array([False, False, True]),
                    "frame_index": pa.array([0, 1, 2], type=pa.int64()),
                }
            )
            pq.write_table(table, data_dir / "episode_000000.parquet")
            xml = (
                "<mujoco><worldbody>"
                "<body>"
                "<inertial mass='1.0' diaginertia='0.1 0.1 0.1' pos='0 0 0' />"
                "<joint name='fridge_left_door_joint' type='hinge' range='0 1' />"
                "</body>"
                "</worldbody></mujoco>"
            )
            (extras_dir / "model.xml.gz").write_bytes(gzip.compress(xml.encode("utf-8")))
            (extras_dir / "ep_meta.json").write_text("{}", encoding="utf-8")
            import numpy as np

            states = np.array(
                [
                    [0.0, 0.90, 0.0],
                    [0.0, 0.70, 0.0],
                    [0.0, 0.70, 0.0],
                ],
                dtype=float,
            )
            np.savez_compressed(extras_dir / "states.npz", states=states)

            builder = CloseFridgeLabelBuilder()
            schema_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                extras_dir / "states.npz",
                readiness_horizon=None,
                readiness_progress_delta=None,
                readiness_distance_threshold=None,
            )
            override_cache = builder.build_cache_for_episode(
                data_dir / "episode_000000.parquet",
                extras_dir / "states.npz",
                readiness_horizon=2,
                readiness_progress_delta=0.5,
                readiness_distance_threshold=None,
            )

        self.assertEqual(schema_cache["readiness_schema_version"], "close_fridge_progress_imminence_v1")
        self.assertEqual(schema_cache["manipulation_readiness"][0], 1.0)
        self.assertEqual(override_cache["manipulation_readiness"][0], 0.0)

    def test_latent_cache_builder_design_smoke_passes(self):
        from tools.mowa.latent_cache_builder_design_smoke import (
            build_latent_cache_builder_design_smoke,
        )

        report = build_latent_cache_builder_design_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: latent cache builder design smoke passed; real builder remains gated",
        )

    def test_e003_history_sampling_consistency_smoke_matches_temporal_policy(self):
        from tools.mowa.e003_history_sampling_consistency_smoke import (
            build_e003_history_sampling_consistency_smoke,
        )

        report = build_e003_history_sampling_consistency_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: history sampling consistency smoke passed; latent-cache builder remains gated",
        )

    def test_shuffled_episode_pairs_are_non_self_and_cyclic(self):
        pairs = build_mowa_shuffled_episode_pairs((0, 1, 4))

        self.assertEqual(pairs, ((0, 1), (1, 4), (4, 0)))

    def test_shuffled_robot_sanity_plan_smoke_passes(self):
        from tools.mowa.shuffled_robot_sanity_plan_smoke import (
            build_shuffled_robot_sanity_plan_smoke,
        )

        report = build_shuffled_robot_sanity_plan_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: shuffled-robot sanity plan smoke passed; rollout remains gated",
        )

    def test_e005_shuffled_robot_checkpoint_preflight_detects_missing_checkpoint(self):
        from tools.mowa.e005_shuffled_robot_checkpoint_preflight_smoke import (
            build_e005_shuffled_robot_checkpoint_preflight_smoke,
        )

        report = build_e005_shuffled_robot_checkpoint_preflight_smoke(Path("."))

        self.assertEqual(report["experiment_id"], "E-005")
        self.assertFalse(report["launch_ready"])
        self.assertFalse(report["checks"]["checkpoint_preflight_passed"])
        self.assertEqual(
            report["go_no_go"],
            "No-Go: E-005 shuffled-robot checkpoint preflight incomplete",
        )

    def test_final_report_template_smoke_passes(self):
        from tools.mowa.final_report_template_smoke import (
            build_final_report_template_smoke,
        )

        report = build_final_report_template_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: final report template smoke passed",
        )

    def test_frozen_decoder_diagnostic_plan_smoke_passes(self):
        from tools.mowa.frozen_decoder_diagnostic_smoke import _build_smoke

        report = _build_smoke()

        self.assertFalse(report["eval_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: frozen decoder diagnostic plan smoke passed; real decoder runs remain gated",
        )


if __name__ == "__main__":
    unittest.main()
