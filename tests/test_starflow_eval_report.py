"""StarFlow-VLA eval report helper 测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from examples.LIBERO.eval_files.starflow_eval_report import (
    build_eval_report,
    infer_failure_category,
    load_eval_metadata,
    write_eval_report,
)


class StarFlowEvalReportTest(unittest.TestCase):
    def test_load_eval_metadata_from_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints" / "steps_1"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "pytorch_model.bin").write_bytes(b"weights")
            (checkpoint_dir / "optimizer.pt").write_bytes(b"optimizer")
            (checkpoint_dir / "starflow_mapping.json").write_text(
                json.dumps({"framework_name": "StarFlowVLA"}),
                encoding="utf-8",
            )
            (checkpoint_dir / "config.yaml").write_text(
                "version_id: '0.21'\n"
                "datasets:\n"
                "  vla_data:\n"
                "    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA\n"
                "    data_mix: libero_goal\n",
                encoding="utf-8",
            )
            (checkpoint_dir / "dataset_statistics.json").write_text('{"franka": {"action": {}}}\n', encoding="utf-8")

            metadata = load_eval_metadata(checkpoint_dir)

        self.assertEqual(metadata["data_version"]["data_mix"], "libero_goal")
        self.assertEqual(metadata["data_version"]["config_schema"], "0.21")
        self.assertEqual(metadata["starflow_mapping"]["framework_name"], "StarFlowVLA")
        self.assertIsNotNone(metadata["checkpoint_hash"])
        self.assertIsNotNone(metadata["config_hash"])

    def test_build_and_write_eval_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_eval_report(
                args={"task_suite_name": "libero_goal", "num_trials_per_task": 1, "max_tasks": 1},
                metadata={
                    "checkpoint_path": "playground/Checkpoints/run/checkpoints/steps_1",
                    "checkpoint_hash": "ckpt-hash",
                    "config_path": "playground/Checkpoints/run/checkpoints/steps_1/config.yaml",
                    "config_hash": "cfg-hash",
                    "data_version": {"data_mix": "libero_goal", "dataset_statistics_hash": "data-hash"},
                    "starflow_mapping": {"framework_name": "StarFlowVLA"},
                },
                total_episodes=2,
                total_successes=1,
                episode_records=[
                    {
                        "task_id": 0,
                        "task_description": "open drawer",
                        "episode_idx": 0,
                        "success": True,
                        "failure_category": infer_failure_category(success=True, runtime_error=None),
                        "runtime_error": None,
                        "steps_executed": 10,
                        "video_path": None,
                    },
                    {
                        "task_id": 0,
                        "task_description": "open drawer",
                        "episode_idx": 1,
                        "success": False,
                        "failure_category": infer_failure_category(success=False, runtime_error=None),
                        "runtime_error": None,
                        "steps_executed": 300,
                        "video_path": "rollout.mp4",
                    },
                ],
            )
            report_path = write_eval_report(tmpdir, report)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["success_rate"], 0.5)
        self.assertEqual(payload["failure_category"], {"timeout_no_success": 1})
        self.assertEqual(payload["checkpoint_hash"], "ckpt-hash")
        self.assertEqual(payload["starflow_mapping"]["framework_name"], "StarFlowVLA")


if __name__ == "__main__":
    unittest.main()
