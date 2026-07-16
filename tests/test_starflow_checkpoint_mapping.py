"""StarFlow-VLA checkpoint mapping sidecar 测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate.utils import SCALER_NAME
from omegaconf import OmegaConf

from starVLA.model.modules.starflow_vla.mapping import save_starflow_checkpoint_mapping
from starVLA.training.trainer_utils.trainer_tools import (
    load_lightweight_scaler_state,
    save_lightweight_checkpoint_metadata,
    save_lightweight_scaler_state,
)
from starVLA.training.train_starvla import _apply_checkpoint_retention_policy


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        version_id="0.21",
        framework=SimpleNamespace(
            action_model=SimpleNamespace(
                action_model_type="LayerwiseFM",
                num_target_vision_tokens=32,
                num_inference_timesteps=4,
            )
        ),
    )


class _DummyScaler:
    def __init__(self, state=None):
        self._state = state or {"scale": 128.0, "growth_tracker": 3}
        self.loaded_state = None

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state_dict):
        self.loaded_state = dict(state_dict)


class StarFlowCheckpointMappingTest(unittest.TestCase):
    def test_save_mapping_inside_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "steps_10"
            path = save_starflow_checkpoint_mapping(
                checkpoint_dir,
                _config(),
                patch_manifest_hash="patch-hash",
                starvla_commit="commit-hash",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "starflow_mapping.json")
        self.assertEqual(payload["framework_name"], "StarFlowVLA")
        self.assertEqual(payload["adapter_mode"], "future_token_cross_dit")
        self.assertEqual(payload["patch_manifest_hash"], "patch-hash")
        self.assertEqual(payload["starvla_commit"], "commit-hash")

    def test_save_mapping_next_to_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = Path(tmpdir) / "steps_10_pytorch_model.pt"
            path = save_starflow_checkpoint_mapping(checkpoint_file, _config())
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "steps_10_pytorch_model.pt.starflow_mapping.json")
        self.assertEqual(payload["config_schema"], "0.21")
        self.assertEqual(payload["num_target_vision_tokens"], 32)


class CheckpointRetentionPolicyTest(unittest.TestCase):
    def test_retains_permanent_and_recent_checkpoints_and_strips_old_optimizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            for step in (5000, 5500, 6000, 6500, 7000):
                path = checkpoint_dir / f"steps_{step}"
                path.mkdir()
                (path / "model.safetensors").write_bytes(b"model")
                (path / "scheduler.pt").write_bytes(b"scheduler")
                (path / "trainer_state.json").write_text("{}", encoding="utf-8")
                (path / "optimizer_rank_00000.pt").write_bytes(b"optimizer")

            _apply_checkpoint_retention_policy(
                checkpoint_dir,
                permanent_steps={5000},
                keep_latest_count=3,
                strip_optimizer_from_non_latest=True,
            )

            self.assertEqual(
                sorted(path.name for path in checkpoint_dir.iterdir()),
                ["steps_5000", "steps_6000", "steps_6500", "steps_7000"],
            )
            self.assertFalse((checkpoint_dir / "steps_5000" / "optimizer_rank_00000.pt").exists())
            self.assertTrue((checkpoint_dir / "steps_6000" / "optimizer_rank_00000.pt").exists())
            self.assertTrue((checkpoint_dir / "steps_6500" / "optimizer_rank_00000.pt").exists())
            self.assertTrue((checkpoint_dir / "steps_7000" / "optimizer_rank_00000.pt").exists())


class StarFlowLightweightCheckpointArtifactTest(unittest.TestCase):
    def test_lightweight_checkpoint_saves_scaler_placeholder_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoints" / "steps_1"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (root / "dataset_statistics.json").write_text('{"dataset":"libero_goal"}\n', encoding="utf-8")

            config = OmegaConf.create(
                {
                    "version_id": "0.21",
                    "framework": {
                        "action_model": {
                            "action_model_type": "LayerwiseFM",
                            "num_target_vision_tokens": 32,
                            "num_inference_timesteps": 4,
                        }
                    },
                }
            )

            torch.save({"state": "optimizer"}, checkpoint_dir / "optimizer_rank_00000.pt")
            torch.save({"state": "scheduler"}, checkpoint_dir / "scheduler.pt")
            (checkpoint_dir / "trainer_state.json").write_text('{"completed_steps": 1}\n', encoding="utf-8")
            save_lightweight_scaler_state(checkpoint_dir, None)
            save_lightweight_checkpoint_metadata(
                checkpoint_dir,
                config,
                local_output_dir=root,
                network_output_dir=root,
            )

            scaler_payload = torch.load(
                checkpoint_dir / SCALER_NAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(scaler_payload, {"has_scaler": False, "state_dict": None})
            self.assertTrue((checkpoint_dir / "config.yaml").exists())
            self.assertTrue((checkpoint_dir / "config.full.yaml").exists())
            self.assertTrue((checkpoint_dir / "dataset_statistics.json").exists())
            self.assertTrue((checkpoint_dir / "starflow_mapping.json").exists())

    def test_lightweight_scaler_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "steps_1"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            scaler = _DummyScaler({"scale": 256.0, "growth_tracker": 7})
            restored = _DummyScaler()

            save_lightweight_scaler_state(checkpoint_dir, scaler)
            loaded = load_lightweight_scaler_state(checkpoint_dir, restored)

            self.assertTrue(loaded)
            self.assertEqual(restored.loaded_state, {"scale": 256.0, "growth_tracker": 7})


if __name__ == "__main__":
    unittest.main()
