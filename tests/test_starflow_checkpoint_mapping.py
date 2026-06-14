"""StarFlow-VLA checkpoint mapping sidecar 测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starVLA.model.modules.starflow_vla.mapping import save_starflow_checkpoint_mapping


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


if __name__ == "__main__":
    unittest.main()
