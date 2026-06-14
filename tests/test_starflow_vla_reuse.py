"""StarFlowVLA 复用 QwenPI_v3 的最小 smoke 测试。"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.framework.VLM4A.StarFlowVLA import StarFlowVLA


def _minimal_config() -> SimpleNamespace:
    return SimpleNamespace(
        version_id="0.21",
        framework=SimpleNamespace(
            name="StarFlowVLA",
            action_model=SimpleNamespace(
                action_model_type="LayerwiseFM",
                num_target_vision_tokens=32,
                num_inference_timesteps=4,
            ),
        ),
    )


class StarFlowVLAReuseTest(unittest.TestCase):
    def test_starflow_vla_inherits_qwenpi_without_copying_main_paths(self):
        self.assertTrue(issubclass(StarFlowVLA, Qwen_PI_v3))
        self.assertNotIn("forward", StarFlowVLA.__dict__)
        self.assertNotIn("predict_action", StarFlowVLA.__dict__)
        self.assertIs(StarFlowVLA.forward, Qwen_PI_v3.forward)
        self.assertIs(StarFlowVLA.predict_action, Qwen_PI_v3.predict_action)

    def test_build_framework_uses_qwenpi_init_once(self):
        cfg = _minimal_config()
        init_calls = []

        def fake_init(self, config=None, **kwargs):
            init_calls.append((config, kwargs))
            self.config = config

        with mock.patch.object(Qwen_PI_v3, "__init__", fake_init):
            model = build_framework(cfg)

        self.assertIsInstance(model, StarFlowVLA)
        self.assertEqual(init_calls, [(cfg, {})])

    def test_state_to_instruction_path_is_reused(self):
        model = object.__new__(StarFlowVLA)
        instructions = ["open the drawer"]
        states = [np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]], dtype=np.float32)]

        updated = model.add_discretized_state_to_instruction(instructions, states)

        self.assertEqual(len(updated), 1)
        self.assertTrue(updated[0].startswith("open the drawer [STATE] "))
        self.assertTrue(updated[0].endswith(" [ACTION]"))
        self.assertEqual(len(updated[0].split("[STATE] ")[1].split(" [ACTION]")[0].split()), 7)

    def test_mapping_method_is_json_ready(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config()

        mapping = model.describe_starflow_mapping()

        self.assertEqual(mapping["framework_name"], "StarFlowVLA")
        self.assertEqual(mapping["base_framework"], "QwenPI_v3")
        self.assertEqual(mapping["action_head"], "LayerwiseFM")
        self.assertFalse(mapping["flow_condition_runtime"])
        self.assertFalse(mapping["perceiver_enabled"])


if __name__ == "__main__":
    unittest.main()
