"""StarFlowVLA 复用 QwenPI_v3 的最小 smoke 测试。"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.framework.VLM4A.StarFlowVLA import StarFlowVLA


class _CaptureActionModel:
    def __init__(self):
        self.captured_state = None
        self.captured_vl_shapes = None
        self.captured_attention_mask_shape = None

    def __call__(self, vl_embs_list, actions, state, encoder_attention_mask=None):
        self.captured_state = state
        self.captured_vl_shapes = [tuple(item.shape) for item in vl_embs_list]
        self.captured_attention_mask_shape = (
            tuple(encoder_attention_mask.shape) if encoder_attention_mask is not None else None
        )
        return torch.tensor(0.0)


def _minimal_config(state_mode=None) -> SimpleNamespace:
    framework = SimpleNamespace(
        name="StarFlowVLA",
        action_model=SimpleNamespace(
            action_model_type="LayerwiseFM",
            num_target_vision_tokens=32,
            num_inference_timesteps=4,
        ),
    )
    if state_mode is not None:
        framework.state_mode = state_mode
    return SimpleNamespace(
        version_id="0.21",
        framework=framework,
    )


def _minimal_config_with_mowa_dtype_alignment(enabled: bool) -> SimpleNamespace:
    cfg = _minimal_config()
    cfg.framework.mowa = SimpleNamespace(enable_qwenpi_projector_dtype_alignment=enabled)
    return cfg


def _minimal_config_with_mowa_layerwise_coupling(enabled: bool) -> SimpleNamespace:
    cfg = _minimal_config()
    cfg.framework.mowa = SimpleNamespace(
        enable_layerwise_bridge_token_coupling=enabled,
        wam_feature_dim=4,
        action_hidden_dim=4,
        num_bridge_tokens=2,
    )
    return cfg


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

    def test_default_state_mode_keeps_discretized_instruction_path(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config()
        instructions = ["open the drawer"]
        states = [np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]], dtype=np.float32)]

        updated_instructions, action_head_state = model._prepare_state_condition(instructions, states)

        self.assertTrue(updated_instructions[0].startswith("open the drawer [STATE] "))
        self.assertTrue(updated_instructions[0].endswith(" [ACTION]"))
        self.assertIsNone(action_head_state)

    def test_continuous_head_state_mode_preserves_raw_state(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config(state_mode="continuous_head")
        instructions = ["open the drawer"]
        states = [np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]], dtype=np.float32)]

        updated_instructions, action_head_state = model._prepare_state_condition(instructions, states)

        self.assertEqual(updated_instructions, instructions)
        self.assertIs(action_head_state, states)

    def test_continuous_head_forward_passes_state_to_action_head(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config(state_mode="continuous_head")
        model.config.trainer = {"repeated_diffusion_steps": 2}
        model.action_horizon = 8
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: ([torch.zeros(1, 1, 1)], None)
        examples = [
            {
                "image": [],
                "lang": "open the drawer",
                "action": np.zeros((8, 7), dtype=np.float32),
                "state": np.zeros((1, 8), dtype=np.float32),
            }
        ]

        output = model.forward(examples)

        self.assertIn("action_loss", output)
        self.assertIsNotNone(action_model.captured_state)
        self.assertEqual(tuple(action_model.captured_state.shape), (2, 1, 8))

    def test_default_forward_does_not_pass_state_to_action_head(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config()
        model.config.trainer = {"repeated_diffusion_steps": 2}
        model.action_horizon = 8
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: ([torch.zeros(1, 1, 1)], None)
        examples = [
            {
                "image": [],
                "lang": "open the drawer",
                "action": np.zeros((8, 7), dtype=np.float32),
                "state": np.zeros((1, 8), dtype=np.float32),
            }
        ]

        output = model.forward(examples)

        self.assertIn("action_loss", output)
        self.assertIsNone(action_model.captured_state)

    def test_unsupported_state_mode_fails_fast(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config(state_mode="hybrid_gated")
        instructions = ["open the drawer"]
        states = [np.array([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25]], dtype=np.float32)]

        with self.assertRaisesRegex(ValueError, "Unsupported StarFlowVLA state_mode"):
            model._prepare_state_condition(instructions, states)

    def test_mapping_method_is_json_ready(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config()

        mapping = model.describe_starflow_mapping()

        self.assertEqual(mapping["framework_name"], "StarFlowVLA")
        self.assertEqual(mapping["base_framework"], "QwenPI_v3")
        self.assertEqual(mapping["action_head"], "LayerwiseFM")
        self.assertEqual(mapping["state_mode"], "discretized_instruction")
        self.assertTrue(mapping["state_enters_instruction"])
        self.assertFalse(mapping["state_enters_action_head"])
        self.assertFalse(mapping["flow_condition_runtime"])
        self.assertFalse(mapping["perceiver_enabled"])

    def test_continuous_head_mapping_is_json_ready(self):
        model = object.__new__(StarFlowVLA)
        model.config = _minimal_config(state_mode="continuous_head")

        mapping = model.describe_starflow_mapping()

        self.assertEqual(mapping["state_mode"], "continuous_head")
        self.assertFalse(mapping["state_enters_instruction"])
        self.assertTrue(mapping["state_enters_action_head"])

    def test_mowa_dtype_alignment_is_opt_in(self):
        class CaptureDtype(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.seen_dtype = None

            def forward(self, x):
                self.seen_dtype = x.dtype
                return x

        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config()
        default_projector = CaptureDtype()
        model.project_layers = torch.nn.ModuleList([default_projector])
        model._project_vl_hidden_for_action([torch.ones(1, 1, 1, dtype=torch.bfloat16)])
        self.assertEqual(default_projector.seen_dtype, torch.bfloat16)

        model.config = _minimal_config_with_mowa_dtype_alignment(True)
        aligned_projector = CaptureDtype()
        model.project_layers = torch.nn.ModuleList([aligned_projector])
        model._project_vl_hidden_for_action([torch.ones(1, 1, 1, dtype=torch.bfloat16)])
        self.assertEqual(aligned_projector.seen_dtype, torch.float32)

    def test_mowa_layerwise_bridge_coupling_is_opt_in(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(True)
        model.config.trainer = {"repeated_diffusion_steps": 2}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
        )
        examples = [
            {
                "image": [],
                "lang": "open the drawer",
                "action": np.zeros((8, 7), dtype=np.float32),
            }
        ]

        output = model.forward(examples)

        self.assertIn("action_loss", output)
        self.assertTrue(output["mowa_layerwise_bridge_coupled"])
        self.assertEqual(output["mowa_layerwise_bridge_token_shape"], (2, 2, 4))
        self.assertEqual(output["mowa_layerwise_bridge_attention_mask_shape"], (2, 5))
        self.assertEqual(action_model.captured_vl_shapes, [(2, 5, 4), (2, 5, 4)])
        self.assertEqual(action_model.captured_attention_mask_shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
