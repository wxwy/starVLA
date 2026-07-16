"""StarFlowVLA 复用 QwenPI_v3 的最小 smoke 测试。"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.framework.VLM4A.StarFlowVLA import StarFlowVLA
from starVLA.model.modules.mowa import MoWAFutureGatedHeads


class _CaptureActionModel:
    def __init__(self):
        self.captured_state = None
        self.captured_vl_shapes = None
        self.captured_vl_embs_list = None
        self.captured_attention_mask_shape = None

    def __call__(self, vl_embs_list, actions, state, encoder_attention_mask=None):
        self.captured_state = state
        self.captured_vl_shapes = [tuple(item.shape) for item in vl_embs_list]
        self.captured_vl_embs_list = vl_embs_list
        self.captured_attention_mask_shape = (
            tuple(encoder_attention_mask.shape) if encoder_attention_mask is not None else None
        )
        return torch.tensor(0.0)

    def predict_action(self, vl_embs_list, state=None, encoder_attention_mask=None):
        self.captured_state = state
        self.captured_vl_shapes = [tuple(item.shape) for item in vl_embs_list]
        self.captured_vl_embs_list = vl_embs_list
        self.captured_attention_mask_shape = (
            tuple(encoder_attention_mask.shape) if encoder_attention_mask is not None else None
        )
        return torch.zeros(vl_embs_list[0].shape[0], 8, 7, device=vl_embs_list[0].device)


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


def _minimal_config_with_mowa_layerwise_coupling(
    enabled: bool,
    intervention: str = "baseline",
    action_hidden_dim: int = 4,
    feature_source: str | None = None,
    gated_heads_enabled: bool = False,
) -> SimpleNamespace:
    active_heads = ("task_progress", "action_outcome_class")
    cfg = _minimal_config()
    mowa = {
        "enable_layerwise_bridge_token_coupling": enabled,
        "wam_feature_dim": 4,
        "action_hidden_dim": action_hidden_dim,
        "num_bridge_tokens": 2,
        "layerwise_bridge_token_intervention": intervention,
    }
    if feature_source is not None:
        mowa["layerwise_bridge_feature_source"] = feature_source
        mowa["layerwise_bridge_active_heads"] = active_heads
    if gated_heads_enabled:
        mowa["gated_heads"] = SimpleNamespace(enabled=True)
    cfg.framework.mowa = SimpleNamespace(**mowa)
    return cfg


def _minimal_config_with_mowa_future_supervision(
    *,
    gated_heads_enabled: bool = False,
) -> SimpleNamespace:
    cfg = _minimal_config()
    mowa = {
        "enable_future_supervision_loss": True,
        "future_supervision_hidden_dim": 4,
        "future_supervision_active_heads": ("task_progress", "action_outcome_class"),
    }
    if gated_heads_enabled:
        mowa["gated_heads"] = SimpleNamespace(enabled=True)
    cfg.framework.mowa = SimpleNamespace(**mowa)
    return cfg


class StarFlowVLAReuseTest(unittest.TestCase):
    def test_optional_data_flow_contract_defaults_off_and_validates_inputs(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            framework=SimpleNamespace(
                name="StarFlowVLA",
                state_mode="continuous_head",
                action_model=SimpleNamespace(action_dim=3, state_dim=4),
            )
        )
        model.action_horizon = 2
        model.mowa_future_supervision_loss_enabled = True
        model.mowa_future_supervision_active_heads = ("task_progress",)
        self.assertFalse(model._starflow_should_validate("train"))

        model.starflow_validate_data_flow = True
        model.starflow_validation_steps = 2
        model._starflow_train_validation_count = 0
        example = {
            "image": [object()],
            "lang": "open drawer",
            "state": np.zeros((1, 4), dtype=np.float32),
            "action": np.zeros((2, 3), dtype=np.float32),
            "mowa_future_targets": {"task_progress": 0.5},
            "mowa_future_masks": {"task_progress": True},
        }
        model._validate_starflow_examples([example], phase="train")
        self.assertTrue(model._starflow_should_validate("train"))
        model._starflow_train_validation_count = 2
        self.assertFalse(model._starflow_should_validate("train"))

        missing_state = dict(example)
        missing_state.pop("state")
        with self.assertRaisesRegex(ValueError, "state"):
            model._validate_starflow_examples([missing_state], phase="train")

        bad_action = dict(example, action=np.zeros((2, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "action"):
            model._validate_starflow_examples([bad_action], phase="train")

    def test_data_flow_contract_validates_hidden_and_outputs(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.num_action_dit_layers = 2
        model.action_dit_hidden_dim = 4
        model.action_horizon = 2
        model.config = SimpleNamespace(
            framework=SimpleNamespace(action_model=SimpleNamespace(action_dim=3))
        )
        model.mowa_layerwise_bridge_coupling_enabled = True
        model.mowa_future_supervision_loss_enabled = True
        hidden = [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)]
        model._validate_starflow_hidden_flow(
            hidden,
            torch.ones(1, 3),
            torch.zeros(1, 3, 8),
            phase="train",
        )
        model._validate_starflow_output(
            {
                "action_loss": torch.tensor(1.0),
                "mowa_future_supervision_loss": torch.tensor(0.5),
                "mowa_layerwise_bridge_coupled": True,
            },
            phase="train",
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            model._validate_starflow_hidden_flow(
                [torch.full((1, 3, 4), float("nan")), hidden[1]],
                torch.ones(1, 3),
                torch.zeros(1, 3, 8),
                phase="train",
            )

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
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 1, 1)],
            None,
            torch.zeros(1, 1, 1),
        )
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
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 1, 1)],
            None,
            torch.zeros(1, 1, 1),
        )
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
            torch.zeros(1, 3, 1),
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
        self.assertEqual(output["mowa_layerwise_bridge_bridge_token_shape"], (2, 2, 4))
        self.assertEqual(output["mowa_layerwise_bridge_adapted_vl_embed_shape"], (2, 5, 4))
        self.assertEqual(output["mowa_layerwise_bridge_intervention"], "baseline")
        self.assertTrue(output["mowa_layerwise_bridge_intervention_applied"])
        self.assertIsNone(output["mowa_layerwise_bridge_intervention_note"])
        self.assertEqual(output["mowa_layerwise_bridge_attention_mask_shape"], (2, 5))
        self.assertEqual(output["mowa_layerwise_bridge_feature_source"], "starflow_condition_probe")
        self.assertEqual(output["mowa_layerwise_bridge_active_heads"], ("starflow_condition_probe",))
        self.assertEqual(action_model.captured_vl_shapes, [(2, 5, 4), (2, 5, 4)])
        self.assertEqual(action_model.captured_attention_mask_shape, (2, 5))

    def test_mowa_layerwise_bridge_can_use_future_feature_heads_source(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(
            True,
            feature_source="mowa_future_feature_heads",
        )
        model.config.trainer = {"repeated_diffusion_steps": 1}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1),
        )

        output = model.forward(
            [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                }
            ]
        )

        self.assertTrue(output["mowa_layerwise_bridge_coupled"])
        self.assertEqual(output["mowa_layerwise_bridge_feature_source"], "mowa_future_feature_heads")
        self.assertEqual(
            output["mowa_layerwise_bridge_active_heads"],
            ("task_progress", "action_outcome_class"),
        )
        self.assertNotIn("starflow_condition_probe", output["mowa_layerwise_bridge_active_heads"])
        self.assertEqual(action_model.captured_vl_shapes, [(1, 5, 4), (1, 5, 4)])

    def test_mowa_layerwise_bridge_accepts_future_feature_source_alias(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(
            True,
            feature_source="mowa_future_feature_heads",
        )
        model.config.trainer = {"repeated_diffusion_steps": 1}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1),
        )

        output = model.forward(
            [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                }
            ]
        )

        self.assertTrue(output["mowa_layerwise_bridge_coupled"])
        self.assertEqual(output["mowa_layerwise_bridge_feature_source"], "mowa_future_feature_heads")
        self.assertEqual(
            output["mowa_layerwise_bridge_active_heads"],
            ("task_progress", "action_outcome_class"),
        )
        self.assertEqual(action_model.captured_vl_shapes, [(1, 5, 4), (1, 5, 4)])

    def test_mowa_layerwise_bridge_uses_gated_future_heads_when_enabled(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(
            True,
            feature_source="mowa_future_feature_heads",
            gated_heads_enabled=True,
        )
        model.config.trainer = {"repeated_diffusion_steps": 1}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        self.assertIsInstance(
            model.mowa_layerwise_bridge_future_feature_heads,
            MoWAFutureGatedHeads,
        )
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1),
        )

        output = model.forward(
            [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                }
            ]
        )

        self.assertTrue(output["mowa_future_gated_heads_enabled"])
        self.assertEqual(
            output["mowa_layerwise_bridge_gated_heads_summary"]["comparison_scope"],
            "single_fullheads_control_only",
        )
        self.assertFalse(
            output["mowa_layerwise_bridge_gated_heads_summary"]["allow_per_head_sweep"]
        )

    def test_mowa_layerwise_bridge_rejects_unknown_feature_source(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(
            True,
            feature_source="unknown_wam_source",
        )
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2

        with self.assertRaisesRegex(ValueError, "feature source"):
            model._setup_mowa_layerwise_bridge_coupling()

    def test_mowa_layerwise_bridge_coupling_fails_fast_on_hidden_dim_mismatch(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(True, action_hidden_dim=8)
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2

        with self.assertRaisesRegex(ValueError, "action_hidden_dim.*action_dit_hidden_dim"):
            model._setup_mowa_layerwise_bridge_coupling()

    def test_mowa_layerwise_bridge_interventions_are_explicit(self):
        def run_forward(intervention):
            torch.manual_seed(0)
            model = object.__new__(StarFlowVLA)
            torch.nn.Module.__init__(model)
            model.config = _minimal_config_with_mowa_layerwise_coupling(
                True,
                intervention,
                feature_source="mowa_future_feature_heads",
            )
            model.config.trainer = {"repeated_diffusion_steps": 1}
            model.action_horizon = 8
            model.action_dit_hidden_dim = 4
            model.num_action_dit_layers = 2
            model._setup_mowa_layerwise_bridge_coupling()
            action_model = _CaptureActionModel()
            model.action_model = action_model
            model._encode_vl_hidden_states = lambda images, instructions: (
                [torch.zeros(2, 3, 4), torch.ones(2, 3, 4)],
                torch.ones(2, 3, dtype=torch.bool),
                torch.zeros(2, 3, 1),
            )
            examples = [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                },
                {
                    "image": [],
                    "lang": "close the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                },
            ]
            output = model.forward(examples)
            bridge_tokens = action_model.captured_vl_embs_list[0][:, -2:, :].detach()
            return output, bridge_tokens

        baseline_output, baseline_tokens = run_forward("baseline")
        zero_output, zero_tokens = run_forward("zero")
        shuffle_output, shuffle_tokens = run_forward("batch_shuffle")
        head_mask_output, head_mask_tokens = run_forward("head_mask_control")

        self.assertEqual(baseline_output["mowa_layerwise_bridge_intervention"], "baseline")
        self.assertEqual(zero_output["mowa_layerwise_bridge_intervention"], "zero")
        self.assertEqual(shuffle_output["mowa_layerwise_bridge_intervention"], "batch_shuffle")
        self.assertEqual(head_mask_output["mowa_layerwise_bridge_intervention"], "head_mask_control")
        self.assertTrue(torch.allclose(zero_tokens, torch.zeros_like(zero_tokens)))
        self.assertTrue(torch.allclose(shuffle_tokens[0], baseline_tokens[1]))
        self.assertTrue(torch.allclose(shuffle_tokens[1], baseline_tokens[0]))
        self.assertFalse(torch.allclose(head_mask_tokens, torch.zeros_like(head_mask_tokens)))
        self.assertFalse(torch.allclose(head_mask_tokens, baseline_tokens))
        self.assertEqual(
            head_mask_output["mowa_layerwise_bridge_intervention_note"],
            "rebuilt_from_constructible_head_outputs",
        )
        self.assertEqual(
            head_mask_output["mowa_layerwise_bridge_active_heads"],
            ("task_progress", "action_outcome_class"),
        )

    def test_mowa_head_mask_control_requires_future_feature_source(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(True, "head_mask_control")
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2

        with self.assertRaisesRegex(ValueError, "future feature-head source"):
            model._setup_mowa_layerwise_bridge_coupling()

    def test_mowa_batch_shuffle_intervention_reports_single_sample_noop(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(True, "batch_shuffle")
        model.config.trainer = {"repeated_diffusion_steps": 1}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        model.action_model = _CaptureActionModel()
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1),
        )

        output = model.forward(
            [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                }
            ]
        )

        self.assertEqual(output["mowa_layerwise_bridge_intervention"], "batch_shuffle")
        self.assertFalse(output["mowa_layerwise_bridge_intervention_applied"])
        self.assertEqual(
            output["mowa_layerwise_bridge_intervention_note"],
            "batch_shuffle_not_applied_due_to_batch_size",
        )

    def test_mowa_layerwise_bridge_coupling_is_applied_in_predict_action(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_layerwise_coupling(True, "zero")
        model.config.datasets = SimpleNamespace(vla_data=SimpleNamespace(obs_image_size=None))
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_layerwise_bridge_coupling()
        action_model = _CaptureActionModel()
        model.action_model = action_model
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(2, 3, 4), torch.ones(2, 3, 4)],
            torch.ones(2, 3, dtype=torch.bool),
            torch.zeros(2, 3, 1),
        )
        examples = [
            {"image": [], "lang": "open the drawer"},
            {"image": [], "lang": "close the drawer"},
        ]

        output = model.predict_action(examples)
        bridge_tokens = action_model.captured_vl_embs_list[0][:, -2:, :].detach()

        self.assertEqual(output["normalized_actions"].shape, (2, 8, 7))
        self.assertTrue(output["mowa_layerwise_bridge_coupled"])
        self.assertEqual(output["mowa_layerwise_bridge_intervention"], "zero")
        self.assertTrue(output["mowa_layerwise_bridge_intervention_applied"])
        self.assertEqual(output["mowa_layerwise_bridge_token_shape"], (2, 2, 4))
        self.assertEqual(output["mowa_layerwise_bridge_bridge_token_shape"], (2, 2, 4))
        self.assertEqual(output["mowa_layerwise_bridge_adapted_vl_embed_shape"], (2, 5, 4))
        self.assertEqual(output["mowa_layerwise_bridge_attention_mask_shape"], (2, 5))
        self.assertEqual(action_model.captured_vl_shapes, [(2, 5, 4), (2, 5, 4)])
        self.assertEqual(action_model.captured_attention_mask_shape, (2, 5))
        self.assertTrue(torch.allclose(bridge_tokens, torch.zeros_like(bridge_tokens)))

    def test_mowa_future_supervision_uses_gated_heads_when_enabled(self):
        model = object.__new__(StarFlowVLA)
        torch.nn.Module.__init__(model)
        model.config = _minimal_config_with_mowa_future_supervision(gated_heads_enabled=True)
        model.config.trainer = {"repeated_diffusion_steps": 1}
        model.action_horizon = 8
        model.action_dit_hidden_dim = 4
        model.num_action_dit_layers = 2
        model._setup_mowa_future_supervision_loss()
        self.assertIsInstance(model.mowa_future_supervision_probe, MoWAFutureGatedHeads)
        model.action_model = _CaptureActionModel()
        model._encode_vl_hidden_states = lambda images, instructions: (
            [torch.zeros(1, 3, 4), torch.ones(1, 3, 4)],
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1),
        )

        output = model.forward(
            [
                {
                    "image": [],
                    "lang": "open the drawer",
                    "action": np.zeros((8, 7), dtype=np.float32),
                    "mowa_future_targets": {
                        "task_progress": 0.5,
                        "action_outcome_class": [0.0, 1.0],
                    },
                    "mowa_future_masks": {
                        "task_progress": True,
                        "action_outcome_class": True,
                    },
                }
            ]
        )

        self.assertTrue(output["mowa_future_supervision_available"])
        self.assertIn("mowa_future_supervision_loss", output)
        self.assertTrue(output["mowa_future_gated_heads_enabled"])
        self.assertEqual(
            output["mowa_future_supervision_gated_heads_summary"]["comparison_scope"],
            "single_fullheads_control_only",
        )
        self.assertFalse(
            output["mowa_future_supervision_gated_heads_summary"]["allow_per_head_sweep"]
        )


if __name__ == "__main__":
    unittest.main()
