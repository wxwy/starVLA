"""StarFlow-VLA checkpoint mapping sidecar 测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate.utils import SCALER_NAME
from omegaconf import OmegaConf

from starVLA.model.modules.starflow_vla.mapping import save_starflow_checkpoint_mapping
from starVLA.model.framework.share_tools import _validate_partial_frozen_backbone_load, load_model_weights
from starVLA.training.train_starvla import _iter_model_state_tensors, _streaming_save_model_shards
from starVLA.training.trainer_utils.trainer_tools import (
    build_param_lr_groups,
    load_lightweight_scaler_state,
    save_lightweight_checkpoint_metadata,
    save_lightweight_scaler_state,
)
from starVLA.training.train_starvla import (
    _apply_checkpoint_retention_policy,
    _should_auto_resume_latest_complete,
    VLATrainer,
)


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

    def test_resume_policy_only_resumes_from_an_existing_complete_checkpoint(self):
        self.assertTrue(
            _should_auto_resume_latest_complete(
                "resume_latest_complete_only", False, "/tmp/checkpoints/steps_100"
            )
        )
        self.assertFalse(_should_auto_resume_latest_complete("resume_latest_complete_only", False, None))
        self.assertFalse(
            _should_auto_resume_latest_complete("resume_latest_complete_only", True, "/tmp/checkpoints/steps_100")
        )
        self.assertFalse(_should_auto_resume_latest_complete("disabled", False, "/tmp/checkpoints/steps_100"))


class FrozenBackboneCheckpointTest(unittest.TestCase):
    @staticmethod
    def _model():
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(3, 4)
        model.head = torch.nn.Linear(4, 2)
        return model

    def test_partial_checkpoint_saves_all_non_backbone_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model()
            checkpoint_dir = Path(tmpdir)
            _streaming_save_model_shards(
                model,
                checkpoint_dir,
                "safetensors",
                "1GB",
                save_frozen_backbone=False,
            )
            weight_map = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(weight_map), {"head.weight", "head.bias"})

    def test_partial_checkpoint_allows_only_backbone_missing_keys(self):
        model = self._model()
        _validate_partial_frozen_backbone_load(model, {"head.weight", "head.bias"})
        with self.assertRaisesRegex(RuntimeError, "missing non-backbone"):
            _validate_partial_frozen_backbone_load(model, {"head.weight"})
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            _validate_partial_frozen_backbone_load(
                model,
                {"head.weight", "head.bias", "other.weight"},
            )

    def test_partial_checkpoint_loads_head_and_keeps_initialized_backbone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = self._model()
            checkpoint_dir = Path(tmpdir)
            _streaming_save_model_shards(
                source,
                checkpoint_dir,
                "safetensors",
                "1GB",
                save_frozen_backbone=False,
            )
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"omitted_model_state_prefixes": ["backbone."]}),
                encoding="utf-8",
            )
            target = self._model()
            target_backbone = target.backbone.weight.detach().clone()
            load_model_weights(target, checkpoint_dir, strict=False)
            self.assertTrue(torch.equal(target.head.weight, source.head.weight))
            self.assertTrue(torch.equal(target.backbone.weight, target_backbone))

    def test_partial_checkpoint_keeps_backbone_lora_parameters(self):
        source = self._model()
        source.backbone.register_parameter("lora_A", torch.nn.Parameter(torch.full((2, 3), 1.5)))
        source.backbone.register_parameter("lora_B", torch.nn.Parameter(torch.full((4, 2), 2.5)))
        saved_names = {
            name
            for name, _ in _iter_model_state_tensors(source, save_frozen_backbone=False)
        }
        self.assertEqual(
            saved_names,
            {"backbone.lora_A", "backbone.lora_B", "head.weight", "head.bias"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _streaming_save_model_shards(
                source,
                checkpoint_dir,
                "safetensors",
                "1GB",
                save_frozen_backbone=False,
            )
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"omitted_model_state_prefixes": ["backbone."]}),
                encoding="utf-8",
            )
            target = self._model()
            target.backbone.register_parameter("lora_A", torch.nn.Parameter(torch.zeros(2, 3)))
            target.backbone.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(4, 2)))
            load_model_weights(target, checkpoint_dir, strict=False)
            self.assertTrue(torch.equal(target.backbone.lora_A, source.backbone.lora_A))
            self.assertTrue(torch.equal(target.backbone.lora_B, source.backbone.lora_B))

    def test_saves_wan_lora_as_standalone_adapter(self):
        from diffusers import WanTransformer3DModel
        from peft import LoraConfig
        from safetensors.torch import load_file

        transformer = WanTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=4,
            out_channels=4,
            text_dim=8,
            freq_dim=8,
            ffn_dim=16,
            num_layers=1,
        )
        transformer.add_adapter(
            LoraConfig(r=2, lora_alpha=4, target_modules=["blocks.0.attn2.to_q"]),
            adapter_name="mowa_wan",
        )
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.lora_enabled = True
        model.backbone.lora_adapter_name = "mowa_wan"
        model.backbone.lora_target_modules = ("blocks.0.attn2.to_q",)
        model.backbone.model_name = "test-wan"
        model.backbone.transformer = transformer

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            with patch("starVLA.training.train_starvla.logger"):
                VLATrainer._save_wan_lora_adapter(model, checkpoint_dir)
            adapter_state = load_file(checkpoint_dir / "wan_lora.safetensors")
            metadata = json.loads((checkpoint_dir / "wan_lora_config.json").read_text(encoding="utf-8"))

        self.assertTrue(adapter_state)
        self.assertTrue(all("lora_" in name for name in adapter_state))
        self.assertEqual(metadata["adapter_name"], "mowa_wan")
        self.assertEqual(metadata["target_modules"], ["blocks.0.attn2.to_q"])


class WanLoraOptimizerTest(unittest.TestCase):
    def test_frozen_backbone_keeps_wan_lora_in_optimizer_and_updates_it(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.transformer = torch.nn.Module()
        model.backbone.transformer.register_parameter("base_weight", torch.nn.Parameter(torch.ones(2, 2)))
        model.backbone.transformer.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2, 1)))
        model.backbone.transformer.register_parameter("lora_B", torch.nn.Parameter(torch.ones(1, 2)))
        model.action_model = torch.nn.Linear(2, 2)

        cfg = OmegaConf.create(
            {
                "trainer": {
                    "freeze_modules": "backbone",
                    "learning_rate": {
                        "base": 2.5e-5,
                        "action_model": 1.0e-4,
                        "wan_lora": 1.0e-5,
                    },
                }
            }
        )
        param_groups = build_param_lr_groups(model, cfg)
        lora_group = next(group for group in param_groups if group["name"] == "wan_lora")
        optimizer_param_ids = {
            id(param)
            for group in param_groups
            for param in group["params"]
        }

        self.assertEqual(lora_group["lr"], 1.0e-5)
        self.assertEqual({id(param) for param in lora_group["params"]}, {
            id(model.backbone.transformer.lora_A),
            id(model.backbone.transformer.lora_B),
        })
        self.assertNotIn(id(model.backbone.transformer.base_weight), optimizer_param_ids)

        optimizer = torch.optim.AdamW(param_groups)
        before = model.backbone.transformer.lora_A.detach().clone()
        (model.backbone.transformer.lora_A.sum() + model.backbone.transformer.lora_B.sum()).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone.transformer.lora_A))


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
