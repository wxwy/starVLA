"""E003/E003-B2 双视角 Wan 结构回归。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.WanPI import WanPIDefaultConfig, Wan_PI, _prepare_action_state, _require_finite_tensor
from starVLA.model.framework.WM4A.WanOFT import WanOFTDefaultConfig, Wan_OFT
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import MLP, masked_action_flow_loss
from starVLA.model.modules.world_model.Wan2 import _Wan2_Interface, resolve_wan_lora_target_modules
from starVLA.model.modules.mowa.multiview_wan import (
    CrossViewAttentionAdapter,
    MultiViewDoneHead,
    MultiViewFutureFusion,
    MultiViewPatchGrid,
    flatten_view_batch,
    masked_future_flow_loss,
    resolve_cross_view_layer_indices,
    unflatten_view_batch,
)


class MultiViewWanTest(unittest.TestCase):
    def test_wan_vae_input_modes_are_explicit_and_isolated(self):
        self.assertTrue(WanOFTDefaultConfig().world_model["legacy_vae_input"])
        self.assertFalse(WanPIDefaultConfig().world_model["legacy_vae_input"])

        repo_root = Path(__file__).resolve().parents[2]
        config_paths = sorted((repo_root / "configs" / "mowa").rglob("*.yaml"))
        wanpi_configs = []
        for config_path in config_paths:
            config = OmegaConf.load(config_path)
            framework = config.get("framework")
            if framework is None or framework.get("name") != "WanPI":
                continue
            wanpi_configs.append(config_path)
            self.assertIn("world_model", framework, config_path.as_posix())
            self.assertFalse(framework.world_model.legacy_vae_input, config_path.as_posix())

        self.assertTrue(wanpi_configs)

    def test_wanoft_multiview_residual_initialization_preserves_base_feature(self):
        hidden_dim = 8
        owner = SimpleNamespace(
            multi_view_residual_enabled=True,
            multi_view_fusion=torch.nn.Linear(2 * hidden_dim, hidden_dim),
        )
        Wan_OFT._reset_multi_view_fusion(owner)
        base = torch.randn(3, hidden_dim, dtype=torch.bfloat16)
        wrist = torch.randn(3, hidden_dim, dtype=torch.bfloat16)

        fused = Wan_OFT._fuse_action_features(owner, base, wrist)

        self.assertTrue(torch.equal(fused, base.float()))
        self.assertTrue(
            torch.equal(
                owner.multi_view_fusion.weight[:, :hidden_dim],
                torch.eye(hidden_dim),
            )
        )
        self.assertEqual(float(owner.multi_view_fusion.weight[:, hidden_dim:].abs().sum()), 0.0)
        self.assertEqual(float(owner.multi_view_fusion.bias.abs().sum()), 0.0)

    def test_wanoft_multiview_residual_requires_exactly_two_views(self):
        with self.assertRaisesRegex(ValueError, "image=.main,wrist."):
            Wan_OFT._wrist_only_images([[torch.zeros(2, 2, 3)]])

        main = torch.zeros(2, 2, 3)
        wrist = torch.ones(2, 2, 3)
        self.assertIs(Wan_OFT._wrist_only_images([[main, wrist]])[0][0], wrist)

    def test_wanoft_cached_text_inputs_squeeze_instruction_dimension(self):
        examples = [
            {
                "text_embeds": torch.zeros(1, 4, 8, dtype=torch.float16),
                "text_attention_mask": torch.ones(1, 4, dtype=torch.int64),
            },
            {
                "text_embeds": torch.ones(1, 4, 8, dtype=torch.float16),
                "text_attention_mask": torch.ones(1, 4, dtype=torch.int64),
            },
        ]
        cached = Wan_OFT._collect_cached_text_inputs(examples)
        self.assertEqual(tuple(cached["text_embeds"].shape), (2, 4, 8))
        self.assertEqual(tuple(cached["text_attention_mask"].shape), (2, 4))

        with self.assertRaisesRegex(ValueError, "every sample or none"):
            Wan_OFT._collect_cached_text_inputs([examples[0], {}])

    def test_wan_instruction_cache_lookup_and_encoder_release(self):
        backbone = _Wan2_Interface.__new__(_Wan2_Interface)
        torch.nn.Module.__init__(backbone)
        backbone._instruction_text_cache = {
            "pick up cup": {
                "text_embeds": torch.ones(1, 3, 4, dtype=torch.float16),
                "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.int64),
            }
        }
        embeds, mask = backbone._lookup_cached_text(["pick up cup", "pick up cup"])
        self.assertEqual(tuple(embeds.shape), (2, 3, 4))
        self.assertEqual(tuple(mask.shape), (2, 3))
        with self.assertRaisesRegex(KeyError, "cache miss"):
            backbone._lookup_cached_text(["unknown instruction"])

        backbone.text_encoder = torch.nn.Linear(2, 2)
        backbone.tokenizer = object()
        backbone.release_text_encoder()
        self.assertIsNone(backbone.text_encoder)
        self.assertIsNone(backbone.tokenizer)

    def test_wanoft_old_checkpoint_only_allows_missing_residual_fusion(self):
        owner = Wan_OFT.__new__(Wan_OFT)
        torch.nn.Module.__init__(owner)
        owner.multi_view_residual_enabled = True
        owner.multi_view_fusion = torch.nn.Linear(4, 2)

        result = owner.load_state_dict({}, strict=True)
        self.assertEqual(
            set(result.missing_keys),
            {"multi_view_fusion.weight", "multi_view_fusion.bias"},
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            owner.load_state_dict({"wrong.weight": torch.zeros(1)}, strict=True)

    def test_wan_lora_targets_are_group_and_layer_configurable(self):
        class _Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4)
                self.to_k = torch.nn.Linear(4, 4)
                self.to_v = torch.nn.Linear(4, 4)
                self.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Dropout(0.0)])

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = _Attention()
                self.attn2 = _Attention()
                self.ffn = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))

        transformer = torch.nn.Module()
        transformer.blocks = torch.nn.ModuleList([_Block(), _Block(), _Block()])
        attention_targets = resolve_wan_lora_target_modules(
            transformer,
            target_groups=["cross_attention", "self_attention"],
            start_layer=1,
            end_layer=2,
        )
        mlp_targets = resolve_wan_lora_target_modules(transformer, target_groups=["mlp"])

        self.assertEqual(len(attention_targets), 16)
        self.assertTrue(all(name.startswith(("blocks.1.", "blocks.2.")) for name in attention_targets))
        self.assertTrue(all(".attn1." in name or ".attn2." in name for name in attention_targets))
        self.assertEqual(len(mlp_targets), 6)
        self.assertTrue(all(".ffn." in name for name in mlp_targets))

    def test_done_head_fuses_views_per_timestep_and_backpropagates(self):
        torch.manual_seed(11)
        head = MultiViewDoneHead(hidden_dim=8, mlp_hidden_dim=4)
        future_hidden = torch.randn(2, 2, 3, 5, 8, requires_grad=True)
        logits = head(future_hidden)
        target = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertGreater(float(head.mlp[0].weight.grad.norm()), 0.0)
        self.assertGreater(float(head.view_embeddings.weight.grad.norm()), 0.0)
        self.assertGreater(float(future_hidden.grad.norm()), 0.0)

    def test_action_flow_loss_excludes_terminal_padding(self):
        prediction = torch.zeros(1, 3, 2, requires_grad=True)
        target = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        loss = masked_action_flow_loss(prediction, target, mask)
        loss.backward()

        self.assertTrue(torch.allclose(loss, torch.tensor(21.0)))
        self.assertEqual(float(prediction.grad[:, 2].abs().sum()), 0.0)
        terminal_prediction = prediction.detach().clone().requires_grad_(True)
        terminal_loss = masked_action_flow_loss(
            terminal_prediction, target, torch.zeros_like(mask)
        )
        terminal_loss.backward()
        self.assertEqual(float(terminal_loss), 0.0)
        self.assertEqual(float(terminal_prediction.grad.abs().sum()), 0.0)

    def test_action_flow_loss_returns_per_sample_metrics(self):
        prediction = torch.zeros(2, 2, 1)
        target = torch.tensor([[[1.0], [3.0]], [[5.0], [7.0]]])
        mask = torch.tensor([[True, False], [True, True]])

        per_sample = masked_action_flow_loss(
            prediction,
            target,
            mask,
            return_per_sample=True,
        )

        self.assertTrue(torch.allclose(per_sample, torch.tensor([1.0, 37.0])))

    def test_task_metric_windows_merge_across_ranks(self):
        from starVLA.training.train_starvla import _merge_mowa_task_metric_accumulators

        merged = _merge_mowa_task_metric_accumulators([
            {
                "OpenDrawer": {
                    "sample_count": 2,
                    "action_loss_sum": 3,
                    "action_loss_count": 2,
                    "future_main_sum": 4,
                    "future_wrist_sum": 5,
                    "done_positive_sum": 0.5,
                },
            },
            {
                "OpenDrawer": {"sample_count": 1, "future_main_sum": 2},
                "OpenCabinet": {"sample_count": 1, "done_positive_sum": 1},
            },
        ])

        self.assertEqual(merged["OpenDrawer"]["sample_count"], 3.0)
        self.assertEqual(merged["OpenDrawer"]["action_loss_sum"], 3.0)
        self.assertEqual(merged["OpenDrawer"]["future_main_sum"], 6.0)
        self.assertEqual(merged["OpenCabinet"]["done_positive_sum"], 1.0)

    def test_mowa_first_batch_example_contract_covers_all_inputs(self):
        owner = SimpleNamespace(
            mowa_multiview_enabled=True,
            action_horizon=4,
            config=SimpleNamespace(
                framework=SimpleNamespace(action_model=SimpleNamespace(action_dim=3))
            ),
        )
        example = {
            "action": torch.zeros(4, 3),
            "state": torch.zeros(1, 3),
            "text_embeds": torch.zeros(5, 8),
            "text_attention_mask": torch.ones(5),
            "mowa_multi_view_history_latents": torch.zeros(2, 1, 4, 2, 2),
            "mowa_multi_view_current_latents": torch.zeros(2, 4, 2, 2),
            "mowa_multi_view_future_latents": torch.zeros(2, 2, 4, 2, 2),
            "mowa_future_valid_mask": torch.ones(2, dtype=torch.bool),
        }
        Wan_PI._validate_mowa_examples_once(owner, [example], phase="train")

        missing = dict(example)
        missing.pop("text_embeds")
        with self.assertRaisesRegex(ValueError, "text_embeds"):
            Wan_PI._validate_mowa_examples_once(owner, [missing], phase="train")

        bad_action = dict(example, action=torch.zeros(3, 3))
        with self.assertRaisesRegex(ValueError, "action"):
            Wan_PI._validate_mowa_examples_once(owner, [bad_action], phase="train")

        bad_mask = dict(example, mowa_future_valid_mask=torch.ones(3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "future mask"):
            Wan_PI._validate_mowa_examples_once(owner, [bad_mask], phase="train")

        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            _require_finite_tensor("future", torch.tensor([float("inf")]))

    def test_mowa_state_contract_rejects_missing_bad_shape_and_nonfinite(self):
        kwargs = {
            "required": True,
            "expected_dim": 3,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
        with self.assertRaisesRegex(ValueError, "missing batch indices"):
            _prepare_action_state([{"state": [[1.0, 2.0, 3.0]]}, {}], **kwargs)
        with self.assertRaisesRegex(ValueError, "shape"):
            _prepare_action_state([{"state": [1.0, 2.0, 3.0]}], **kwargs)
        with self.assertRaisesRegex(ValueError, "state_dim mismatch"):
            _prepare_action_state([{"state": [[1.0, 2.0]]}], **kwargs)
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            _prepare_action_state([{"state": [[1.0, float("nan"), 3.0]]}], **kwargs)

    def test_mowa_state_encoder_has_gradient_and_state_changes_output(self):
        torch.manual_seed(7)
        encoder = MLP(input_dim=3, output_dim=8)
        state_a = _prepare_action_state(
            [{"state": [[0.0, 0.0, 0.0]]}],
            required=True,
            expected_dim=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state_b = _prepare_action_state(
            [{"state": [[1.0, -2.0, 3.0]]}],
            required=True,
            expected_dim=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        output_a = encoder(state_a)
        output_b = encoder(state_b)
        self.assertFalse(torch.allclose(output_a, output_b))
        output_b.square().mean().backward()
        self.assertGreater(float(encoder.layer1.weight.grad.norm()), 0.0)

    def test_architecture_smoke_forward_backward_optimizer_step(self):
        torch.manual_seed(3)
        batch_size, num_views, time, hidden_dim = 2, 2, 4, 16
        frozen_wan = torch.nn.Linear(hidden_dim, hidden_dim)
        frozen_wan.requires_grad_(False)
        view_embeddings = torch.nn.Embedding(num_views, hidden_dim)
        adapter = CrossViewAttentionAdapter(hidden_dim, 4, bottleneck_dim=8, gate_init=1e-3)
        fusion = MultiViewFutureFusion(hidden_dim, 4, num_queries=3)
        future_head = torch.nn.Linear(hidden_dim, 2)
        action_head = torch.nn.Linear(3 * hidden_dim, 5 * 3)
        trainable = torch.nn.ModuleList(
            [view_embeddings, adapter, fusion, future_head, action_head]
        )
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=1e-3)

        hidden = torch.randn(batch_size, num_views, time, hidden_dim)
        view_ids = torch.arange(num_views)[None, :, None]
        hidden = frozen_wan(hidden) + view_embeddings(view_ids)
        grid = MultiViewPatchGrid(batch_size, num_views, time, 1, 1)
        hidden = unflatten_view_batch(
            adapter(hidden.reshape(batch_size * num_views, time, hidden_dim), grid),
            batch_size,
            num_views,
        )
        future_prediction = future_head(hidden[:, :, -2:]).permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
        future_target = torch.randn_like(future_prediction)
        future_mask = torch.ones(batch_size, num_views, 2, dtype=torch.bool)
        per_view = masked_future_flow_loss(future_prediction, future_target, future_mask)
        loss_future_main = per_view[:, 0].mean()
        loss_future_wrist = per_view[:, 1].mean()
        fused = fusion(hidden[:, :, -2:])
        action_prediction = action_head(fused.reshape(batch_size, -1)).reshape(batch_size, 5, 3)
        loss_action = action_prediction.square().mean()
        loss_total = loss_future_main + loss_future_wrist + loss_action
        before = adapter.gate.detach().clone()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        for loss in (loss_future_main, loss_future_wrist, loss_action, loss_total):
            self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(frozen_wan.weight.grad)
        self.assertGreater(float(view_embeddings.weight.grad.norm()), 0.0)
        self.assertGreater(float(adapter.attention.in_proj_weight.grad.norm()), 0.0)
        self.assertGreater(float(fusion.queries.grad.norm()), 0.0)
        self.assertGreater(float(action_head.weight.grad.norm()), 0.0)
        self.assertFalse(torch.equal(before, adapter.gate.detach()))

    def test_latent_cache_action_horizon_reuses_existing_transform(self):
        from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

        class FakeDataConfig:
            embodiment_tag = "robot"

            def __init__(self):
                self.action_indices = list(range(8))

            def modality_config(self):
                return {"action": SimpleNamespace(delta_indices=tuple(self.action_indices))}

            def transform(self):
                return ("existing_normalizer", len(self.action_indices))

        original = FakeDataConfig()
        data_cfg = {
            "mowa_latent_cache": {
                "future_window_steps": 8,
                "override_action_horizon_from_cache": True,
            },
            "video_backend": "torchvision_av",
        }
        with patch.dict(
            "starVLA.dataloader.lerobot_datasets.ROBOT_TYPE_CONFIG_MAP",
            {"fake": original},
            clear=False,
        ), patch("starVLA.dataloader.lerobot_datasets.LeRobotSingleDataset") as dataset_cls:
            make_LeRobotSingleDataset(Path("/tmp"), "dataset", "fake", data_cfg=data_cfg)

        passed = dataset_cls.call_args.kwargs
        self.assertEqual(tuple(passed["modality_configs"]["action"].delta_indices), tuple(range(32)))
        self.assertEqual(passed["transforms"], ("existing_normalizer", 32))
        self.assertEqual(original.action_indices, list(range(8)))

        with patch.dict(
            "starVLA.dataloader.lerobot_datasets.ROBOT_TYPE_CONFIG_MAP",
            {"fake": original},
            clear=False,
        ), patch("starVLA.dataloader.lerobot_datasets.LeRobotSingleDataset") as dataset_cls:
            make_LeRobotSingleDataset(Path("/tmp"), "dataset", "fake", data_cfg={})
        self.assertEqual(
            tuple(dataset_cls.call_args.kwargs["modality_configs"]["action"].delta_indices),
            tuple(range(8)),
        )

    def test_e003_and_b2_share_multiview_topology(self):
        e003 = OmegaConf.load(
            "configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml"
        )
        b2 = OmegaConf.load(
            "configs/mowa/mowa_e003_b2_multiview_wan_long_training_candidate.yaml"
        )
        self.assertEqual(e003.latent_cache.history_window_steps, 0)
        self.assertEqual(b2.latent_cache.history_window_steps, 10)
        self.assertEqual(e003.interface.batch_size_smoke, 4)
        self.assertEqual(e003.datasets.vla_data.per_device_batch_size, 2)
        self.assertEqual(e003.training.per_device_batch_size, 2)
        self.assertEqual(e003.training.gradient_accumulation_steps, 4)
        self.assertEqual(e003.training.effective_batch_size, 32)
        self.assertEqual(e003.trainer.gradient_accumulation_steps, 4)
        self.assertEqual(
            OmegaConf.to_container(e003.framework.mowa.multi_view, resolve=True),
            OmegaConf.to_container(b2.framework.mowa.multi_view, resolve=True),
        )
        for key in (
            "enabled",
            "mode",
            "bidirectional",
            "num_layers",
            "start_layer_ratio",
            "num_heads",
            "bottleneck_dim",
            "zero_init_output",
        ):
            self.assertEqual(e003.framework.mowa.cross_view[key], b2.framework.mowa.cross_view[key])
        self.assertEqual(tuple(e003.latent_cache.video_keys), tuple(b2.latent_cache.video_keys))
        self.assertFalse(bool(e003.framework.mowa.get("validate_data_flow", False)))
        self.assertEqual(int(e003.framework.mowa.get("validation_steps", 2)), 2)

    def test_default_cross_view_uses_last_ten_blocks(self):
        self.assertEqual(resolve_cross_view_layer_indices(30), tuple(range(20, 30)))
        self.assertEqual(
            resolve_cross_view_layer_indices(30, num_layers=6), tuple(range(24, 30))
        )

    def test_cross_view_bottleneck_parameter_count(self):
        adapter = CrossViewAttentionAdapter(3072, 8, bottleneck_dim=512)
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 4_206_081)
        self.assertEqual(adapter.input_projection.in_features, 3072)
        self.assertEqual(adapter.input_projection.out_features, 512)
        self.assertEqual(adapter.output_projection.in_features, 512)
        self.assertEqual(adapter.output_projection.out_features, 3072)

    def test_flatten_unflatten_preserves_view_pairing(self):
        latents = torch.zeros(3, 2, 1, 2, 1, 1)
        for batch_index in range(3):
            latents[batch_index, 0] = batch_index * 10 + 1
            latents[batch_index, 1] = batch_index * 10 + 2

        flattened = flatten_view_batch(latents)
        restored = unflatten_view_batch(flattened, batch_size=3, num_views=2)

        self.assertEqual(tuple(flattened[:, 0, 0, 0, 0].tolist()), (1, 2, 11, 12, 21, 22))
        self.assertTrue(torch.equal(restored, latents))

    def test_cross_view_only_mixes_same_timestep(self):
        torch.manual_seed(0)
        grid = MultiViewPatchGrid(batch_size=1, num_views=2, time=3, height=1, width=1)
        adapter = CrossViewAttentionAdapter(8, 2, gate_init=1.0, zero_init_output=False).eval()
        hidden = torch.randn(2, 3, 8)
        changed = hidden.clone()
        changed[:, 2] += 100.0

        output = adapter(hidden, grid)
        changed_output = adapter(changed, grid)

        self.assertTrue(torch.allclose(output[:, :2], changed_output[:, :2], atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(output[:, 2], changed_output[:, 2]))

    def test_near_zero_init_and_gradients(self):
        torch.manual_seed(1)
        grid = MultiViewPatchGrid(batch_size=2, num_views=2, time=2, height=2, width=2)
        adapter = CrossViewAttentionAdapter(16, 4, gate_init=1e-3, zero_init_output=True)
        hidden = torch.randn(4, 8, 16, requires_grad=True)

        output = adapter(hidden, grid)
        delta = (output - hidden).abs().max()
        expected_ratio = (output - hidden).detach().float().norm() / hidden.detach().float().norm()
        output.square().mean().backward()

        self.assertLess(float(delta), 1e-5)
        self.assertTrue(torch.allclose(adapter.last_residual_ratio, expected_ratio))
        self.assertGreater(float(adapter.gate.grad.abs()), 0.0)
        self.assertGreater(float(adapter.attention.in_proj_weight.grad.norm()), 0.0)
        self.assertGreater(float(adapter.output_projection.weight.grad.norm()), 0.0)

    def test_future_loss_is_per_view_masked_and_weightable(self):
        prediction = torch.zeros(1, 2, 1, 3, 1, 1, requires_grad=True)
        target = torch.tensor([[[[[[1.0]], [[3.0]], [[100.0]]]], [[[[2.0]], [[100.0]], [[100.0]]]]]])
        valid_mask = torch.tensor([[[True, True, False], [True, False, False]]])

        per_view = masked_future_flow_loss(prediction, target, valid_mask)
        total = per_view[:, 0].mean() + 0.0 * per_view[:, 1].mean()
        total.backward()

        self.assertTrue(torch.allclose(per_view, torch.tensor([[5.0, 4.0]])))
        self.assertEqual(float(prediction.grad[0, 0, 0, 2]), 0.0)
        self.assertEqual(float(prediction.grad[0, 1].abs().sum()), 0.0)


    def test_fusion_restores_original_batch_and_checkpoint(self):
        torch.manual_seed(2)
        fusion = MultiViewFutureFusion(hidden_dim=16, num_heads=4, num_queries=8)
        hidden = torch.randn(3, 2, 5, 16, requires_grad=True)
        output = fusion(hidden)
        output.mean().backward()
        state = fusion.state_dict()
        restored = MultiViewFutureFusion(hidden_dim=16, num_heads=4, num_queries=8)
        restored.load_state_dict(state)

        self.assertEqual(tuple(output.shape), (3, 8, 16))
        self.assertGreater(float(fusion.queries.grad.norm()), 0.0)
        self.assertTrue(torch.equal(restored.queries, fusion.queries))

    def test_predict_action_uses_dual_view_flow_without_future_targets(self):
        class FakeBackbone:
            def __init__(self, owner):
                self.owner = owner
                self.transformer = torch.nn.Linear(1, 1, bias=False)
                self.transformer.config = SimpleNamespace(patch_size=(1, 1, 1))

            def build_inputs(self, *, visual_latents, text_embeds, text_attention_mask, **kwargs):
                batch_size, _, time, height, width = visual_latents.shape
                return {
                    "hidden_states": visual_latents,
                    "timestep": torch.zeros(batch_size, time * height * width, dtype=torch.long),
                    "encoder_hidden_states": text_embeds,
                    "encoder_attention_mask": text_attention_mask,
                }

            def __call__(self, **kwargs):
                hidden_states = kwargs["hidden_states"]
                hidden = hidden_states.permute(0, 2, 3, 4, 1).reshape(
                    hidden_states.shape[0], -1, hidden_states.shape[1]
                )
                self.owner._all_hidden_states[:] = [hidden]
                return SimpleNamespace(sample=torch.zeros_like(hidden_states))

        class FakeActionModel:
            state_encoder = object()

            @staticmethod
            def predict_action(vl_embs_list, state):
                return torch.zeros(vl_embs_list[-1].shape[0], 3, 2, device=vl_embs_list[-1].device)

        class FakeFusion:
            @staticmethod
            def __call__(future_hidden):
                return future_hidden.mean(dim=1)

        model = Wan_PI.__new__(Wan_PI)
        torch.nn.Module.__init__(model)
        model.mowa_multiview_enabled = True
        model.mowa_validate_data_flow = True
        model.mowa_validation_steps = 2
        model._mowa_predict_validation_count = 0
        model._mowa_expected_num_blocks = 1
        model._mowa_multiview_grid = None
        model._all_hidden_states = []
        model.mowa_multiview_debug_mode = "normal"
        model.cross_view_adapters = torch.nn.ModuleDict()
        model.wm_projector = torch.nn.Identity()
        model.mowa_action_view_embeddings = torch.nn.Embedding(2, 4)
        model.mowa_future_fusion = FakeFusion()
        model.mowa_done_head = MultiViewDoneHead(4, mlp_hidden_dim=4)
        model.action_model = FakeActionModel()
        model.action_horizon = 3
        model.config = SimpleNamespace(
            framework=SimpleNamespace(
                action_model=SimpleNamespace(action_dim=2, state_dim=3),
                mowa=SimpleNamespace(
                    multi_view=SimpleNamespace(inference_future_steps=2, inference_flow_steps=2)
                ),
            )
        )
        model.backbone = FakeBackbone(model)
        examples = [{
            "state": torch.zeros(1, 3),
            "text_embeds": torch.zeros(3, 4),
            "text_attention_mask": torch.ones(3),
            "mowa_multi_view_history_latents": torch.zeros(2, 0, 4, 2, 2),
            "mowa_multi_view_current_latents": torch.zeros(2, 4, 2, 2),
        }]

        output = model.predict_action(examples)

        self.assertEqual(output["normalized_actions"].shape, (1, 3, 2))
        self.assertEqual(tuple(output["mowa_predicted_future_latents"].shape), (1, 2, 4, 2, 2, 2))
        self.assertEqual(tuple(output["mowa_future_done_logits"].shape), (1, 2))
        self.assertEqual(model._mowa_predict_validation_count, 1)

    def test_online_raw_views_convert_to_regular_multiview_latents(self):
        class FakeBackbone:
            def __init__(self):
                self.vae_ready = False

            def ensure_vae_for_inference(self):
                self.vae_ready = True

            @staticmethod
            def _encode_images_vae(images, num_frames):
                batch_size = len(images)
                regular_steps = (num_frames - 1) // 4
                values = torch.arange(regular_steps + 1, dtype=torch.float32)
                return values[None, None, :, None, None].expand(batch_size, 4, -1, 2, 2).clone()

        owner = SimpleNamespace(
            config=SimpleNamespace(latent_cache=SimpleNamespace(history_window_steps=1)),
            backbone=FakeBackbone(),
        )
        frames = [object() for _ in range(9)]
        output = Wan_PI._populate_mowa_inference_visual_latents(
            owner,
            [{"mowa_multi_view_images": [frames, frames]}],
        )

        self.assertTrue(owner.backbone.vae_ready)
        self.assertEqual(tuple(output[0]["mowa_multi_view_history_latents"].shape), (2, 1, 4, 2, 2))
        self.assertEqual(tuple(output[0]["mowa_multi_view_current_latents"].shape), (2, 4, 2, 2))
        self.assertTrue(torch.equal(output[0]["mowa_multi_view_history_latents"][:, 0], torch.ones(2, 4, 2, 2)))
        self.assertTrue(torch.equal(output[0]["mowa_multi_view_current_latents"], torch.full((2, 4, 2, 2), 2.0)))

    def test_online_streaming_views_preserve_causal_state_across_requests(self):
        class FakeBackbone:
            def __init__(self):
                self.counts = {}

            def encode_streaming_vae_frames(
                self,
                stream_id,
                frames,
                *,
                reset,
                max_regular_latents,
            ):
                if reset:
                    self.counts[stream_id] = 0
                new_regular = len(frames) // 4
                self.counts[stream_id] = self.counts.get(stream_id, 0) + new_regular
                value = float(self.counts[stream_id])
                return (
                    torch.full((1, 4, 1, 2, 2), value),
                )

        owner = SimpleNamespace(
            config=SimpleNamespace(latent_cache=SimpleNamespace(history_window_steps=0)),
            backbone=FakeBackbone(),
        )
        initial_frames = [object() for _ in range(9)]
        first = Wan_PI._populate_mowa_streaming_visual_latents(
            owner,
            [{
                "mowa_multi_view_images": [initial_frames, initial_frames],
                "mowa_stream_id": "episode-0",
                "mowa_stream_reset": True,
            }],
        )
        next_frames = [object() for _ in range(8)]
        second = Wan_PI._populate_mowa_streaming_visual_latents(
            owner,
            [{
                "mowa_multi_view_images": [next_frames, next_frames],
                "mowa_stream_id": "episode-0",
                "mowa_stream_reset": False,
            }],
        )

        self.assertEqual(tuple(first[0]["mowa_multi_view_history_latents"].shape), (2, 0, 4, 2, 2))
        self.assertTrue(torch.equal(first[0]["mowa_multi_view_current_latents"], torch.full((2, 4, 2, 2), 2.0)))
        self.assertTrue(torch.equal(second[0]["mowa_multi_view_current_latents"], torch.full((2, 4, 2, 2), 4.0)))

    def test_libero_wanpi_client_accumulates_only_new_frames(self):
        import numpy as np

        from examples.LIBERO.eval_files.model2libero_interface import ModelClient

        client = object.__new__(ModelClient)
        client.payload_style = "wanpi"
        client.image_size = (2, 2)
        client._wan_pending_views = [[], []]
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        client.observe_wan_images([frame, frame])
        client.observe_wan_images([frame + 1, frame + 1])

        self.assertEqual([len(frames) for frames in client._wan_pending_views], [2, 2])
        self.assertTrue(np.array_equal(client._wan_pending_views[0][1], frame + 1))

    def test_robocasa_wanpi_client_keeps_initial_and_executed_frames(self):
        import numpy as np

        from examples.Robocasa_365.eval_files.model2robocasa365_interface import PolicyWarper

        client = object.__new__(PolicyWarper)
        client.payload_style = "wanpi"
        client._wan_pending_views = None
        client._wan_stream_reset = None
        initial = np.zeros((1, 8, 2, 2, 3), dtype=np.uint8)
        executed = np.ones((1, 8, 2, 2, 3), dtype=np.uint8)

        client.observe_wan_frames(
            {
                "video.robot0_agentview_left": initial,
                "video.robot0_eye_in_hand": initial,
            },
            initial_only=True,
        )
        client.observe_wan_frames(
            {
                "video.robot0_agentview_left": executed,
                "video.robot0_eye_in_hand": executed,
            }
        )

        self.assertEqual([len(frames) for frames in client._wan_pending_views[0]], [9, 9])
        self.assertTrue(client._wan_stream_reset[0])

    def test_strict_checkpoint_load_ignores_wan_runtime_buffers(self):
        from starVLA.model.framework.share_tools import _filter_strict_key_mismatches

        missing, unexpected = _filter_strict_key_mismatches(
            {"cross_view_adapters.20.gate"},
            {
                "cross_view_adapters.20.gate",
                "cross_view_adapters.20.last_output_norm",
                "cross_view_adapters.20.last_grad_norm",
                "cross_view_adapters.20.last_residual_ratio",
                "backbone.transformer.rope.freqs_cos",
                "backbone.transformer.rope.freqs_sin",
            },
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
