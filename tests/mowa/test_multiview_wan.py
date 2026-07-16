"""E003/E003-B2 双视角 Wan 结构回归。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.WM4A.WanPI import Wan_PI, _prepare_action_state, _require_finite_tensor
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import MLP, masked_action_flow_loss
from starVLA.model.modules.mowa.multiview_wan import (
    CrossViewAttentionAdapter,
    MultiViewFutureFusion,
    MultiViewPatchGrid,
    flatten_view_batch,
    masked_future_flow_loss,
    resolve_cross_view_layer_indices,
    unflatten_view_batch,
)


class MultiViewWanTest(unittest.TestCase):
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

    def test_e003_and_b2_share_multiview_config_except_history(self):
        e003 = OmegaConf.load(
            "configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml"
        )
        b2 = OmegaConf.load(
            "configs/mowa/mowa_e003_b2_multiview_wan_long_training_candidate.yaml"
        )
        self.assertEqual(e003.latent_cache.history_window_steps, 0)
        self.assertEqual(b2.latent_cache.history_window_steps, 10)
        self.assertEqual(e003.interface.batch_size_smoke, 4)
        self.assertEqual(e003.datasets.vla_data.per_device_batch_size, 4)
        self.assertEqual(e003.training.per_device_batch_size, 4)
        self.assertEqual(e003.training.gradient_accumulation_steps, 8)
        self.assertEqual(e003.training.effective_batch_size, 32)
        self.assertEqual(e003.trainer.gradient_accumulation_steps, 8)
        self.assertEqual(
            OmegaConf.to_container(e003.framework, resolve=True),
            OmegaConf.to_container(b2.framework, resolve=True),
        )
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
        output.square().mean().backward()

        self.assertLess(float(delta), 1e-5)
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


if __name__ == "__main__":
    unittest.main()
