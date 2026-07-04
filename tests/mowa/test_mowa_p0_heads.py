import json
from pathlib import Path
import tempfile
import unittest


class MoWAP0HeadsTest(unittest.TestCase):
    def test_constructible_heads_forward_loss_and_optimizer_step(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAP0ConstructibleHeads,
            MoWAP0ConstructibleHeadsConfig,
            mowa_manual_sgd_step,
        )

        torch.manual_seed(0)
        model = MoWAP0ConstructibleHeads(
            MoWAP0ConstructibleHeadsConfig(input_dim=4, hidden_dim=8)
        )
        features = torch.randn(5, 4)
        targets = {
            "task_progress": torch.rand(5),
            "action_outcome_class": torch.rand(5, 2),
        }
        masks = {
            "task_progress": True,
            "action_outcome_class": True,
            "manipulation_readiness": False,
            "failure_risk": False,
            "next_best_view_score": False,
            "subgoal_feasibility": False,
            "object_visibility_future": False,
        }

        loss, losses, outputs = model.compute_loss(features, targets, masks)
        loss.backward()
        mowa_manual_sgd_step(model, 1e-3)

        self.assertEqual(outputs["task_progress"].shape, (5,))
        self.assertEqual(outputs["action_outcome_class"].shape, (5, 2))
        self.assertIn("task_progress", losses)
        self.assertIn("action_outcome_class", losses)
        self.assertIn("total", losses)
        self.assertNotIn("manipulation_readiness", outputs)
        self.assertTrue(torch.isfinite(loss))

    def test_full_heads_forward_loss_respects_masks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MOWA_P0_FULL_HEADS,
            MoWAP0FullHeads,
            MoWAP0FullHeadsConfig,
        )

        torch.manual_seed(0)
        model = MoWAP0FullHeads(MoWAP0FullHeadsConfig(input_dim=4, hidden_dim=8))
        features = torch.randn(5, 4)
        outputs = model(features)

        self.assertEqual(tuple(outputs.keys()), MOWA_P0_FULL_HEADS)
        self.assertEqual(outputs["task_progress"].shape, (5,))
        self.assertEqual(outputs["manipulation_readiness"].shape, (5,))
        self.assertEqual(outputs["failure_risk"].shape, (5,))
        self.assertEqual(outputs["next_best_view_score"].shape, (5,))
        self.assertEqual(outputs["subgoal_feasibility"].shape, (5,))
        self.assertEqual(outputs["object_visibility_future"].shape, (5,))
        self.assertEqual(outputs["action_outcome_class"].shape, (5, 2))

        masks = {
            "task_progress": True,
            "manipulation_readiness": False,
            "failure_risk": False,
            "next_best_view_score": False,
            "subgoal_feasibility": False,
            "object_visibility_future": False,
            "action_outcome_class": True,
        }
        targets = {
            "task_progress": torch.rand(5),
            "action_outcome_class": torch.rand(5, 2),
        }
        loss, losses, _ = model.compute_loss(features, targets, masks)
        future_features = model.future_features(features, masks)

        self.assertIn("task_progress", losses)
        self.assertIn("action_outcome_class", losses)
        self.assertIn("total", losses)
        self.assertNotIn("failure_risk", losses)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(future_features.hidden_features.shape, (5, 8))
        self.assertEqual(
            future_features.active_heads,
            ("task_progress", "action_outcome_class"),
        )
        self.assertIn("failure_risk", future_features.masked_heads)

    def test_action_bridge_exports_layerwise_condition_features(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAActionBridge,
            MoWAActionBridgeConfig,
            MoWAP0FullHeads,
            MoWAP0FullHeadsConfig,
        )

        torch.manual_seed(0)
        p0_model = MoWAP0FullHeads(MoWAP0FullHeadsConfig(input_dim=4, hidden_dim=8))
        bridge = MoWAActionBridge(
            MoWAActionBridgeConfig(
                wam_feature_dim=8,
                action_hidden_dim=16,
                num_action_layers=3,
                num_bridge_tokens=2,
            )
        )
        features = torch.randn(5, 4)
        masks = {
            "task_progress": True,
            "manipulation_readiness": False,
            "failure_risk": False,
            "next_best_view_score": False,
            "subgoal_feasibility": False,
            "object_visibility_future": False,
            "action_outcome_class": True,
        }

        future_features = p0_model.future_features(features, masks)
        bridge_output = bridge(future_features)

        self.assertEqual(len(bridge_output.layerwise_condition_features), 3)
        for layer_features in bridge_output.layerwise_condition_features:
            self.assertEqual(layer_features.shape, (5, 2, 16))
        self.assertEqual(bridge_output.attention_mask.shape, (5, 2))
        self.assertEqual(bridge_output.attention_mask.dtype, torch.bool)
        self.assertEqual(
            bridge_output.active_heads,
            ("task_progress", "action_outcome_class"),
        )
        self.assertIn("failure_risk", bridge_output.masked_heads)

    def test_action_head_binding_resolves_without_hardcoding_layerwisefm_only(self):
        from starVLA.model.modules.mowa import (
            describe_mowa_action_head_bindings,
            resolve_mowa_action_head_binding,
        )

        layerwise = resolve_mowa_action_head_binding("LayerwiseFM")
        mlp = resolve_mowa_action_head_binding("MLP")
        dit = resolve_mowa_action_head_binding("DiT-B")
        vla_adapter = resolve_mowa_action_head_binding("VLA_Adapter")

        self.assertTrue(layerwise.implemented)
        self.assertEqual(layerwise.injection_mode, "append_bridge_tokens_to_condition_side")
        self.assertTrue(mlp.implemented)
        self.assertEqual(mlp.injection_mode, "add_bridge_summary_to_action_hidden_state")
        self.assertTrue(dit.implemented)
        self.assertEqual(dit.condition_kind, "single_condition_sequence")
        self.assertTrue(vla_adapter.implemented)
        self.assertEqual(vla_adapter.injection_mode, "insert_bridge_tokens_before_action_queries")
        binding_report = describe_mowa_action_head_bindings()
        self.assertIn("LayerwiseFM", {item["action_head_type"] for item in binding_report})
        self.assertIn("VLA_Adapter", {item["action_head_type"] for item in binding_report})

    def test_layerwise_adapter_appends_bridge_tokens_and_attention_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAActionBridge,
            MoWAActionBridgeConfig,
            P0FutureFeatures,
            append_layerwise_bridge_tokens,
        )

        torch.manual_seed(0)
        vl_embs_list = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
        encoder_attention_mask = torch.ones(2, 3, dtype=torch.bool)
        bridge = MoWAActionBridge(
            MoWAActionBridgeConfig(
                wam_feature_dim=6,
                action_hidden_dim=4,
                num_action_layers=2,
                num_bridge_tokens=2,
            )
        )
        bridge_output = bridge(
            P0FutureFeatures(
                hidden_features=torch.randn(2, 6),
                head_outputs={},
                active_heads=("task_progress",),
                masked_heads=(),
            )
        )

        adapted, adapted_mask = append_layerwise_bridge_tokens(
            vl_embs_list,
            encoder_attention_mask,
            bridge_output,
        )

        self.assertEqual(len(adapted), 2)
        self.assertEqual(adapted[0].shape, (2, 5, 4))
        self.assertEqual(adapted[1].shape, (2, 5, 4))
        self.assertEqual(adapted_mask.shape, (2, 5))
        self.assertTrue(torch.equal(adapted_mask[:, :3], encoder_attention_mask))
        self.assertTrue(adapted_mask[:, 3:].all())

    def test_non_layerwise_adapters_transform_without_modifying_action_heads(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAActionBridge,
            MoWAActionBridgeConfig,
            P0FutureFeatures,
            append_single_sequence_bridge_tokens,
            append_vla_adapter_bridge_tokens,
            fuse_mlp_bridge_features,
        )

        torch.manual_seed(0)
        bridge = MoWAActionBridge(
            MoWAActionBridgeConfig(
                wam_feature_dim=6,
                action_hidden_dim=4,
                num_action_layers=3,
                num_bridge_tokens=2,
            )
        )
        bridge_output = bridge(
            P0FutureFeatures(
                hidden_features=torch.randn(2, 6),
                head_outputs={},
                active_heads=("task_progress",),
                masked_heads=(),
            )
        )

        action_hidden = torch.zeros(2, 5, 4)
        fused_hidden = fuse_mlp_bridge_features(action_hidden, bridge_output)
        self.assertEqual(fused_hidden.shape, action_hidden.shape)
        self.assertFalse(torch.equal(fused_hidden, action_hidden))

        condition = torch.zeros(2, 3, 4)
        condition_mask = torch.ones(2, 3, dtype=torch.bool)
        adapted_condition, adapted_condition_mask = append_single_sequence_bridge_tokens(
            condition,
            condition_mask,
            bridge_output,
        )
        self.assertEqual(adapted_condition.shape, (2, 5, 4))
        self.assertEqual(adapted_condition_mask.shape, (2, 5))
        self.assertTrue(adapted_condition_mask[:, 3:].all())

        vla_hidden = torch.zeros(2, 3, 6, 4)
        adapted_vla_hidden = append_vla_adapter_bridge_tokens(
            vla_hidden,
            bridge_output,
            action_query_num=2,
        )
        self.assertEqual(adapted_vla_hidden.shape, (2, 3, 8, 4))
        self.assertTrue(torch.equal(adapted_vla_hidden[:, :, -2:, :], vla_hidden[:, :, -2:, :]))

    def test_e006_coupling_eval_plan_smoke_keeps_eval_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e006_coupling_eval_plan.yaml").write_text(
                "config_role: eval_plan_not_launch\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_action_bridge_interface.yaml").write_text(
                "config_role: interface_draft_not_launch\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_readiness_smoke.json").write_text(
                json.dumps(
                    {
                        "training_started": False,
                        "checks": {"action_bridge_interface_created": True},
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e006_coupling_eval_plan_smoke import (
                build_e006_coupling_eval_plan_smoke,
            )

            report = build_e006_coupling_eval_plan_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertFalse(report["eval_started"])
        self.assertTrue(report["checks"]["plan_config_created"])
        self.assertTrue(report["checks"]["readiness_bridge_check_passed"])
        self.assertIn("feature_removal_bridge_tokens_zeroed", report["interventions"])
        self.assertFalse(report["guardrails"]["modify_layerwisefm_internal_logic"])

    def test_e006_coupling_intervention_smoke_runs_without_training(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not available")

        from tools.mowa.e006_coupling_intervention_smoke import (
            build_e006_coupling_intervention_smoke,
        )

        report = build_e006_coupling_intervention_smoke()

        self.assertFalse(report["training_started"])
        self.assertFalse(report["eval_started"])
        self.assertTrue(report["checks"]["baseline_coupled"])
        self.assertTrue(report["checks"]["zero_tokens_zeroed"])
        self.assertTrue(report["checks"]["batch_shuffle_swaps_samples"])
        self.assertTrue(report["checks"]["head_mask_control_keeps_constructible_heads"])
        self.assertIn("zero", report["interventions"])
        self.assertIn("batch_shuffle", report["interventions"])

    def test_e001_launch_draft_smoke_keeps_launch_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_launch_draft.yaml").write_text(
                (
                    "launch_ready: false\n"
                    "training_started: false\n"
                    "launch_blockers:\n"
                    "  - executable training command candidate exists but is not human-confirmed\n"
                ),
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "policy_confirmed: false\nlaunch_ready: false\ncheckpoint_logic_change_allowed: false\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_draft.yaml").write_text(
                "dry_run_only: true\ncommand: TBD_FULL_E001_ENTRYPOINT\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_ready: false\nrequires_human_confirmation: true\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_launch_candidate.yaml").write_text(
                "launch_ready: false\npolicy_confirmed: false\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_a100_throughput_smoke_plan.yaml").write_text(
                "dry_run_until_on_a100: false\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_a100_throughput_smoke.json").write_text(
                json.dumps({"training_started": False, "checkpoint_saved": False}),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_readiness_smoke.json").write_text(
                json.dumps({"training_started": False}),
                encoding="utf-8",
            )

            from tools.mowa.e001_launch_draft_smoke import build_e001_launch_draft_smoke

            report = build_e001_launch_draft_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertFalse(report["launch_ready"])
        self.assertTrue(report["checks"]["training_command_dry_run_only"])
        self.assertTrue(report["checks"]["training_command_entrypoint_tbd"])
        self.assertTrue(report["checks"]["training_command_candidate_created"])
        self.assertTrue(report["checks"]["launch_candidate_created"])
        self.assertTrue(report["checks"]["candidate_command_not_launch_approved"])
        self.assertTrue(report["checks"]["launch_candidate_not_launch_approved"])
        self.assertTrue(report["checks"]["executable_training_command_candidate_recorded"])
        self.assertTrue(report["checks"]["a100_smoke_no_longer_waiting_for_a100"])
        self.assertTrue(report["checks"]["a100_throughput_report_created"])

    def test_e001_training_config_smoke_requires_mowa_ckpt_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_training_smoke.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: false\n"
                "  training_started: false\n"
                "  dry_run_only: true\n"
                "checkpoint:\n"
                "  save_checkpoint_during_smoke: false\n"
                "  checkpoint_logic_change_allowed: false\n"
                "  run_root_dir: playground/mowa_ckpt\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_draft.yaml").write_text(
                "config: configs/mowa/mowa_e001_training_smoke.yaml\n"
                "entrypoint: TBD_FULL_E001_ENTRYPOINT\n"
                "bounded_smoke_command:\n"
                "  entrypoint: tools/mowa/e001_full_vla_runtime_sweep_smoke.py\n"
                "  counted_as_full_launch: false\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "resource_budget:\n"
                "  full_vla_runtime_sweep_status: passed_bs1_bs2_bs4\n"
                "status:\n  policy_confirmed: false\n"
                "  checkpoint_policy_confirmed: true\n"
                "  logging_policy_confirmed: true\n"
                "  resource_policy_confirmed: false\n"
                "checkpoint:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  local_checkpoint_root: playground/mowa_ckpt\n"
                "  save_resume_smoke_status: passed_steps_1_to_2\n"
                "  eval_load_smoke_status: passed_steps_2_model_load\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_a100_throughput_smoke.json").write_text(
                json.dumps(
                    {
                        "benchmark": "a100_throughput_smoke",
                        "training_started": False,
                        "checkpoint_saved": False,
                        "stable_candidate": {"status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_full_vla_runtime_sweep_bs4_smoke.json").write_text(
                json.dumps(
                    {
                        "bounded_runtime_sweep": True,
                        "full_training_launch": False,
                        "checks": {"ok": True},
                    }
                ),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_readiness_smoke.json").write_text(
                json.dumps({"training_started": False}),
                encoding="utf-8",
            )

            from tools.mowa.e001_training_config_smoke import build_e001_training_config_smoke

            report = build_e001_training_config_smoke(root)

        self.assertTrue(report["checks"]["checkpoint_root_is_mowa_ckpt"])
        self.assertTrue(report["checks"]["checkpoint_save_disabled"])
        self.assertTrue(report["checks"]["checkpoint_policy_confirmed"])
        self.assertTrue(report["checks"]["logging_policy_confirmed"])
        self.assertTrue(report["checks"]["resource_policy_unconfirmed"])
        self.assertTrue(report["checks"]["save_resume_smoke_recorded"])
        self.assertTrue(report["checks"]["eval_load_smoke_recorded"])
        self.assertTrue(report["checks"]["full_vla_runtime_sweep_recorded"])
        self.assertTrue(report["checks"]["bounded_smoke_command_not_full_launch"])
        self.assertTrue(report["checks"]["full_vla_runtime_sweep_bs4_passed"])

    def test_e001_train_starvla_full_path_dry_run_smoke_keeps_training_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_train_starvla_full_path_dry_run.yaml").write_text(
                "trainer:\n  full_path_dry_run_only: true\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_train_starvla_full_path_dry_run.json").write_text(
                json.dumps(
                    {
                        "entrypoint": "starVLA/training/train_starvla.py",
                        "full_path_dry_run_only": True,
                        "training_started": False,
                        "checkpoint_saved": False,
                        "wandb_started": False,
                        "framework": {
                            "name": "QwenOFT",
                            "mowa_action_bridge_probe_enabled": True,
                            "mowa_p0_supervision_probe_enabled": True,
                            "mowa_p0_supervision_label_status": "forward_evaluated_in_full_path_dry_run",
                        },
                        "data": {
                            "data_mix": "robocasa365_open_drawer_target_human",
                            "mowa_p0_labels_enabled": True,
                            "batch_summary": {
                                "fetched": True,
                                "first_item_keys": [
                                    "action",
                                    "image",
                                    "lang",
                                    "mowa_p0_masks",
                                    "mowa_p0_targets",
                                ],
                            },
                        },
                        "forward": {
                            "evaluated": True,
                            "keys": [
                                "action_loss",
                                "mowa_p0_supervision_loss",
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e001_train_starvla_full_path_dry_run_smoke import (
                build_train_starvla_full_path_dry_run_smoke,
            )

            report = build_train_starvla_full_path_dry_run_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertTrue(report["checks"]["training_not_started"])
        self.assertTrue(report["checks"]["checkpoint_not_saved"])
        self.assertTrue(report["checks"]["mowa_p0_supervision_probe_enabled"])
        self.assertTrue(report["checks"]["mowa_p0_supervision_labels_forward_evaluated"])
        self.assertTrue(report["checks"]["batch_has_mowa_p0_targets"])
        self.assertTrue(report["checks"]["forward_has_mowa_p0_supervision_loss"])
        self.assertTrue(report["checks"]["batch_fetched"])

    def test_e001_starflow_ft0_full_path_dry_run_smoke_keeps_training_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_full_path_dry_run.yaml").write_text(
                "trainer:\n  full_path_dry_run_only: true\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_starflow_ft0_full_path_dry_run.json").write_text(
                json.dumps(
                    {
                        "entrypoint": "starVLA/training/train_starvla.py",
                        "full_path_dry_run_only": True,
                        "training_started": False,
                        "checkpoint_saved": False,
                        "wandb_started": False,
                        "framework": {
                            "name": "StarFlowVLA",
                            "action_model_type": "LayerwiseFM",
                            "starflow_ft_variant": "custom",
                            "num_target_vision_tokens": 8,
                            "mowa_layerwise_bridge_coupling_enabled": True,
                            "mowa_layerwise_bridge_coupling_status": "forward_coupled_in_full_path_dry_run",
                            "mowa_layerwise_bridge_feature_source": "mowa_p0_fullheads",
                        },
                        "data": {
                            "data_mix": "robocasa365_open_drawer_target_human",
                            "mowa_p0_labels_enabled": True,
                            "batch_summary": {
                                "first_item_keys": [
                                    "action",
                                    "image",
                                    "lang",
                                    "mowa_p0_masks",
                                    "mowa_p0_targets",
                                    "state",
                                ],
                            },
                        },
                        "forward": {
                            "evaluated": True,
                            "keys": [
                                "action_loss",
                                "mowa_layerwise_bridge_active_heads",
                                "mowa_layerwise_bridge_coupled",
                                "mowa_layerwise_bridge_feature_source",
                            ],
                            "mowa_layerwise_bridge_feature_source": "mowa_p0_fullheads",
                            "mowa_layerwise_bridge_active_heads": [
                                "task_progress",
                                "action_outcome_class",
                            ],
                            "metric_scope": "one_batch_no_backward_forward_dry_run",
                            "elapsed_sec": 1.0,
                            "batch_size": 1,
                            "samples_per_sec": 1.0,
                            "cuda_available": True,
                            "cuda_device": "cuda:0",
                            "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                            "allocated_vram_gb": 0.1,
                            "reserved_vram_gb": 0.2,
                            "peak_vram_gb": 0.1,
                            "peak_reserved_vram_gb": 0.2,
                            "vram_metric_scope": "torch_cuda_allocator_in_full_path_dry_run",
                        },
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e001_starflow_ft0_full_path_dry_run_smoke import (
                build_starflow_ft0_full_path_dry_run_smoke,
            )

            report = build_starflow_ft0_full_path_dry_run_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertTrue(report["checks"]["training_not_started"])
        self.assertTrue(report["checks"]["framework_starflow"])
        self.assertTrue(report["checks"]["action_head_layerwisefm"])
        self.assertTrue(report["checks"]["starflow_ft_variant_configured"])
        self.assertTrue(report["checks"]["forward_has_action_loss"])
        self.assertTrue(report["checks"]["forward_metric_scope_one_batch"])
        self.assertTrue(report["checks"]["forward_has_elapsed_sec"])
        self.assertTrue(report["checks"]["forward_has_samples_per_sec"])
        self.assertTrue(report["checks"]["forward_has_allocated_vram_field"])
        self.assertTrue(report["checks"]["forward_has_reserved_vram_field"])
        self.assertTrue(report["checks"]["forward_has_peak_vram_field"])
        self.assertTrue(report["checks"]["forward_has_peak_reserved_vram_field"])
        self.assertTrue(report["checks"]["forward_vram_metric_scope_recorded"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_coupling_enabled"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_forward_coupled"])
        self.assertTrue(report["checks"]["forward_has_mowa_layerwise_bridge_coupled"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_uses_p0_fullheads"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_active_heads_are_p0"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_not_probe_source"])

    def test_e001_starflow_full_path_dry_run_cli_overrides_are_merged(self):
        try:
            from omegaconf import OmegaConf
            from starVLA.model.framework.share_tools import apply_config_compat
            from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
        except ImportError:
            self.skipTest("OmegaConf or StarVLA config helpers are unavailable")

        cfg = OmegaConf.load("configs/mowa/mowa_e001_starflow_ft0_full_path_dry_run.yaml")
        dotlist = normalize_dotlist_args(
            [
                "--trainer.full_path_dry_run_report",
                "docs_zh/mowa/mowa_e001_starflow_cli_override_dry_run.json",
                "--framework.starflow_ft_variant=custom",
                "--framework.action_model.num_target_vision_tokens",
                "8",
                "--datasets.vla_data.per_device_batch_size",
                "2",
            ]
        )
        cfg = apply_config_compat(OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist)))

        self.assertEqual(
            cfg.trainer.full_path_dry_run_report,
            "docs_zh/mowa/mowa_e001_starflow_cli_override_dry_run.json",
        )
        self.assertEqual(cfg.framework.starflow_ft_variant, "custom")
        self.assertEqual(cfg.framework.action_model.num_target_vision_tokens, 8)
        self.assertEqual(cfg.datasets.vla_data.per_device_batch_size, 2)

    def test_e001_starflow_training_smoke_validates_save_resume_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "MoWA-E-001_starflow_ft0_save_resume_smoke_test"
            run_dir = root / "playground" / "mowa_ckpt" / run_id
            steps_1 = run_dir / "checkpoints" / "steps_1"
            steps_2 = run_dir / "checkpoints" / "steps_2"
            final_model = run_dir / "final_model"
            (root / "configs" / "mowa").mkdir(parents=True)
            steps_1.mkdir(parents=True)
            steps_2.mkdir(parents=True)
            final_model.mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_training_throughput_smoke.yaml").write_text(
                "run_root_dir: playground/mowa_ckpt\n",
                encoding="utf-8",
            )
            (run_dir / "config.full.yaml").write_text("run_id: test\n", encoding="utf-8")
            (steps_1 / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 1}),
                encoding="utf-8",
            )
            (steps_2 / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 2}),
                encoding="utf-8",
            )
            for checkpoint_dir in (steps_1, steps_2):
                (checkpoint_dir / "optimizer_rank_00000.pt").write_bytes(b"placeholder")
                (checkpoint_dir / "scheduler.pt").write_bytes(b"placeholder")
                (checkpoint_dir / "random_states_0.pkl").write_bytes(b"placeholder")

            from tools.mowa.e001_starflow_ft0_training_throughput_smoke import (
                build_starflow_ft0_training_throughput_smoke,
            )

            report = build_starflow_ft0_training_throughput_smoke(
                root,
                run_id,
                commands=[
                    {"phase": "first_train", "returncode": 0},
                    {"phase": "resume", "returncode": 0},
                ],
            )

        self.assertTrue(report["checks"]["run_root_is_mowa_ckpt"])
        self.assertTrue(report["checks"]["first_checkpoint_saved"])
        self.assertTrue(report["checks"]["resume_checkpoint_saved"])
        self.assertTrue(report["checks"]["final_model_saved"])
        self.assertTrue(report["checks"]["first_trainer_state_step_1"])
        self.assertTrue(report["checks"]["resume_trainer_state_step_2"])
        self.assertTrue(report["checks"]["command_runs_succeeded"])

    def test_e001_full_vla_runtime_sweep_summarizes_metrics(self):
        from tools.mowa.e001_full_vla_runtime_sweep_smoke import summarize_runtime_metrics

        summary = summarize_runtime_metrics(
            [
                {
                    "completed_steps": 1,
                    "total_batch_size": 2,
                    "step_time_sec": 4.0,
                    "samples_per_sec": 0.5,
                    "peak_vram_gb": 40.0,
                    "peak_reserved_gb": 42.0,
                    "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                },
                {
                    "completed_steps": 2,
                    "total_batch_size": 2,
                    "step_time_sec": 2.0,
                    "samples_per_sec": 1.0,
                    "peak_vram_gb": 41.0,
                    "peak_reserved_gb": 43.0,
                    "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                },
            ]
        )

        self.assertEqual(summary["completed_steps"], 2)
        self.assertEqual(summary["total_batch_size"], 2)
        self.assertEqual(summary["mean_step_time_sec"], 3.0)
        self.assertEqual(summary["samples_per_sec"], 1.0)
        self.assertEqual(summary["peak_vram_gb"], 41.0)
        self.assertEqual(summary["peak_reserved_gb"], 43.0)

    def test_e001_launch_candidate_smoke_keeps_launch_gated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_launch_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: false\n"
                "  policy_confirmed: false\n"
                "  requires_human_confirmation: true\n"
                "run_root_dir: playground/mowa_ckpt\n"
                "framework:\n"
                "  name: StarFlowVLA\n"
                "  action_model:\n"
                "    action_model_type: LayerwiseFM\n"
                "  mowa:\n"
                "    layerwise_bridge_feature_source: mowa_p0_fullheads\n"
                "datasets:\n"
                "  vla_data:\n"
                "    per_device_batch_size: 4\n"
                "trainer:\n"
                "  max_train_steps: 1000\n"
                "  save_interval: 1000\n"
                "  gradient_accumulation_steps: 1\n"
                "  disable_wandb: true\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: false\n"
                "  requires_human_confirmation: true\n"
                "command_candidate:\n"
                "  config_yaml: configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "status:\n  policy_confirmed: false\n",
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_full_vla_runtime_sweep_bs4_smoke.json").write_text(
                json.dumps(
                    {
                        "bounded_runtime_sweep": True,
                        "full_training_launch": False,
                        "checks": {"ok": True},
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e001_launch_candidate_smoke import build_e001_launch_candidate_smoke

            report = build_e001_launch_candidate_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertFalse(report["launch_ready"])
        self.assertTrue(report["checks"]["launch_candidate_config_created"])
        self.assertTrue(report["checks"]["command_candidate_config_created"])
        self.assertTrue(report["checks"]["candidate_launch_ready_false"])
        self.assertTrue(report["checks"]["command_requires_human_confirmation"])
        self.assertTrue(report["checks"]["candidate_batch_size_4"])
        self.assertTrue(report["checks"]["candidate_max_steps_1000"])
        self.assertTrue(report["checks"]["runtime_policy_still_unconfirmed"])

    def test_e006_eval_load_smoke_validates_checkpoint_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_save_resume_smoke_test"
                / "checkpoints"
                / "steps_2"
            )
            final_model = checkpoint.parents[1] / "final_model"
            (root / "configs" / "mowa").mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            final_model.mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e006_eval_load_smoke.yaml").write_text(
                "checkpoint:\n  checkpoint_root_policy: playground/mowa_ckpt\n",
                encoding="utf-8",
            )
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 2}),
                encoding="utf-8",
            )
            (checkpoint / "starflow_mapping.json").write_text(
                json.dumps({"framework_name": "StarFlowVLA", "action_head": "LayerwiseFM"}),
                encoding="utf-8",
            )
            for name in (
                "model.safetensors.index.json",
                "config.full.yaml",
                "dataset_statistics.json",
                "optimizer_rank_00000.pt",
                "scheduler.pt",
                "random_states_0.pkl",
            ):
                (checkpoint / name).write_bytes(b"placeholder")
            (checkpoint / "model-00001.safetensors").write_bytes(b"placeholder")

            from tools.mowa.e006_eval_load_smoke import build_e006_eval_load_smoke

            report = build_e006_eval_load_smoke(
                root,
                checkpoint=Path(
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/checkpoints/steps_2"
                ),
                final_model=Path(
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/final_model"
                ),
                execute_load=False,
            )

        self.assertTrue(report["checks"]["checkpoint_under_mowa_ckpt"])
        self.assertTrue(report["checks"]["model_shards_exist"])
        self.assertTrue(report["checks"]["trainer_state_step_2"])
        self.assertTrue(report["checks"]["mapping_framework_starflow"])
        self.assertTrue(report["checks"]["mapping_action_head_layerwisefm"])
        self.assertTrue(report["checks"]["model_load_executed_or_not_required"])

    def test_e006_checkpoint_intervention_forward_smoke_builds_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_save_resume_smoke_test"
                / "checkpoints"
                / "steps_2"
            )
            checkpoint.mkdir(parents=True)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_full_path_dry_run.yaml").write_text(
                "trainer:\n  full_path_dry_run_only: true\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_checkpoint_intervention_forward_smoke import (
                build_command,
                run_or_plan_checkpoint_intervention_forward_smoke,
            )

            relative_checkpoint = Path(
                "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/checkpoints/steps_2"
            )
            report = run_or_plan_checkpoint_intervention_forward_smoke(
                root,
                checkpoint=relative_checkpoint,
                execute=False,
                batch_size=2,
            )
            command = build_command(
                checkpoint=relative_checkpoint,
                intervention="batch_shuffle",
                run_id="test",
                report_path=Path("docs_zh/mowa/test.json"),
                batch_size=2,
            )

        self.assertFalse(report["training_started"])
        self.assertFalse(report["eval_started"])
        self.assertTrue(report["checks"]["config_exists"])
        self.assertTrue(report["checks"]["checkpoint_exists"])
        self.assertTrue(report["checks"]["checkpoint_under_mowa_ckpt"])
        self.assertTrue(report["checks"]["batch_size_allows_shuffle"])
        self.assertIn("--framework.mowa.layerwise_bridge_token_intervention", command)
        self.assertIn("batch_shuffle", command)
        self.assertIn("--trainer.full_path_dry_run_load_checkpoint", command)
        self.assertIn("true", command)
        self.assertIn("--trainer.full_path_dry_run_checkpoint", command)
        self.assertIn(str(relative_checkpoint), command)
        self.assertIn("--datasets.vla_data.per_device_batch_size", command)
        self.assertIn("2", command)

    def test_qwenoft_mowa_p0_supervision_probe_requires_explicit_labels(self):
        try:
            import torch
            import torch.nn as nn
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("torch or omegaconf is not available")

        from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

        cfg = OmegaConf.create(
            {
                "framework": {
                    "action_model": {"action_hidden_dim": 8},
                    "mowa": {
                        "enable_p0_supervision_probe": True,
                        "p0_supervision_hidden_dim": 6,
                        "p0_supervision_active_heads": [
                            "task_progress",
                            "action_outcome_class",
                        ],
                    },
                }
            }
        )
        probe_owner = object.__new__(Qwenvl_OFT)
        nn.Module.__init__(probe_owner)
        probe_owner.config = cfg
        probe_owner._setup_mowa_p0_supervision_probe()

        action_queries = torch.randn(4, 5, 8)
        missing = probe_owner._maybe_run_mowa_p0_supervision_probe(action_queries, [{} for _ in range(4)])
        self.assertIsNotNone(missing)
        self.assertFalse(missing["supervision_available"])
        self.assertIsNone(missing["loss"])
        self.assertEqual(missing["active_heads"], ())

        examples = [
            {
                "mowa_p0_targets": {
                    "task_progress": 0.25,
                    "action_outcome_class": [1.0, 0.0],
                },
                "mowa_p0_masks": {
                    "task_progress": True,
                    "action_outcome_class": True,
                },
            }
            for _ in range(4)
        ]
        supervised = probe_owner._maybe_run_mowa_p0_supervision_probe(action_queries, examples)

        self.assertTrue(supervised["supervision_available"])
        self.assertEqual(
            supervised["active_heads"],
            ("task_progress", "action_outcome_class"),
        )
        self.assertIn("task_progress", supervised["losses"])
        self.assertIn("action_outcome_class", supervised["losses"])
        self.assertTrue(torch.isfinite(supervised["loss"]))

    def test_qwenoft_mowa_bridge_probe_uses_action_hidden_without_label_input(self):
        try:
            import torch
            import torch.nn as nn
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("torch or omegaconf is not available")

        from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

        cfg = OmegaConf.create(
            {
                "framework": {
                    "action_model": {"action_hidden_dim": 8},
                    "mowa": {
                        "enable_action_bridge_probe": True,
                        "action_hidden_dim": 16,
                        "num_action_layers": 3,
                        "num_bridge_tokens": 2,
                    },
                }
            }
        )
        probe_owner = object.__new__(Qwenvl_OFT)
        nn.Module.__init__(probe_owner)
        probe_owner.config = cfg
        probe_owner._setup_mowa_action_bridge_probe()

        action_queries = torch.randn(4, 5, 8)
        probe = probe_owner._maybe_run_mowa_action_bridge_probe(action_queries)

        self.assertIsNotNone(probe)
        self.assertEqual(probe["token_shape"], (4, 2, 16))
        self.assertEqual(probe["active_heads"], ("qwen_action_token_probe",))
        self.assertEqual(probe["masked_heads"], ())
        self.assertTrue(torch.isfinite(probe["probe_loss"]))
