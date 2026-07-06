import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


class MoWAFutureHeadsTest(unittest.TestCase):
    def test_future_head_constants_are_shared_across_runtime_paths(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import (
            MOWA_FUTURE_CONSTRUCTIBLE_HEADS as lerobot_constructible_heads,
            MOWA_FUTURE_FULL_HEADS as lerobot_full_heads,
        )
        from starVLA.dataloader.mowa.full_head_label_builder import (
            MOWA_FUTURE_CONSTRUCTIBLE_HEADS as label_builder_constructible_heads,
            MOWA_FUTURE_FULL_HEADS as label_builder_full_heads,
        )
        from starVLA.model.modules.mowa import (
            MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            MOWA_FUTURE_FULL_HEADS,
            MOWA_FUTURE_MASKED_HEADS,
        )

        self.assertEqual(label_builder_full_heads, MOWA_FUTURE_FULL_HEADS)
        self.assertEqual(lerobot_full_heads, MOWA_FUTURE_FULL_HEADS)
        self.assertEqual(label_builder_constructible_heads, MOWA_FUTURE_CONSTRUCTIBLE_HEADS)
        self.assertEqual(lerobot_constructible_heads, MOWA_FUTURE_CONSTRUCTIBLE_HEADS)
        self.assertEqual(
            MOWA_FUTURE_MASKED_HEADS,
            tuple(head for head in MOWA_FUTURE_FULL_HEADS if head not in MOWA_FUTURE_CONSTRUCTIBLE_HEADS),
        )

    def test_future_head_aliases_preserve_legacy_compatibility(self):
        from starVLA.model.modules.mowa import (
            MOWA_FUTURE_CONSTRUCTIBLE_HEADS,
            MOWA_FUTURE_FEATURE_HEADS_SOURCE,
            MOWA_FUTURE_FEATURE_SOURCE_ALIASES,
            MOWA_FUTURE_FULL_HEADS,
            MOWA_FUTURE_HEAD_OUTPUT_DIMS,
            MOWA_FUTURE_MASKED_HEADS,
            MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE,
            MoWAFutureConstructibleHeads,
            MoWAFutureConstructibleHeadsConfig,
            MoWAFutureFeatureHeads,
            MoWAFutureFeatureHeadsConfig,
            MoWAFutureFeatures,
            MoWAFutureFullHeads,
            MoWAFutureFullHeadsConfig,
            build_mowa_future_constructible_batch_from_smoke,
        )

        self.assertIs(MOWA_FUTURE_CONSTRUCTIBLE_HEADS, MOWA_FUTURE_CONSTRUCTIBLE_HEADS)
        self.assertIs(MOWA_FUTURE_FULL_HEADS, MOWA_FUTURE_FULL_HEADS)
        self.assertIs(MOWA_FUTURE_MASKED_HEADS, MOWA_FUTURE_MASKED_HEADS)
        self.assertIs(MOWA_FUTURE_HEAD_OUTPUT_DIMS, MOWA_FUTURE_HEAD_OUTPUT_DIMS)
        self.assertEqual(MOWA_STARFLOW_CONDITION_PROBE_FEATURE_SOURCE, "starflow_condition_probe")
        self.assertEqual(MOWA_FUTURE_FEATURE_HEADS_SOURCE, "mowa_future_feature_heads")
        self.assertEqual(
            MOWA_FUTURE_FEATURE_SOURCE_ALIASES,
            (MOWA_FUTURE_FEATURE_HEADS_SOURCE,),
        )
        self.assertIs(MoWAFutureConstructibleHeads, MoWAFutureConstructibleHeads)
        self.assertIs(MoWAFutureConstructibleHeadsConfig, MoWAFutureConstructibleHeadsConfig)
        self.assertIs(MoWAFutureFeatureHeads, MoWAFutureFullHeads)
        self.assertIs(MoWAFutureFeatureHeadsConfig, MoWAFutureFullHeadsConfig)
        self.assertIs(MoWAFutureFeatures, MoWAFutureFeatures)
        self.assertIs(
            build_mowa_future_constructible_batch_from_smoke,
            build_mowa_future_constructible_batch_from_smoke,
        )

    def test_constructible_heads_forward_loss_and_optimizer_step(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAFutureConstructibleHeads,
            MoWAFutureConstructibleHeadsConfig,
            mowa_manual_sgd_step,
        )

        torch.manual_seed(0)
        model = MoWAFutureConstructibleHeads(
            MoWAFutureConstructibleHeadsConfig(input_dim=4, hidden_dim=8)
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

    def test_action_outcome_class_loss_type_can_use_done_cross_entropy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAFutureConstructibleHeads,
            MoWAFutureConstructibleHeadsConfig,
        )

        model = MoWAFutureConstructibleHeads(
            MoWAFutureConstructibleHeadsConfig(
                input_dim=4,
                hidden_dim=8,
                action_outcome_loss_type="cross_entropy_done",
            )
        )
        features = torch.randn(4, 4)
        targets = {
            "action_outcome_class": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        }
        masks = {
            "task_progress": False,
            "action_outcome_class": True,
        }

        loss, losses, outputs = model.compute_loss(features, targets, masks)

        self.assertEqual(outputs["action_outcome_class"].shape, (4, 2))
        self.assertIn("action_outcome_class", losses)
        self.assertTrue(torch.isfinite(loss))

    def test_full_heads_forward_loss_respects_masks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MOWA_FUTURE_FULL_HEADS,
            MoWAFutureFullHeads,
            MoWAFutureFullHeadsConfig,
        )

        torch.manual_seed(0)
        model = MoWAFutureFullHeads(MoWAFutureFullHeadsConfig(input_dim=4, hidden_dim=8))
        features = torch.randn(5, 4)
        outputs = model(features)

        self.assertEqual(tuple(outputs.keys()), MOWA_FUTURE_FULL_HEADS)
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
            MoWAFutureFullHeads,
            MoWAFutureFullHeadsConfig,
        )

        torch.manual_seed(0)
        future_model = MoWAFutureFullHeads(MoWAFutureFullHeadsConfig(input_dim=4, hidden_dim=8))
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

        future_features = future_model.future_features(features, masks)
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

        self.assertTrue(layerwise.adapter_helper_implemented)
        self.assertTrue(layerwise.framework_forward_integrated)
        self.assertEqual(layerwise.injection_mode, "append_bridge_tokens_to_condition_side")
        self.assertTrue(mlp.adapter_helper_implemented)
        self.assertFalse(mlp.framework_forward_integrated)
        self.assertEqual(mlp.injection_mode, "add_bridge_summary_to_action_hidden_state")
        self.assertTrue(dit.adapter_helper_implemented)
        self.assertFalse(dit.framework_forward_integrated)
        self.assertEqual(dit.condition_kind, "single_condition_sequence")
        self.assertTrue(vla_adapter.adapter_helper_implemented)
        self.assertFalse(vla_adapter.framework_forward_integrated)
        self.assertEqual(vla_adapter.injection_mode, "insert_bridge_tokens_before_action_queries")
        binding_report = describe_mowa_action_head_bindings()
        self.assertIn("LayerwiseFM", {item["action_head_type"] for item in binding_report})
        self.assertIn("VLA_Adapter", {item["action_head_type"] for item in binding_report})
        by_head = {item["action_head_type"]: item for item in binding_report}
        self.assertTrue(by_head["LayerwiseFM"]["framework_forward_integrated"])
        self.assertFalse(by_head["MLP"]["framework_forward_integrated"])

    def test_layerwise_adapter_appends_bridge_tokens_and_attention_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAActionBridge,
            MoWAActionBridgeConfig,
            MoWAFutureFeatures,
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
            MoWAFutureFeatures(
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
            MoWAFutureFeatures,
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
            MoWAFutureFeatures(
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

    def test_e001_readiness_reports_missing_core_sot_docs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_root = root / "docs_zh" / "mowa"
            docs_root.mkdir(parents=True)
            for name in (
                "03_agent_implementation_plan.md",
                "04_task_breakdown.md",
                "05_experiment_registry.md",
                "06_data_gate_report.md",
                "07_implementation_log.md",
                "08_starvla_data_benchmark_support_matrix.md",
                "09_future_label_builder_design.md",
                "10_future_latent_cache_manifest_design.md",
            ):
                (docs_root / name).write_text("# placeholder\n", encoding="utf-8")

            from tools.mowa.e001_readiness_smoke import build_e001_readiness_report

            report = build_e001_readiness_report(root)

        self.assertEqual(report["sot"]["status"], "missing_core_sot")
        self.assertEqual(
            report["sot"]["missing_core_docs"],
            (
                "00_project_proposal.md",
                "01_technical_survey.md",
                "02_detailed_design.md",
            ),
        )
        self.assertEqual(report["sot"]["missing_derived_design_docs"], ())
        self.assertTrue(
            any(
                item.startswith("core SOT docs missing=")
                for item in report["unresolved_items"]
            )
        )

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

    def test_e001_launch_draft_smoke_tracks_launch_approved_bounded_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_launch_draft.yaml").write_text(
                (
                    "launch:\n"
                    "  launch_ready: true\n"
                    "  training_started: true\n"
                    "  requires_human_confirmation: true\n"
                    "  human_confirmed: true\n"
                    "  training_completed_1000_steps: true\n"
                    "  reason: short smoke training (1000 steps, bs4) completed; long-training resource policy, "
                    "core SOT, and action-gain evidence remain unconfirmed for full-scale training\n"
                    "launch_blockers:\n"
                    "  - StarFlow ft0 baseline/MoWA launch candidates are aligned and gated, "
                    "but action-gain evidence is not validated\n"
                    "  - executable training command candidate exists and is human-confirmed\n"
                ),
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "status:\n"
                "  policy_confirmed: true\n"
                "  launch_ready: true\n"
                "  resource_policy_confirmed: true\n"
                "checkpoint_logic_change_allowed: false\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_draft.yaml").write_text(
                "dry_run_only: true\ncommand: TBD_FULL_E001_ENTRYPOINT\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: true\n"
                "  requires_human_confirmation: true\n"
                "  human_confirmed: true\n"
                "  policy_confirmed: true\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_launch_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: true\n"
                "  requires_human_confirmation: true\n"
                "  human_confirmed: true\n"
                "  policy_confirmed: true\n",
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
                json.dumps(
                    {
                        "training_started": False,
                        "checks": {
                            "launch_candidate_smoke_passed": True,
                            "starflow_ft0_comparison_smoke_passed": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_launch_candidate_smoke.json").write_text(
                json.dumps({"checks": {"final_parameter_alignment_passed": True}}),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e001_starflow_ft0_comparison_smoke.json").write_text(
                json.dumps({"checks": {"paired_runtime_symmetry_passed": True}}),
                encoding="utf-8",
            )

            from tools.mowa.e001_launch_draft_smoke import build_e001_launch_draft_smoke

            report = build_e001_launch_draft_smoke(root)

        self.assertFalse(report["training_started"])
        self.assertTrue(report["launch_ready"])
        self.assertTrue(report["checks"]["training_command_dry_run_only"])
        self.assertTrue(report["checks"]["training_command_entrypoint_tbd"])
        self.assertTrue(report["checks"]["training_command_candidate_created"])
        self.assertTrue(report["checks"]["launch_candidate_created"])
        self.assertTrue(report["checks"]["launch_draft_state_recorded"])
        self.assertTrue(report["checks"]["runtime_policy_state_consistent"])
        self.assertTrue(report["checks"]["training_command_candidate_state_consistent"])
        self.assertTrue(report["checks"]["launch_candidate_state_consistent"])
        self.assertTrue(report["checks"]["launch_candidate_smoke_passed"])
        self.assertTrue(report["checks"]["starflow_comparison_smoke_passed"])
        self.assertTrue(report["checks"]["executable_training_command_candidate_recorded"])
        self.assertTrue(report["checks"]["launch_draft_reason_current"])
        self.assertTrue(report["checks"]["launch_blocker_mentions_action_gain_not_feature_source"])
        self.assertTrue(report["checks"]["a100_smoke_no_longer_waiting_for_a100"])
        self.assertTrue(report["checks"]["a100_throughput_report_created"])
        self.assertTrue(report["checks"]["readiness_launch_candidate_smoke_passed"])
        self.assertTrue(report["checks"]["readiness_starflow_comparison_smoke_passed"])

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
                            "mowa_future_supervision_probe_enabled": True,
                            "mowa_future_supervision_label_status": "forward_evaluated_in_full_path_dry_run",
                        },
                        "data": {
                            "data_mix": "robocasa365_open_drawer_target_human",
                            "mowa_future_labels_enabled": True,
                            "batch_summary": {
                                "fetched": True,
                                "first_item_keys": [
                                    "action",
                                    "image",
                                    "lang",
                                    "mowa_future_masks",
                                    "mowa_future_targets",
                                ],
                            },
                        },
                        "forward": {
                            "evaluated": True,
                            "keys": [
                                "action_loss",
                                "mowa_future_supervision_loss",
                                "mowa_future_supervision_loss",
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
        self.assertTrue(report["checks"]["mowa_future_supervision_probe_enabled"])
        self.assertTrue(report["checks"]["mowa_future_supervision_labels_forward_evaluated"])
        self.assertTrue(report["checks"]["batch_has_mowa_future_targets"])
        self.assertTrue(report["checks"]["forward_has_mowa_future_supervision_loss"])
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
                            "mowa_future_gated_heads_enabled": True,
                            "mowa_layerwise_bridge_coupling_status": "forward_coupled_in_full_path_dry_run",
                            "mowa_layerwise_bridge_feature_source": "mowa_future_feature_heads",
                            "mowa_layerwise_bridge_gated_heads_summary": {
                                "comparison_scope": "single_fullheads_control_only",
                                "allow_per_head_sweep": False,
                                "step": 0,
                            },
                        },
                        "data": {
                            "data_mix": "robocasa365_open_drawer_target_human",
                            "mowa_future_labels_enabled": True,
                            "batch_summary": {
                                "first_item_keys": [
                                    "action",
                                    "image",
                                    "lang",
                                    "mowa_future_masks",
                                    "mowa_future_targets",
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
                            "mowa_layerwise_bridge_feature_source": "mowa_future_feature_heads",
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
        self.assertTrue(report["checks"]["mowa_future_gated_heads_flag_recorded"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_forward_coupled"])
        self.assertTrue(report["checks"]["forward_has_mowa_layerwise_bridge_coupled"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_uses_future_feature_heads"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_active_heads_are_p0"])
        self.assertTrue(report["checks"]["mowa_layerwise_bridge_not_probe_source"])
        self.assertTrue(report["checks"]["mowa_gated_heads_summary_is_structured_when_enabled"])

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

    def test_e001_launch_candidate_smoke_accepts_launch_gated_state(self):
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
                "    enable_future_supervision_loss: true\n"
                "    layerwise_bridge_feature_source: mowa_future_feature_heads\n"
                "datasets:\n"
                "  vla_data:\n"
                "    per_device_batch_size: 4\n"
                "trainer:\n"
                "  max_train_steps: 1000\n"
                "  save_interval: 1000\n"
                "  gradient_accumulation_steps: 1\n"
                "  disable_wandb: true\n"
                "  checkpoint_format: lightweight\n"
                "  save_checkpoint_as_directory: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: false\n"
                "  requires_human_confirmation: true\n"
                "  policy_confirmed: false\n"
                "command_candidate:\n"
                "  config_yaml: configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml\n"
                "runtime_targets:\n"
                "  per_device_batch_size: 4\n"
                "  gradient_accumulation_steps: 1\n"
                "  effective_batch_size: 4\n"
                "  max_train_steps: 1000\n"
                "checkpoint_policy:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: 1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "resource_budget:\n"
                "  per_device_batch_size: bounded_full_vla_smoke_passed_4\n"
                "  gradient_accumulation_steps: bounded_full_vla_smoke_1\n"
                "  effective_batch_size: bounded_full_vla_smoke_4\n"
                "  max_train_steps: TBD_long_training\n"
                "checkpoint:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: initial_target_1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n"
                "status:\n"
                "  launch_ready: false\n"
                "  policy_confirmed: false\n",
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
        self.assertTrue(report["checks"]["candidate_has_launch_guard"])
        self.assertTrue(report["checks"]["candidate_launch_state_consistent"])
        self.assertTrue(report["checks"]["command_has_launch_guard"])
        self.assertTrue(report["checks"]["command_launch_state_matches_candidate"])
        self.assertTrue(report["checks"]["command_requires_human_confirmation"])
        self.assertTrue(report["checks"]["candidate_enables_future_supervision_loss"])
        self.assertTrue(report["checks"]["candidate_uses_future_feature_heads_source"])
        self.assertTrue(report["checks"]["candidate_batch_size_4"])
        self.assertTrue(report["checks"]["candidate_max_steps_1000"])
        self.assertTrue(report["checks"]["runtime_policy_state_matches_candidate"])
        self.assertTrue(report["checks"]["final_parameter_alignment_passed"])

    def test_e001_launch_candidate_smoke_accepts_future_feature_source_alias(self):
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
                "    enable_future_supervision_loss: true\n"
                "    layerwise_bridge_feature_source: mowa_future_feature_heads\n"
                "datasets:\n"
                "  vla_data:\n"
                "    per_device_batch_size: 4\n"
                "trainer:\n"
                "  max_train_steps: 1000\n"
                "  save_interval: 1000\n"
                "  gradient_accumulation_steps: 1\n"
                "  disable_wandb: true\n"
                "  checkpoint_format: lightweight\n"
                "  save_checkpoint_as_directory: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: false\n"
                "  requires_human_confirmation: true\n"
                "  policy_confirmed: false\n"
                "command_candidate:\n"
                "  config_yaml: configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml\n"
                "runtime_targets:\n"
                "  per_device_batch_size: 4\n"
                "  gradient_accumulation_steps: 1\n"
                "  effective_batch_size: 4\n"
                "  max_train_steps: 1000\n"
                "checkpoint_policy:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: 1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "resource_budget:\n"
                "  per_device_batch_size: bounded_full_vla_smoke_passed_4\n"
                "  gradient_accumulation_steps: bounded_full_vla_smoke_1\n"
                "  effective_batch_size: bounded_full_vla_smoke_4\n"
                "  max_train_steps: TBD_long_training\n"
                "checkpoint:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: initial_target_1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n"
                "status:\n"
                "  launch_ready: false\n"
                "  policy_confirmed: false\n",
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

        self.assertTrue(report["checks"]["candidate_uses_future_feature_heads_source"])
        self.assertEqual(report["go_no_go"], "TBD: launch candidate is wired; training remains gated")

    def test_e001_launch_candidate_smoke_accepts_launch_approved_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e001_starflow_ft0_launch_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: true\n"
                "  policy_confirmed: true\n"
                "  requires_human_confirmation: true\n"
                "  human_confirmed: true\n"
                "run_root_dir: playground/mowa_ckpt\n"
                "framework:\n"
                "  name: StarFlowVLA\n"
                "  action_model:\n"
                "    action_model_type: LayerwiseFM\n"
                "  mowa:\n"
                "    enable_future_supervision_loss: true\n"
                "    layerwise_bridge_feature_source: mowa_future_feature_heads\n"
                "datasets:\n"
                "  vla_data:\n"
                "    per_device_batch_size: 4\n"
                "trainer:\n"
                "  max_train_steps: 1000\n"
                "  save_interval: 1000\n"
                "  gradient_accumulation_steps: 1\n"
                "  disable_wandb: true\n"
                "  checkpoint_format: lightweight\n"
                "  save_checkpoint_as_directory: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_training_command_candidate.yaml").write_text(
                "launch_guard:\n"
                "  launch_ready: true\n"
                "  requires_human_confirmation: true\n"
                "  human_confirmed: true\n"
                "  policy_confirmed: true\n"
                "command_candidate:\n"
                "  config_yaml: configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml\n"
                "runtime_targets:\n"
                "  per_device_batch_size: 4\n"
                "  gradient_accumulation_steps: 1\n"
                "  effective_batch_size: 4\n"
                "  max_train_steps: 1000\n"
                "checkpoint_policy:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: 1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n",
                encoding="utf-8",
            )
            (root / "configs" / "mowa" / "mowa_e001_runtime_policy_draft.yaml").write_text(
                "resource_budget:\n"
                "  per_device_batch_size: bounded_full_vla_smoke_passed_4\n"
                "  gradient_accumulation_steps: bounded_full_vla_smoke_1\n"
                "  effective_batch_size: bounded_full_vla_smoke_4\n"
                "  max_train_steps: 1000\n"
                "checkpoint:\n"
                "  run_root_dir: playground/mowa_ckpt\n"
                "  checkpoint_format: lightweight\n"
                "  save_interval: initial_target_1000\n"
                "  save_checkpoint_as_directory: true\n"
                "logging:\n"
                "  disable_wandb: true\n"
                "  logging_frequency: 10\n"
                "status:\n"
                "  resource_policy_confirmed: true\n"
                "  policy_confirmed: true\n"
                "  launch_ready: true\n",
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

        self.assertTrue(report["launch_ready"])
        self.assertTrue(report["checks"]["candidate_launch_state_consistent"])
        self.assertTrue(report["checks"]["command_launch_state_matches_candidate"])
        self.assertTrue(report["checks"]["runtime_policy_state_matches_candidate"])
        self.assertEqual(report["go_no_go"], "TBD: launch candidate is wired and launch-approved")

    def test_e001_starflow_ft0_comparison_smoke_pins_only_mowa_delta(self):
        from tools.mowa.e001_starflow_ft0_comparison_smoke import (
            build_starflow_ft0_comparison_smoke,
        )

        report = build_starflow_ft0_comparison_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(report["checks"]["baseline_config_created"])
        self.assertTrue(report["checks"]["mowa_candidate_config_created"])
        self.assertTrue(report["checks"]["baseline_launch_gated"])
        # 2026-07-05: MoWA E-001 launch guard intentionally ungated after
        # freeze_modules fix (VLM must be frozen) and human confirmation.
        self.assertTrue(report["checks"]["mowa_launch_state_consistent"])
        self.assertTrue(report["checks"]["baseline_bridge_disabled"])
        self.assertTrue(report["checks"]["mowa_bridge_enabled"])
        self.assertTrue(report["checks"]["baseline_future_labels_disabled"])
        self.assertTrue(report["checks"]["mowa_future_labels_enabled"])
        self.assertTrue(report["checks"]["paired_invariants_match"])
        self.assertTrue(report["checks"]["paired_runtime_symmetry_passed"])
        self.assertTrue(report["checks"]["official_ft0_is_reference_only"])
        self.assertEqual(report["runtime_symmetry"]["unexpected_difference_paths"], ())
        self.assertEqual(
            report["expected_differences"]["framework.mowa.enable_layerwise_bridge_token_coupling"],
            {"baseline": False, "mowa": True},
        )

    def test_e001_starflow_ft0_runtime_symmetry_rejects_unexpected_drift(self):
        from omegaconf import OmegaConf

        from tools.mowa.e001_starflow_ft0_comparison_smoke import (
            _build_runtime_symmetry_report,
        )

        baseline = OmegaConf.create(
            {
                "run_id": "baseline",
                "launch_guard": {"reason": "baseline"},
                "framework": {
                    "mowa": {
                        "enable_layerwise_bridge_token_coupling": False,
                    }
                },
                "datasets": {
                    "vla_data": {
                        "per_device_batch_size": 4,
                        "enable_mowa_future_labels": False,
                    }
                },
            }
        )
        mowa = OmegaConf.create(
            {
                "run_id": "mowa",
                "launch_guard": {"reason": "mowa"},
                "framework": {
                    "mowa": {
                        "enable_layerwise_bridge_token_coupling": True,
                    }
                },
                "datasets": {
                    "vla_data": {
                        "per_device_batch_size": 8,
                        "enable_mowa_future_labels": True,
                    }
                },
            }
        )

        report = _build_runtime_symmetry_report(baseline, mowa)

        self.assertFalse(report["passed"])
        self.assertIn(
            "datasets.vla_data.per_device_batch_size",
            report["unexpected_difference_paths"],
        )

    def test_e001_starflow_ft0_comparison_smoke_can_execute_launch_guard(self):
        from tools.mowa.e001_starflow_ft0_comparison_smoke import (
            build_starflow_ft0_comparison_smoke,
        )

        commands = []

        def fake_runner(command, cwd):
            commands.append((command, cwd))
            if command[-1].endswith("mowa_e001_starflow_ft0_baseline_candidate.yaml"):
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="RuntimeError: Training launch blocked by launch_guard",
                )
            return SimpleNamespace(
                returncode=0,
                stdout="",
                stderr="",
            )

        report = build_starflow_ft0_comparison_smoke(
            Path("."),
            check_launch_guard=True,
            command_runner=fake_runner,
        )

        self.assertEqual(len(commands), 2)
        self.assertTrue(report["launch_guard_execution"]["checked"])
        self.assertTrue(report["checks"]["baseline_launch_guard_blocks_entrypoint"])
        self.assertFalse(report["checks"]["mowa_launch_guard_blocks_entrypoint"])
        self.assertTrue(
            report["launch_guard_execution"]["baseline"]["blocked_by_launch_guard"]
        )
        self.assertFalse(report["launch_guard_execution"]["mowa"]["blocked_by_launch_guard"])

    def test_e001_readiness_accepts_static_or_executed_starflow_comparison_smoke(self):
        from tools.mowa.e001_readiness_smoke import _starflow_ft0_comparison_smoke_passed

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "comparison.json"
            report_path.write_text(
                json.dumps(
                    {
                        "checks": {
                            "baseline_launch_gated": True,
                            "mowa_launch_state_consistent": True,
                        },
                        "launch_guard_execution": {"checked": False},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_starflow_ft0_comparison_smoke_passed(report_path))

            report_path.write_text(
                json.dumps(
                    {
                        "checks": {
                            "baseline_launch_gated": True,
                            "mowa_launch_state_consistent": True,
                        },
                        "launch_guard_execution": {
                            "checked": True,
                            "baseline": {"blocked_by_launch_guard": True},
                            "mowa": {"blocked_by_launch_guard": False},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_starflow_ft0_comparison_smoke_passed(report_path))

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
        self.assertTrue(report["checks"]["trainer_state_matches_checkpoint"])
        self.assertTrue(report["checks"]["mapping_framework_starflow"])
        self.assertTrue(report["checks"]["mapping_action_head_layerwisefm"])
        self.assertTrue(report["checks"]["model_load_executed_or_not_required"])

    def test_e006_eval_load_defaults_are_read_from_config(self):
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
            ckpt_path = "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test"
            (root / "configs" / "mowa" / "mowa_e006_eval_load_smoke.yaml").write_text(
                "checkpoint:\n"
                f"  eval_candidate_checkpoint: {ckpt_path}/checkpoints/steps_2\n"
                f"  final_model_checkpoint: {ckpt_path}/final_model\n"
                "  checkpoint_root_policy: playground/mowa_ckpt\n",
                encoding="utf-8",
            )
            (checkpoint / "trainer_state.json").write_text(json.dumps({"completed_steps": 2}), encoding="utf-8")
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

            report = build_e006_eval_load_smoke(root, execute_load=False)

        self.assertEqual(
            report["observed"]["checkpoint"],
            "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/checkpoints/steps_2",
        )
        self.assertTrue(report["checks"]["checkpoint_under_mowa_ckpt"])
        self.assertTrue(report["checks"]["final_model_dir_exists"])

    def test_e006_eval_load_accepts_non_step2_checkpoint_when_path_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_bs4_candidate"
                / "checkpoints"
                / "steps_1000"
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
                json.dumps({"completed_steps": 1000}),
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
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_bs4_candidate/checkpoints/steps_1000"
                ),
                final_model=Path(
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_bs4_candidate/final_model"
                ),
                execute_load=False,
            )

        self.assertEqual(report["observed"]["expected_completed_steps"], 1000)
        self.assertTrue(report["checks"]["trainer_state_matches_checkpoint"])

    def test_e006_eval_load_resolves_latest_complete_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs" / "mowa").mkdir(parents=True)
            for run_id, completed_steps in (
                ("MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_001000", 1),
                ("MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_002000", 2),
            ):
                checkpoint = (
                    root
                    / "playground"
                    / "mowa_ckpt"
                    / run_id
                    / "checkpoints"
                    / "steps_2"
                )
                final_model = checkpoint.parents[1] / "final_model"
                checkpoint.mkdir(parents=True)
                final_model.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"completed_steps": completed_steps}),
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
            (root / "configs" / "mowa" / "mowa_e006_eval_load_smoke.yaml").write_text(
                "checkpoint:\n"
                "  eval_candidate_checkpoint: latest_complete\n"
                "  final_model_checkpoint: latest_complete\n"
                "  checkpoint_root_policy: playground/mowa_ckpt\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_eval_load_smoke import build_e006_eval_load_smoke

            report = build_e006_eval_load_smoke(root, execute_load=False)

        self.assertEqual(
            report["observed"]["checkpoint"],
            "playground/mowa_ckpt/"
            "MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_002000/checkpoints/steps_2",
        )
        self.assertEqual(
            report["observed"]["final_model"],
            "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_002000/final_model",
        )
        self.assertTrue(report["checks"]["checkpoint_dir_exists"])
        self.assertTrue(report["checks"]["final_model_dir_exists"])

    def test_action_bridge_interface_records_starflow_candidate_values(self):
        from omegaconf import OmegaConf

        bridge_cfg = OmegaConf.load("configs/mowa/mowa_action_bridge_interface.yaml")
        launch_cfg = OmegaConf.load("configs/mowa/mowa_e001_starflow_ft0_launch_candidate.yaml")

        candidate = bridge_cfg.bridge.starflow_e001_candidate
        self.assertEqual(candidate.wam_feature_dim, launch_cfg.framework.mowa.wam_feature_dim)
        self.assertEqual(candidate.action_hidden_dim, launch_cfg.framework.mowa.action_hidden_dim)
        self.assertEqual(candidate.num_bridge_tokens, launch_cfg.framework.mowa.num_bridge_tokens)
        self.assertEqual(
            candidate.num_action_layers,
            launch_cfg.framework.qwenvl.num_vl_layers,
        )
        self.assertEqual(bridge_cfg.data_gate.production_wam_hz, "Data Gate")
        self.assertEqual(bridge_cfg.data_gate.production_window, "Data Gate")
        self.assertEqual(bridge_cfg.data_gate.e001_preflight_target_wam_hz, 5)

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

    def test_e006_checkpoint_intervention_forward_default_checkpoint_comes_from_config(self):
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
            ckpt_path = "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test"
            (root / "configs" / "mowa" / "mowa_e006_eval_load_smoke.yaml").write_text(
                "checkpoint:\n"
                f"  eval_candidate_checkpoint: {ckpt_path}/checkpoints/steps_2\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_checkpoint_intervention_forward_smoke import (
                run_or_plan_checkpoint_intervention_forward_smoke,
            )

            report = run_or_plan_checkpoint_intervention_forward_smoke(
                root,
                checkpoint=None,
                execute=False,
                batch_size=2,
            )

        self.assertEqual(
            report["checkpoint"],
            "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/checkpoints/steps_2",
        )
        self.assertTrue(report["checks"]["checkpoint_exists"])
        self.assertTrue(report["checks"]["checkpoint_under_mowa_ckpt"])

    def test_server_policy_accepts_eval_time_config_overrides(self):
        from deployment.model_server.server_policy import build_argparser

        args = build_argparser().parse_args(
            [
                "--ckpt_path",
                "playground/mowa_ckpt/run/checkpoints/steps_2",
                "--config_override",
                "framework.mowa.layerwise_bridge_token_intervention=zero",
                "--config_override",
                "framework.mowa.num_bridge_tokens=2",
            ]
        )

        self.assertEqual(
            args.config_override,
            [
                "framework.mowa.layerwise_bridge_token_intervention=zero",
                "framework.mowa.num_bridge_tokens=2",
            ],
        )

    def test_policy_wrapper_metadata_uses_override_applied_action_horizon(self):
        from types import SimpleNamespace
        from unittest import mock

        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        framework = SimpleNamespace(
            action_horizon=5,
            config=SimpleNamespace(
                framework=SimpleNamespace(
                    action_model=SimpleNamespace(action_horizon=5),
                ),
            ),
        )
        framework.to = mock.Mock(return_value=framework)
        framework.eval = mock.Mock(return_value=framework)

        with mock.patch(
            "deployment.model_server.policy_wrapper.baseframework.from_pretrained",
            return_value=framework,
        ) as from_pretrained:
            with mock.patch(
                "deployment.model_server.policy_wrapper.read_mode_config",
                return_value=(
                    {
                        "framework": {
                            "action_model": {
                                "action_horizon": 16,
                            }
                        }
                    },
                    {"key_a": {}, "key_b": {}},
                ),
            ):
                wrapper = PolicyServerWrapper(
                    "playground/mowa_ckpt/run/checkpoints/steps_2",
                    device="cpu",
                    config_overrides=["framework.action_model.action_horizon=5"],
                )

        from_pretrained.assert_called_once_with(
            "playground/mowa_ckpt/run/checkpoints/steps_2",
            config_overrides=["framework.action_model.action_horizon=5"],
        )
        self.assertEqual(wrapper.metadata["action_chunk_size"], 5)
        self.assertEqual(
            wrapper.metadata["config_overrides"],
            ["framework.action_model.action_horizon=5"],
        )

    def test_train_starvla_launch_guard_blocks_unconfirmed_launch_only(self):
        from omegaconf import OmegaConf

        from starVLA.training.train_starvla import _enforce_launch_guard

        plain_cfg = OmegaConf.create({"trainer": {"max_train_steps": 1}})
        _enforce_launch_guard(plain_cfg, full_path_dry_run_only=False)

        guarded_cfg = OmegaConf.create(
            {
                "launch_guard": {
                    "launch_ready": False,
                    "requires_human_confirmation": True,
                    "policy_confirmed": False,
                }
            }
        )
        with self.assertRaisesRegex(RuntimeError, "Training launch blocked"):
            _enforce_launch_guard(guarded_cfg, full_path_dry_run_only=False)

        _enforce_launch_guard(guarded_cfg, full_path_dry_run_only=True)

        policy_unconfirmed_cfg = OmegaConf.create(
            {
                "launch_guard": {
                    "launch_ready": True,
                    "requires_human_confirmation": True,
                    "human_confirmed": True,
                    "policy_confirmed": False,
                }
            }
        )
        with self.assertRaisesRegex(RuntimeError, "policy_confirmed=False"):
            _enforce_launch_guard(policy_unconfirmed_cfg, full_path_dry_run_only=False)

        approved_cfg = OmegaConf.create(
            {
                "launch_guard": {
                    "launch_ready": True,
                    "requires_human_confirmation": True,
                    "human_confirmed": True,
                    "policy_confirmed": True,
                }
            }
        )
        _enforce_launch_guard(approved_cfg, full_path_dry_run_only=False)

    def test_training_audit_config_touch_exports_key_fields(self):
        from omegaconf import OmegaConf

        from starVLA.training.train_starvla import _touch_training_audit_config
        from starVLA.training.trainer_utils.config_tracker import wrap_config

        cfg = wrap_config(
            OmegaConf.create(
                {
                    "trainer": {
                        "is_resume": True,
                        "pretrained_checkpoint": "playground/mowa_ckpt/run/checkpoints/steps_2",
                        "gradient_accumulation_steps": 4,
                    },
                    "datasets": {
                        "vla_data": {
                            "data_mix": "robocasa365_open_drawer_target_human",
                        }
                    },
                    "framework": {
                        "name": "StarFlowVLA",
                        "action_model": {
                            "action_model_type": "LayerwiseFM",
                            "num_target_vision_tokens": 0,
                        },
                        "mowa": {
                            "enable_layerwise_bridge_token_coupling": True,
                            "layerwise_bridge_feature_source": "mowa_future_feature_heads",
                            "layerwise_bridge_token_intervention": "baseline",
                            "num_bridge_tokens": 2,
                            "layerwise_bridge_active_heads": [
                                "task_progress",
                                "action_outcome_class",
                            ],
                        },
                    },
                }
            )
        )

        _touch_training_audit_config(cfg)
        accessed = cfg.export_accessed_config(use_original_values=False)

        self.assertTrue(accessed["trainer"]["is_resume"])
        self.assertEqual(
            accessed["trainer"]["pretrained_checkpoint"],
            "playground/mowa_ckpt/run/checkpoints/steps_2",
        )
        self.assertEqual(accessed["trainer"]["gradient_accumulation_steps"], 4)
        self.assertEqual(accessed["datasets"]["vla_data"]["data_mix"], "robocasa365_open_drawer_target_human")
        self.assertEqual(accessed["framework"]["name"], "StarFlowVLA")
        self.assertEqual(accessed["framework"]["action_model"]["action_model_type"], "LayerwiseFM")
        self.assertEqual(accessed["framework"]["action_model"]["num_target_vision_tokens"], 0)
        self.assertTrue(accessed["framework"]["mowa"]["enable_layerwise_bridge_token_coupling"])
        self.assertEqual(accessed["framework"]["mowa"]["layerwise_bridge_feature_source"], "mowa_future_feature_heads")
        self.assertEqual(accessed["framework"]["mowa"]["layerwise_bridge_token_intervention"], "baseline")
        self.assertEqual(accessed["framework"]["mowa"]["num_bridge_tokens"], 2)
        self.assertEqual(
            accessed["framework"]["mowa"]["layerwise_bridge_active_heads"],
            ["task_progress", "action_outcome_class"],
        )

    def test_train_starvla_train_step_adds_future_supervision_loss_when_enabled(self):
        from types import SimpleNamespace

        import torch

        from starVLA.training.train_starvla import VLATrainer

        class _DummyModel:
            def forward(self, batch_vla):
                return {
                    "action_loss": torch.tensor(2.0, requires_grad=True),
                    "mowa_future_supervision_loss": torch.tensor(4.0, requires_grad=True),
                }

            def parameters(self):
                return []

        class _DummyAccelerator:
            def __init__(self):
                self.sync_gradients = True
                self.backward_calls = []
                self.clip_calls = []
                self.accumulate_calls = 0

            def accumulate(self, model):
                from contextlib import nullcontext

                self.accumulate_calls += 1
                return nullcontext()

            def backward(self, loss):
                self.backward_calls.append(loss.detach().item())

            def clip_grad_norm_(self, parameters, max_norm):
                self.clip_calls.append((tuple(parameters), max_norm))

        class _DummyOptimizer:
            def __init__(self):
                self.step_calls = 0
                self.zero_grad_calls = 0

            def step(self):
                self.step_calls += 1

            def zero_grad(self):
                self.zero_grad_calls += 1

        class _DummyScheduler:
            def __init__(self):
                self.step_calls = 0

            def step(self):
                self.step_calls += 1

        trainer = object.__new__(VLATrainer)
        trainer.config = SimpleNamespace(
            trainer=SimpleNamespace(
                enable_mowa_future_supervision_loss=True,
                loss_scale=SimpleNamespace(mowa_future_supervision=0.25),
                gradient_clipping=None,
            )
        )
        trainer.model = _DummyModel()
        trainer.accelerator = _DummyAccelerator()
        trainer.optimizer = _DummyOptimizer()
        trainer.lr_scheduler = _DummyScheduler()

        metrics = trainer._train_step([{"dummy": True}])

        self.assertAlmostEqual(trainer.accelerator.backward_calls[-1], 3.0)
        self.assertEqual(metrics["action_dit_loss"], 2.0)
        self.assertEqual(metrics["mowa_future_supervision_loss"], 4.0)
        self.assertEqual(trainer.optimizer.step_calls, 1)
        self.assertEqual(trainer.optimizer.zero_grad_calls, 1)
        self.assertEqual(trainer.lr_scheduler.step_calls, 1)

    def test_e006_policy_rollout_preflight_builds_gated_commands(self):
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
            (root / "deployment" / "model_server").mkdir(parents=True)
            (root / "examples" / "Robocasa_365" / "eval_files").mkdir(parents=True)
            (root / ".venv" / "bin").mkdir(parents=True)
            (root / ".robocase" / "bin").mkdir(parents=True)
            (root / "deployment" / "model_server" / "server_policy.py").write_text("", encoding="utf-8")
            (root / "examples" / "Robocasa_365" / "eval_files" / "simulation_env.py").write_text(
                "",
                encoding="utf-8",
            )
            (root / "examples" / "Robocasa_365" / "eval_files" / "run_eval.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n",
                encoding="utf-8",
            )
            (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
            (root / ".robocase" / "bin" / "python").write_text("", encoding="utf-8")
            config = root / "configs" / "mowa" / "mowa_e006_policy_rollout_candidate.yaml"
            config.write_text(
                "\n".join(
                    [
                        "stage: M5",
                        "task_id: M5-007",
                        "experiment_id: E-006",
                        "experiment_name: test",
                        "launch_ready: false",
                        "eval_started: false",
                        "requires_human_confirmation: true",
                        "checkpoint: "
                        "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/"
                        "checkpoints/steps_2",
                        "checkpoint_root_policy: playground/mowa_ckpt",
                        "server:",
                        "  python: .venv/bin/python",
                        "  entrypoint: deployment/model_server/server_policy.py",
                        "  port_base: 5686",
                        "  use_bf16: true",
                        "  idle_timeout: 1800",
                        "client:",
                        "  python: .robocase/bin/python",
                        "  module: examples.Robocasa_365.eval_files.simulation_env",
                        "  env_name: robocasa/OpenDrawer",
                        "  n_episodes: 2",
                        "  n_envs: 2",
                        "  max_episode_steps: 100",
                        "  n_action_steps: 8",
                        "  video_out_path: playground/eval_results/mowa_e006_robocasa365_open_drawer_smoke/videos",
                        "interventions:",
                        "  - baseline",
                        "  - zero",
                        "  - batch_shuffle",
                        "  - head_mask_control",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_policy_rollout_preflight_smoke import (
                build_e006_policy_rollout_preflight_smoke,
            )

            report = build_e006_policy_rollout_preflight_smoke(
                root,
                Path("configs/mowa/mowa_e006_policy_rollout_candidate.yaml"),
            )

        self.assertFalse(report["eval_started"])
        self.assertFalse(report["launch_ready"])
        self.assertTrue(report["checks"]["checkpoint_under_mowa_ckpt"])
        self.assertTrue(report["checks"]["client_batch_size_allows_shuffle"])
        self.assertTrue(report["checks"]["client_episode_count_covers_vector_envs"])
        self.assertTrue(report["checks"]["all_interventions_have_commands"])
        self.assertTrue(report["checks"]["non_baseline_commands_use_config_override"])
        self.assertTrue(report["checks"]["baseline_command_pins_baseline_override"])
        self.assertIn("--config_override", report["commands"]["zero"]["server_command"])
        self.assertIn(
            "framework.mowa.layerwise_bridge_token_intervention=zero",
            report["commands"]["zero"]["server_command"],
        )

    def test_e006_policy_rollout_preflight_resolves_latest_complete_checkpoint_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_003000"
                / "checkpoints"
                / "steps_2"
            )
            checkpoint.mkdir(parents=True)
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "deployment" / "model_server").mkdir(parents=True)
            (root / "examples" / "Robocasa_365" / "eval_files").mkdir(parents=True)
            (root / ".venv" / "bin").mkdir(parents=True)
            (root / ".robocase" / "bin").mkdir(parents=True)
            (root / "deployment" / "model_server" / "server_policy.py").write_text(
                "",
                encoding="utf-8",
            )
            (root / "examples" / "Robocasa_365" / "eval_files" / "simulation_env.py").write_text(
                "",
                encoding="utf-8",
            )
            (root / "examples" / "Robocasa_365" / "eval_files" / "run_eval.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n",
                encoding="utf-8",
            )
            (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
            (root / ".robocase" / "bin" / "python").write_text("", encoding="utf-8")
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
            config = root / "configs" / "mowa" / "mowa_e006_policy_rollout_candidate.yaml"
            config.write_text(
                "\n".join(
                    [
                        "stage: M5",
                        "task_id: M5-007",
                        "experiment_id: E-006",
                        "experiment_name: test",
                        "launch_ready: false",
                        "eval_started: false",
                        "requires_human_confirmation: true",
                        "checkpoint: latest_complete",
                        "checkpoint_root_policy: playground/mowa_ckpt",
                        "server:",
                        "  python: .venv/bin/python",
                        "  entrypoint: deployment/model_server/server_policy.py",
                        "  port_base: 5686",
                        "  use_bf16: true",
                        "  idle_timeout: 1800",
                        "client:",
                        "  python: .robocase/bin/python",
                        "  module: examples.Robocasa_365.eval_files.simulation_env",
                        "  env_name: robocasa/OpenDrawer",
                        "  n_episodes: 2",
                        "  n_envs: 2",
                        "  max_episode_steps: 100",
                        "  n_action_steps: 8",
                        "  video_out_path: playground/eval_results/mowa/videos",
                        "interventions:",
                        "  - baseline",
                        "  - zero",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_policy_rollout_preflight_smoke import (
                build_e006_policy_rollout_preflight_smoke,
            )

            report = build_e006_policy_rollout_preflight_smoke(
                root,
                Path("configs/mowa/mowa_e006_policy_rollout_candidate.yaml"),
            )

        resolved = (
            "playground/mowa_ckpt/"
            "MoWA-E-001_starflow_ft0_save_resume_smoke_20260704_003000/checkpoints/steps_2"
        )
        self.assertTrue(report["checks"]["checkpoint_exists"])
        self.assertIn(resolved, report["commands"]["baseline"]["server_command"])
        self.assertIn(resolved, report["commands"]["baseline"]["client_command"])

    def test_e006_policy_rollout_smoke_plan_does_not_execute(self):
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
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa" / "mowa_e006_policy_rollout_preflight_smoke.json").write_text(
                json.dumps({"checks": {"ok": True}}),
                encoding="utf-8",
            )
            config = root / "configs" / "mowa" / "mowa_e006_policy_rollout_candidate.yaml"
            config.write_text(
                "\n".join(
                    [
                        "stage: M5",
                        "task_id: M5-007",
                        "experiment_id: E-006",
                        "experiment_name: test",
                        "launch_ready: false",
                        "eval_started: false",
                        "requires_human_confirmation: true",
                        "checkpoint: "
                        "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_save_resume_smoke_test/"
                        "checkpoints/steps_2",
                        "checkpoint_root_policy: playground/mowa_ckpt",
                        "server:",
                        "  python: .venv/bin/python",
                        "  entrypoint: deployment/model_server/server_policy.py",
                        "  port_base: 5686",
                        "  use_bf16: true",
                        "  idle_timeout: 1800",
                        "client:",
                        "  python: .robocase/bin/python",
                        "  module: examples.Robocasa_365.eval_files.simulation_env",
                        "  env_name: robocasa/OpenDrawer",
                        "  n_episodes: 2",
                        "  n_envs: 2",
                        "  max_episode_steps: 100",
                        "  n_action_steps: 8",
                        "  video_out_path: playground/eval_results/mowa_e006_robocasa365_open_drawer_smoke/videos",
                        "interventions:",
                        "  - baseline",
                        "  - zero",
                        "  - batch_shuffle",
                        "  - head_mask_control",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_policy_rollout_smoke import run_or_plan_e006_policy_rollout_smoke

            report = run_or_plan_e006_policy_rollout_smoke(
                root,
                config_path=Path("configs/mowa/mowa_e006_policy_rollout_candidate.yaml"),
                execute=False,
                server_ready_timeout=1,
            )

        self.assertFalse(report["eval_started"])
        self.assertTrue(report["checks"]["preflight_report_exists"])
        self.assertTrue(report["checks"]["batch_shuffle_batch_size_gt_1"])
        self.assertTrue(report["checks"]["executed_when_requested"])
        self.assertEqual(
            {run["intervention"] for run in report["runs"]},
            {"baseline", "zero", "batch_shuffle", "head_mask_control"},
        )
        self.assertTrue(all(run["executed"] is False for run in report["runs"]))

    def test_e006_rollout_assets_blocker_is_detected_from_failed_runs(self):
        from tools.mowa.e006_policy_rollout_smoke import _extract_rollout_blocker

        runs = [
            {
                "intervention": "baseline",
                "client_result": {
                    "failure_category": "missing_robocasa_asset",
                    "tail": [
                        "FileNotFoundError: [Errno 2] No such file or directory: "
                        "'/tmp/robocasa/models/assets/fixtures/sinks/Sink025/model.xml'"
                    ],
                },
            },
            {
                "intervention": "zero",
                "client_result": {
                    "failure_category": "missing_robocasa_asset",
                    "tail": [
                        "FileNotFoundError: [Errno 2] No such file or directory: "
                        "'/tmp/robocasa/models/assets/objects/lightwheel/utensil_rack/"
                        "UtensilRack007/model.xml'"
                    ],
                },
            },
        ]

        blocker = _extract_rollout_blocker(runs)

        self.assertEqual(blocker["status"], "missing_robocasa_assets")
        self.assertEqual(blocker["scope"], "environment")
        self.assertEqual(blocker["blocked_interventions"], ["baseline", "zero"])
        self.assertEqual(
            blocker["missing_asset_paths"],
            [
                "/tmp/robocasa/models/assets/fixtures/sinks/Sink025/model.xml",
                "/tmp/robocasa/models/assets/objects/lightwheel/utensil_rack/UtensilRack007/model.xml",
            ],
        )

    def test_e006_rollout_render_backend_blocker_is_detected_from_failed_runs(self):
        from tools.mowa.e006_policy_rollout_smoke import _extract_rollout_blocker

        runs = [
            {
                "intervention": "baseline",
                "client_result": {
                    "failure_category": "robocasa_render_backend_unavailable",
                    "tail": [
                        "AttributeError: 'NoneType' object has no attribute 'glGetError'",
                    ],
                },
            }
        ]

        blocker = _extract_rollout_blocker(runs)

        self.assertEqual(blocker["status"], "robocasa_render_backend_unavailable")
        self.assertEqual(blocker["scope"], "environment")
        self.assertEqual(blocker["blocked_interventions"], ["baseline"])
        self.assertEqual(
            blocker["backend_signatures"],
            ["AttributeError: 'NoneType' object has no attribute 'glGetError'"],
        )

    def test_e006_rollout_server_blocker_is_detected_from_failed_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            server_log = root / "server.log"
            server_log.write_text(
                "\n".join(
                    [
                        "RuntimeError: Error(s) in loading state_dict for StarFlowVLA:",
                        "Missing key(s) in state_dict: "
                        "['mowa_layerwise_bridge_future_feature_heads.trunk.0.weight', "
                        "'mowa_layerwise_bridge_head_mask_projector.weight']",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            from tools.mowa.e006_policy_rollout_smoke import _extract_rollout_blocker

            blocker = _extract_rollout_blocker(
                [
                    {
                        "intervention": "baseline",
                        "server_failure_category": "checkpoint_model_incompatible",
                        "server_log": str(server_log),
                    }
                ]
            )

        self.assertEqual(blocker["status"], "checkpoint_model_incompatible")
        self.assertEqual(blocker["scope"], "checkpoint")
        self.assertEqual(blocker["blocked_interventions"], ["baseline"])
        self.assertIn(
            "mowa_layerwise_bridge_future_feature_heads.trunk.0.weight",
            blocker["missing_state_keys"],
        )

    def test_e001_readiness_accepts_e006_rollout_blocker_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            report = root / "docs_zh" / "mowa" / "mowa_e006_policy_rollout_smoke.json"
            report.write_text(
                json.dumps(
                    {
                        "rollout_blocker": {
                            "status": "missing_robocasa_assets",
                            "scope": "environment",
                            "blocked_interventions": ["baseline", "zero"],
                            "missing_asset_paths": [
                                "/tmp/robocasa/models/assets/fixtures/sinks/Sink025/model.xml"
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e001_readiness_smoke import _e006_rollout_outcome_recorded

            self.assertTrue(_e006_rollout_outcome_recorded(report))

    def test_e001_readiness_accepts_e006_rollout_success_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            report = root / "docs_zh" / "mowa" / "mowa_e006_policy_rollout_smoke.json"
            report.write_text(
                json.dumps(
                    {
                        "eval_started": False,
                        "checks": {
                            "server_started_when_executed": True,
                            "client_succeeded_when_executed": True,
                            "result_json_collected_when_executed": True,
                            "success_rate_recorded_when_executed": True,
                        },
                        "rollout_blocker": {},
                    }
                ),
                encoding="utf-8",
            )

            from tools.mowa.e001_readiness_smoke import _e006_rollout_outcome_recorded

            self.assertTrue(_e006_rollout_outcome_recorded(report))

    def test_e006_rollout_go_no_go_accepts_success_without_blocker(self):
        from tools.mowa.e006_policy_rollout_smoke import _rollout_go_no_go

        result = _rollout_go_no_go(
            execute=True,
            checks={
                "preflight_report_exists": True,
                "checkpoint_exists": True,
                "batch_shuffle_batch_size_gt_1": True,
                "executed_when_requested": True,
                "server_started_when_executed": True,
                "client_succeeded_when_executed": True,
                "result_json_collected_when_executed": True,
                "success_rate_recorded_when_executed": True,
                "structured_rollout_outcome_recorded_when_executed": True,
            },
        )

        self.assertEqual(
            result,
            "TBD: E-006 rollout smoke executed; review success deltas before any claim",
        )

    def test_latest_complete_checkpoint_prefers_completed_steps_then_run_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "playground" / "mowa_ckpt"
            for run_id, completed_steps in (
                ("z_backup", 1),
                ("a_real_run", 3),
            ):
                checkpoint = base / run_id / "checkpoints" / "steps_2"
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"completed_steps": completed_steps}),
                    encoding="utf-8",
                )
                (checkpoint / "starflow_mapping.json").write_text("{}", encoding="utf-8")
                for name in (
                    "model.safetensors.index.json",
                    "config.full.yaml",
                    "dataset_statistics.json",
                    "scheduler.pt",
                    "random_states_0.pkl",
                    "optimizer_rank_00000.pt",
                ):
                    (checkpoint / name).write_bytes(b"x")
                (checkpoint / "model-00001.safetensors").write_bytes(b"x")

            from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference

            resolved = resolve_mowa_checkpoint_reference(
                root,
                "latest_complete",
            )

        self.assertEqual(
            resolved,
            Path("playground/mowa_ckpt/a_real_run/checkpoints/steps_2"),
        )

    def test_latest_complete_checkpoint_accepts_zero_optimizer_state_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "playground" / "mowa_ckpt" / "run" / "checkpoints" / "steps_2"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 2}),
                encoding="utf-8",
            )
            (checkpoint / "starflow_mapping.json").write_text("{}", encoding="utf-8")
            for name in (
                "model.safetensors.index.json",
                "config.full.yaml",
                "dataset_statistics.json",
                "scheduler.pt",
                "random_states_0.pkl",
            ):
                (checkpoint / name).write_bytes(b"x")
            (checkpoint / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"x")
            (checkpoint / "model-00001.safetensors").write_bytes(b"x")

            from tools.mowa.mowa_checkpoint_resolver import resolve_mowa_checkpoint_reference

            resolved = resolve_mowa_checkpoint_reference(root, "latest_complete")

        self.assertEqual(resolved, Path("playground/mowa_ckpt/run/checkpoints/steps_2"))

    def test_robocasa365_eval_client_defaults_to_egl_when_backend_unset(self):
        source = Path("examples/Robocasa_365/eval_files/simulation_env.py").read_text(encoding="utf-8")

        self.assertIn('os.environ.setdefault("MUJOCO_GL", "egl")', source)
        self.assertIn('os.environ.setdefault("PYOPENGL_PLATFORM", "egl")', source)

    def test_robocasa365_eval_client_has_argparse_fallback_without_tyro(self):
        source = Path("examples/Robocasa_365/eval_files/simulation_env.py").read_text(encoding="utf-8")

        self.assertNotIn("\nimport tyro\n", source)
        self.assertIn("def _parse_args_without_tyro", source)
        self.assertIn("--args.pretrained-path", source)
        self.assertIn("except ModuleNotFoundError as exc", source)

    def test_qwenoft_mowa_future_supervision_probe_requires_explicit_labels(self):
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
                        "enable_future_supervision_probe": True,
                        "future_supervision_hidden_dim": 6,
                        "future_supervision_active_heads": [
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
        probe_owner._setup_mowa_future_supervision_probe()

        action_queries = torch.randn(4, 5, 8)
        missing = probe_owner._maybe_run_mowa_future_supervision_probe(action_queries, [{} for _ in range(4)])
        self.assertIsNotNone(missing)
        self.assertFalse(missing["supervision_available"])
        self.assertIsNone(missing["loss"])
        self.assertEqual(missing["active_heads"], ())

        examples = [
            {
                "mowa_future_targets": {
                    "task_progress": 0.25,
                    "action_outcome_class": [1.0, 0.0],
                },
                "mowa_future_masks": {
                    "task_progress": True,
                    "action_outcome_class": True,
                },
            }
            for _ in range(4)
        ]
        supervised = probe_owner._maybe_run_mowa_future_supervision_probe(action_queries, examples)

        self.assertTrue(supervised["supervision_available"])
        self.assertEqual(
            supervised["active_heads"],
            ("task_progress", "action_outcome_class"),
        )
        self.assertIn("task_progress", supervised["losses"])
        self.assertIn("action_outcome_class", supervised["losses"])
        self.assertTrue(torch.isfinite(supervised["loss"]))

    def test_qwenoft_mowa_future_supervision_aliases_match_legacy_probe(self):
        try:
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
                        "enable_future_supervision_probe": True,
                        "future_supervision_hidden_dim": 6,
                        "future_supervision_active_heads": [
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
        probe_owner._setup_mowa_future_supervision_probe()

        self.assertIsNotNone(probe_owner.mowa_future_supervision_probe)
        self.assertEqual(
            probe_owner.mowa_future_supervision_active_heads,
            ("task_progress", "action_outcome_class"),
        )
        self.assertEqual(probe_owner.mowa_future_supervision_probe.config.hidden_dim, 6)

    def test_qwenoft_mowa_future_supervision_accepts_action_outcome_loss_type(self):
        try:
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
                        "enable_future_supervision_probe": True,
                        "future_supervision_hidden_dim": 6,
                        "future_supervision_action_outcome_loss_type": "cross_entropy_done",
                    },
                }
            }
        )
        probe_owner = object.__new__(Qwenvl_OFT)
        nn.Module.__init__(probe_owner)
        probe_owner.config = cfg
        probe_owner._setup_mowa_future_supervision_probe()

        self.assertEqual(
            probe_owner.mowa_future_supervision_probe.config.action_outcome_loss_type,
            "cross_entropy_done",
        )

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

    def test_qwenoft_mowa_bridge_probe_infers_mlp_layer_count(self):
        try:
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
                        "num_bridge_tokens": 2,
                    },
                }
            }
        )
        probe_owner = object.__new__(Qwenvl_OFT)
        nn.Module.__init__(probe_owner)
        probe_owner.config = cfg
        probe_owner.action_model = nn.Module()
        probe_owner.action_model.model = nn.Module()
        probe_owner.action_model.model.mlp_resnet_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        probe_owner._setup_mowa_action_bridge_probe()

        self.assertEqual(probe_owner.mowa_action_bridge_probe.config.num_action_layers, 2)

    def test_future_latent_prior_predicts_future_latent_from_current_plus_text_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import (
            MoWAFutureLatentPrior,
            MoWAFutureLatentPriorConfig,
        )

        model = MoWAFutureLatentPrior(
            MoWAFutureLatentPriorConfig(
                current_latent_dim=4,
                text_hidden_dim=6,
                hidden_dim=8,
                future_latent_dim=5,
            )
        )
        current = torch.randn(2, 4)
        text = torch.randn(2, 6)
        target = torch.randn(2, 5)

        loss, losses, output = model.compute_loss(current, text, target)

        self.assertEqual(tuple(output.predicted_future_latent.shape), (2, 5))
        self.assertFalse(output.history_latent_used)
        self.assertTrue(output.future_latent_target_required)
        self.assertIn("future_latent_mse", losses)
        self.assertTrue(torch.isfinite(loss))

    def test_future_latent_prior_rejects_history_latent_input(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import MoWAFutureLatentPrior

        model = MoWAFutureLatentPrior()
        current = torch.randn(2, 1024)
        text = torch.randn(2, 2048)
        history = torch.randn(2, 1024)

        with self.assertRaisesRegex(ValueError, "does not accept history_latent"):
            model.forward(current, text, history_latent=history)

    def test_future_latent_prior_interface_smoke_passes(self):
        from tools.mowa.future_latent_prior_interface_smoke import (
            build_future_latent_prior_interface_smoke,
        )

        report = build_future_latent_prior_interface_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: future latent prior interface smoke passed; "
            "latent cache builder remains gated",
        )

    def test_hlc_gci_interface_shapes_and_gate_range(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not available")

        from starVLA.model.modules.mowa import MoWAHLCGCI, MoWAHLCGCIConfig

        model = MoWAHLCGCI(
            MoWAHLCGCIConfig(
                history_latent_dim=6,
                condition_hidden_dim=8,
                history_steps=3,
                compressed_history_dim=4,
                gate_hidden_dim=5,
            )
        )
        history_latent = torch.randn(2, 3, 6)
        condition_tokens = torch.randn(2, 4, 8)

        output = model(history_latent, condition_tokens)

        self.assertEqual(tuple(output.compressed_history.shape), (2, 4))
        self.assertEqual(tuple(output.gate_values.shape), (2, 8))
        self.assertEqual(tuple(output.gated_condition_tokens.shape), (2, 4, 8))
        self.assertGreaterEqual(float(output.gate_values.min().item()), 0.0)
        self.assertLessEqual(float(output.gate_values.max().item()), 1.0)

    def test_hlc_gci_interface_smoke_passes(self):
        from tools.mowa.hlc_gci_interface_smoke import (
            build_hlc_gci_interface_smoke,
        )

        report = build_hlc_gci_interface_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: HLC-GCI interface smoke passed; framework integration remains gated",
        )

    def test_future_gated_heads_aliases_preserve_legacy_compatibility(self):
        from starVLA.model.modules.mowa import (
            MoWAFutureGatedHeads,
            MoWAFutureGatedHeadsConfig,
            MoWAGatedHeads,
            MoWAGatedHeadsConfig,
        )

        self.assertIs(MoWAFutureGatedHeads, MoWAGatedHeads)
        self.assertIs(MoWAFutureGatedHeadsConfig, MoWAGatedHeadsConfig)
        self.assertIs(MoWAGatedHeads, MoWAGatedHeads)
        self.assertIs(MoWAGatedHeadsConfig, MoWAGatedHeadsConfig)

    def test_future_gated_heads_use_interpretable_init_gate_value(self):
        from starVLA.model.modules.mowa import (
            MoWAFutureFullHeadsConfig,
            MoWAFutureGatedHeads,
            MoWAFutureGatedHeadsConfig,
        )

        model = MoWAFutureGatedHeads(
            MoWAFutureGatedHeadsConfig(
                heads_config=MoWAFutureFullHeadsConfig(input_dim=8, hidden_dim=4),
                init_gate_value=0.5,
            )
        )
        gate_values = model.gate_values()
        gate_summary = model.gate_summary(step=7)

        self.assertTrue(all(abs(value - 0.5) < 1e-5 for value in gate_values.values()))
        self.assertEqual(gate_summary["comparison_scope"], "single_fullheads_control_only")
        self.assertFalse(gate_summary["allow_per_head_sweep"])
        self.assertEqual(gate_summary["step"], 7)

    def test_future_gated_heads_reject_invalid_scope_or_gate_value(self):
        from starVLA.model.modules.mowa import (
            MoWAFutureFullHeadsConfig,
            MoWAFutureGatedHeadsConfig,
        )

        with self.assertRaisesRegex(ValueError, "init_gate_value"):
            MoWAFutureGatedHeadsConfig(
                heads_config=MoWAFutureFullHeadsConfig(input_dim=8, hidden_dim=4),
                init_gate_value=1.2,
            )
        with self.assertRaisesRegex(ValueError, "comparison_scope"):
            MoWAFutureGatedHeadsConfig(
                heads_config=MoWAFutureFullHeadsConfig(input_dim=8, hidden_dim=4),
                comparison_scope="per_head_sweep",
            )
        with self.assertRaisesRegex(ValueError, "must not enable per-head sweep"):
            MoWAFutureGatedHeadsConfig(
                heads_config=MoWAFutureFullHeadsConfig(input_dim=8, hidden_dim=4),
                allow_per_head_sweep=True,
            )

    def test_future_gated_heads_interface_smoke_passes(self):
        from tools.mowa.future_gated_heads_interface_smoke import _build_smoke

        report = _build_smoke()

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["observed"]["gate_summary"]["comparison_scope"], "single_fullheads_control_only")
        self.assertFalse(report["observed"]["gate_summary"]["allow_per_head_sweep"])
        self.assertEqual(
            report["go_no_go"],
            "TBD: gated-heads interface smoke passed",
        )

    def test_e002_future_gated_heads_comparison_smoke_passes(self):
        from tools.mowa.e002_future_gated_heads_comparison_smoke import (
            build_e002_future_gated_heads_comparison_smoke,
        )

        report = build_e002_future_gated_heads_comparison_smoke(Path("."))

        self.assertFalse(report["training_started"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["go_no_go"],
            "TBD: E-002 single FullHeads comparison entry is wired; training remains gated",
        )
        self.assertIn(
            "runtime integration exists but training remains gated",
            report["unresolved_items"][0],
        )

    def test_experiment_launch_readiness_matrix_reports_suite_not_ready(self):
        from tools.mowa.experiment_launch_readiness_matrix import (
            build_experiment_launch_readiness_matrix,
        )

        report = build_experiment_launch_readiness_matrix(Path("."))

        self.assertFalse(report["all_training_experiments_ready"])
        self.assertIn("E-001", report["not_ready_training_experiments"])
        self.assertIn("E-002", report["not_ready_training_experiments"])
        entries = {entry["experiment_id"]: entry for entry in report["entries"]}
        self.assertEqual(entries["E-001"]["status"], "bounded_executable_but_full_launch_blocked")
        self.assertEqual(entries["E-002"]["status"], "runtime_integrated_but_training_gated")
        self.assertEqual(entries["E-006"]["status"], "rollout_executed_with_noninformative_checkpoint")

    def test_share_tools_strict_mismatch_accepts_legacy_mowa_bridge_key_alias(self):
        from starVLA.model.framework.share_tools import _filter_strict_key_mismatches

        missing_keys, unexpected_keys = _filter_strict_key_mismatches(
            {"mowa_layerwise_bridge_future_feature_heads.trunk.0.weight"},
            {"mowa_layerwise_bridge_future_heads.trunk.0.weight"},
        )

        self.assertEqual(missing_keys, [])
        self.assertEqual(unexpected_keys, [])

    def test_qwenpi_rewrites_legacy_mowa_checkpoint_keys_for_compatibility(self):
        from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3

        state_dict = {
            "mowa_layerwise_bridge_future_heads.trunk.0.weight": "legacy_weight",
            "mowa_layerwise_bridge_future_heads.trunk.0.bias": "legacy_bias",
        }

        Qwen_PI_v3._rewrite_mowa_checkpoint_state_dict_keys_for_compatibility(state_dict)

        self.assertNotIn("mowa_layerwise_bridge_future_heads.trunk.0.weight", state_dict)
        self.assertEqual(
            state_dict["mowa_layerwise_bridge_future_feature_heads.trunk.0.weight"],
            "legacy_weight",
        )
        self.assertEqual(
            state_dict["mowa_layerwise_bridge_future_feature_heads.trunk.0.bias"],
            "legacy_bias",
        )

    def test_steps_1000_offline_diagnostic_detects_wired_but_zero_gain_case(self):
        from tools.mowa.e001_steps_1000_offline_diagnostic import _diagnose_checkpoint_behavior

        forward_report = {
            "action_loss_delta_vs_baseline": {
                "baseline": 0.0,
                "zero": 0.01,
                "batch_shuffle": -0.02,
                "head_mask_control": 0.015,
            }
        }
        rollout_report = {
            "runs": [
                {"intervention": "baseline", "rollout_result": {"success_rate": 0.0}},
                {"intervention": "zero", "rollout_result": {"success_rate": 0.0}},
            ]
        }

        diagnosis = _diagnose_checkpoint_behavior(forward_report, rollout_report)

        self.assertEqual(diagnosis["forward_sensitivity"], "observable_effect")
        self.assertEqual(diagnosis["max_rollout_success_rate"], 0.0)
        self.assertEqual(
            diagnosis["verdict"],
            "bridge_is_wired_but_checkpoint_has_no_action_gain",
        )

    def test_steps_1000_offline_diagnostic_detects_weak_bridge_case(self):
        from tools.mowa.e001_steps_1000_offline_diagnostic import _diagnose_checkpoint_behavior

        forward_report = {
            "action_loss_delta_vs_baseline": {
                "baseline": 0.0,
                "zero": 1e-6,
                "batch_shuffle": -2e-6,
                "head_mask_control": 3e-6,
            }
        }
        rollout_report = {
            "runs": [
                {"intervention": "baseline", "rollout_result": {"success_rate": 0.0}},
                {"intervention": "zero", "rollout_result": {"success_rate": 0.0}},
            ]
        }

        diagnosis = _diagnose_checkpoint_behavior(forward_report, rollout_report)

        self.assertEqual(diagnosis["forward_sensitivity"], "no_observable_effect")
        self.assertEqual(
            diagnosis["verdict"],
            "checkpoint_is_undertrained_and_bridge_effect_is_weak",
        )

    def test_steps_1000_offline_diagnostic_reuses_existing_forward_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_bs4_candidate"
                / "checkpoints"
                / "steps_1000"
            )
            final_model = checkpoint.parents[1] / "final_model"
            (root / "configs" / "mowa").mkdir(parents=True)
            (root / "docs_zh" / "mowa").mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            final_model.mkdir(parents=True)
            baseline_checkpoint = (
                root
                / "playground"
                / "mowa_ckpt"
                / "MoWA-E-001_starflow_ft0_baseline_bs4_candidate"
                / "checkpoints"
                / "steps_1000"
            )
            baseline_final_model = baseline_checkpoint.parents[1] / "final_model"
            baseline_checkpoint.mkdir(parents=True)
            baseline_final_model.mkdir(parents=True)
            (root / "configs" / "mowa" / "mowa_e006_eval_load_smoke.yaml").write_text(
                "checkpoint:\n  checkpoint_root_policy: playground/mowa_ckpt\n",
                encoding="utf-8",
            )
            for name in (
                "model.safetensors.index.json",
                "config.full.yaml",
                "dataset_statistics.json",
                "optimizer_rank_00000.pt",
                "scheduler.pt",
                "random_states_0.pkl",
                "starflow_mapping.json",
            ):
                (checkpoint / name).write_text("{}", encoding="utf-8")
                (baseline_checkpoint / name).write_text("{}", encoding="utf-8")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 1000}),
                encoding="utf-8",
            )
            (baseline_checkpoint / "trainer_state.json").write_text(
                json.dumps({"completed_steps": 1000}),
                encoding="utf-8",
            )
            (checkpoint / "model-00001.safetensors").write_bytes(b"placeholder")
            (baseline_checkpoint / "model-00001.safetensors").write_bytes(b"placeholder")
            (root / "docs_zh" / "mowa" / "mowa_e006_checkpoint_intervention_forward_smoke.json").write_text(
                json.dumps(
                    {
                        "checkpoint": (
                            "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_bs4_candidate"
                            "/checkpoints/steps_1000"
                        ),
                        "checks": {"checkpoint_exists": True},
                        "interventions": [
                            "baseline",
                            "zero",
                            "batch_shuffle",
                            "head_mask_control",
                        ],
                        "action_loss_delta_vs_baseline": {
                            "baseline": 0.0,
                            "zero": 0.001,
                            "batch_shuffle": 0.02,
                            "head_mask_control": 0.03,
                        },
                        "runs": [
                            {"intervention": "baseline", "forward": {"action_loss": 0.1}},
                            {"intervention": "zero", "forward": {"action_loss": 0.101}},
                            {"intervention": "batch_shuffle", "forward": {"action_loss": 0.12}},
                            {"intervention": "head_mask_control", "forward": {"action_loss": 0.13}},
                        ],
                        "go_no_go": "TBD",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (root / "docs_zh" / "mowa" / "mowa_e006_policy_rollout_smoke.json").write_text(
                json.dumps(
                    {
                        "eval_started": True,
                        "runs": [
                            {
                                "intervention": "baseline",
                                "rollout_result": {"success_rate": 0.0},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            from tools.mowa.e001_steps_1000_offline_diagnostic import (
                build_e001_steps_1000_offline_diagnostic,
            )

            report = build_e001_steps_1000_offline_diagnostic(
                root,
                checkpoint=Path(
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_bs4_candidate/checkpoints/steps_1000"
                ),
                baseline_checkpoint=Path(
                    "playground/mowa_ckpt/MoWA-E-001_starflow_ft0_baseline_bs4_candidate/checkpoints/steps_1000"
                ),
                batch_size=2,
                execute_forward_smoke=False,
                execute_eval_load=False,
            )

        self.assertEqual(report["diagnosis"]["forward_sensitivity"], "observable_effect")
        self.assertEqual(
            report["diagnosis"]["verdict"],
            "bridge_is_wired_but_checkpoint_has_no_action_gain",
        )
        self.assertEqual(
            report["forward"]["action_loss_delta_vs_baseline"]["head_mask_control"],
            0.03,
        )
