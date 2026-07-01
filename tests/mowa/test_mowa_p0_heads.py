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
