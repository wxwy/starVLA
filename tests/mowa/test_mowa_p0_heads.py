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
