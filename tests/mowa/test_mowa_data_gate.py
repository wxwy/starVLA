import unittest

from starVLA.dataloader.mowa import (
    DATA_GATE,
    MOWA_PRIMARY_CANDIDATE,
    MoWAEpisodeToWindowSampler,
    MoWAUnifiedEpisode,
    MoWAWindowConfig,
    MoWAWindowSample,
    build_mowa_g0_report_skeleton,
)


class MoWADataGateTest(unittest.TestCase):
    def _episode(self) -> MoWAUnifiedEpisode:
        return MoWAUnifiedEpisode(
            episode_id="ep_001",
            dataset_source="robocasa365",
            split="train",
            instruction="open the drawer",
            timestamps=[0, 1, 2, 3, 4, 5],
            observations={"rgb_front": "Data Gate", "robot_state": "Data Gate"},
            actions={"canonical_action": "Data Gate"},
            wam_targets={"p0_labels": "Data Gate", "future_wan_latent": "Data Gate"},
            metadata={"obs_fps": DATA_GATE, "action_hz": DATA_GATE},
        )

    def test_unified_episode_rejects_non_monotonic_timestamps(self):
        episode = MoWAUnifiedEpisode(
            episode_id="ep_bad",
            dataset_source="robocasa365",
            split="train",
            instruction="open the drawer",
            timestamps=[0, 2, 1],
        )

        with self.assertRaisesRegex(ValueError, "timestamps must be monotonic"):
            episode.validate()

    def test_episode_to_window_sampler_keeps_future_targets_out_of_inputs(self):
        sampler = MoWAEpisodeToWindowSampler(
            MoWAWindowConfig(history_steps=3, future_steps=2, action_chunk_steps=2)
        )
        sample = sampler.sample(self._episode(), anchor_index=3)

        self.assertEqual(sample.history_indices, (1, 2, 3))
        self.assertEqual(sample.future_indices, (4, 5))
        self.assertEqual(sample.action_target_indices, (3, 4))
        self.assertIn("history_actions", sample.inputs)
        self.assertIn("action_chunk_target", sample.targets)
        self.assertNotIn("action_chunk_target", sample.inputs)
        sample.validate()

    def test_window_sample_blocks_future_action_leakage(self):
        sample = MoWAWindowSample(
            episode_id="ep_001",
            dataset_source="robocasa365",
            anchor_index=1,
            history_indices=(0, 1),
            current_index=1,
            future_indices=(2,),
            action_target_indices=(1, 2),
            inputs={"future_action_label": "must not enter WAM"},
            targets={},
        )

        with self.assertRaisesRegex(ValueError, "forbidden future keys"):
            sample.validate()

    def test_g0_report_skeleton_marks_robocasa365_as_primary_without_measured_values(self):
        report = build_mowa_g0_report_skeleton()
        payload = report.to_dict()

        self.assertEqual(payload["stage"], "G0")
        self.assertEqual(payload["experiment_budget"], "not_counted")
        self.assertEqual(payload["candidates"][0]["dataset"], MOWA_PRIMARY_CANDIDATE)
        self.assertEqual(payload["candidates"][0]["role"], "PrimaryCandidate")
        self.assertEqual(payload["candidates"][0]["temporal_profile_status"], DATA_GATE)
        self.assertNotIn("measured", str(payload).lower())


if __name__ == "__main__":
    unittest.main()
