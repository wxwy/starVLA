"""Unit tests for E-003/E-004 window-latent training smokes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    build_mowa_window_manifest,
)
from tools.mowa.e003_window_latent_train_smoke import (
    run_e003_window_latent_train_smoke,
)
from tools.mowa.e004_window_latent_train_smoke import (
    run_e004_window_latent_train_smoke,
)


class WindowLatentTrainSmokeTest(unittest.TestCase):
    def _write_minimal_episode(
        self,
        dataset_path: Path,
        *,
        episode_index: int = 0,
        length: int = 16,
        state_dim: int = 16,
        action_dim: int = 12,
        video_key: str = "observation.images.robot0_agentview_left",
        instruction: str = "pick the red object",
    ) -> None:
        data_dir = dataset_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)

        parquet = pa.table(
            {
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
                "timestamp": pa.array([0.05 * idx for idx in range(length)], type=pa.float32()),
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * state_dim)]),
                    state_dim,
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * action_dim)]),
                    action_dim,
                ),
                "next.reward": pa.array([0.0] * (length - 1) + [1.0], type=pa.float32()),
                "next.done": pa.array([False] * (length - 1) + [True]),
            }
        )
        pq.write_table(parquet, data_dir / f"episode_{episode_index:06d}.parquet")

        video_dir = dataset_path / "videos" / "chunk-000" / video_key
        video_dir.mkdir(parents=True)
        (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"fake-video-bytes")

        meta_dir = dataset_path / "meta"
        meta_dir.mkdir(parents=True)
        with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as file:
            file.write(
                json.dumps({"episode_index": episode_index, "task": instruction}) + "\n"
            )

    def test_e003_window_latent_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            latent_dim = 16
            config = {
                "experiment_id": "E-003",
                "stage": "future_latent_prior",
                "latent_cache": {
                    "dataset_path": str(dataset_path),
                    "cache_root": str(cache_root),
                    "encoder_kind": "fake",
                    "encoder_name": "mowa-fake-encoder",
                    "encoder_version": "fake-v1",
                    "latent_dim": latent_dim,
                    "video_keys": ["observation.images.robot0_agentview_left"],
                    "episode_indices": [0],
                    "future_window_steps": 1,
                    "allow_partial_windows": False,
                },
                "interface": {
                    "current_latent_dim": latent_dim,
                    "text_hidden_dim": 32,
                    "hidden_dim": 32,
                    "future_latent_dim": latent_dim,
                    "batch_size_smoke": 2,
                },
            }

            torch.manual_seed(42)
            report = run_e003_window_latent_train_smoke(
                config,
                execute_cache=True,
                train_steps=2,
                device="cpu",
            )

            self.assertNotEqual(report["go_no_go"], "No-Go", report)
            self.assertTrue(report["loss_is_finite"], report)
            self.assertEqual(report["steps_completed"], 2)
            self.assertEqual(
                report["shapes"]["current_latent"],
                [2, latent_dim],
            )
            self.assertEqual(
                report["shapes"]["future_latent_target"],
                [2, latent_dim],
            )
            self.assertEqual(
                report["shapes"]["predicted_future_latent"],
                [2, latent_dim],
            )

    def test_e004_window_latent_train_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            history_steps = 4
            self._write_minimal_episode(dataset_path, length=16)

            latent_dim = 16
            condition_hidden_dim = 16
            config = {
                "experiment_id": "E-004",
                "stage": "hlc_gci",
                "latent_cache": {
                    "dataset_path": str(dataset_path),
                    "cache_root": str(cache_root),
                    "encoder_kind": "fake",
                    "encoder_name": "mowa-fake-encoder",
                    "encoder_version": "fake-v1",
                    "latent_dim": latent_dim,
                    "video_keys": ["observation.images.robot0_agentview_left"],
                    "episode_indices": [0],
                    "future_window_steps": 1,
                    "allow_partial_windows": False,
                },
                "interface": {
                    "history_latent_dim": latent_dim,
                    "condition_hidden_dim": condition_hidden_dim,
                    "history_steps": history_steps,
                    "compressed_history_dim": 8,
                    "gate_hidden_dim": 4,
                    "condition_token_count": 2,
                    "batch_size_smoke": 2,
                },
            }

            torch.manual_seed(42)
            report = run_e004_window_latent_train_smoke(
                config,
                execute_cache=True,
                train_steps=2,
                device="cpu",
            )

            self.assertNotEqual(report["go_no_go"], "No-Go", report)
            self.assertTrue(report["loss_is_finite"], report)
            self.assertTrue(report["gradients_finite"], report)
            self.assertEqual(report["steps_completed"], 2)
            self.assertEqual(
                report["shapes"]["history_latent"],
                [2, history_steps, latent_dim],
            )
            self.assertEqual(
                report["shapes"]["condition_tokens"],
                [2, 2, condition_hidden_dim],
            )
            self.assertEqual(
                report["shapes"]["gated_condition_tokens"],
                [2, 2, condition_hidden_dim],
            )

    def test_window_manifest_does_not_leak_future_into_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            self._write_minimal_episode(dataset_path, length=16)

            from starVLA.dataloader.mowa.episode_latent_store import (
                MoWAEpisodeLatentStoreConfig,
                build_mowa_episode_latent_store,
            )

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=8,
                    dry_run=False,
                )
            )
            build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=4,
                        future_steps=4,
                        action_chunk_steps=4,
                    ),
                    video_keys=("observation.images.robot0_agentview_left",),
                )
            )

            from starVLA.dataloader.mowa.window_latent_sample import (
                MoWAWindowLatentSampleDataset,
                assert_no_future_leakage,
            )

            dataset = MoWAWindowLatentSampleDataset(manifest_path)
            sample = dataset[0]
            assert_no_future_leakage(sample)
            self.assertNotIn("future_latents", {"current_latent", "history_latents", "language"})


if __name__ == "__main__":
    unittest.main()
