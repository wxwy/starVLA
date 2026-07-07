"""Tests for the fake MoWA latent cache builder / loader loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from starVLA.dataloader.mowa.latent_cache_builder import (
    MoWALatentCacheBuildConfig,
    MoWALatentCacheDataset,
    build_mowa_latent_cache,
    validate_mowa_latent_cache,
)


class MoWALatentCacheBuilderTest(unittest.TestCase):
    def _write_minimal_episode(self, dataset_path: Path, *, length: int = 8) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        data_dir = dataset_path / "data" / "chunk-000"
        video_dir = (
            dataset_path
            / "videos"
            / "chunk-000"
            / "observation.images.robot0_agentview_left"
        )
        data_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)

        parquet = pa.table(
            {
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([0] * length, type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
                "timestamp": pa.array([0.05 * idx for idx in range(length)], type=pa.float32()),
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * 16)]),
                    16,
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array([float(value) for value in range(length * 12)]),
                    12,
                ),
                "next.reward": pa.array([0.0] * (length - 1) + [1.0], type=pa.float32()),
                "next.done": pa.array([False] * (length - 1) + [True]),
            }
        )
        pq.write_table(parquet, data_dir / "episode_000000.parquet")
        (video_dir / "episode_000000.mp4").write_bytes(b"fake-video-bytes")

    def test_build_validate_and_load_fake_latent_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            report = build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    encoder_name="wan-fake-encoder",
                    encoder_version="fake-v1",
                    latent_dim=16,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=2,
                    history_window_steps=3,
                    anchor_mode="smoke",
                    dry_run=False,
                )
            ).to_dict()

            validation = validate_mowa_latent_cache(cache_root).to_dict()
            dataset = MoWALatentCacheDataset(cache_root)
            sample = dataset.get_sample(
                episode_index=0,
                anchor_index=2,
                video_key="observation.images.robot0_agentview_left",
            )

            self.assertTrue((cache_root / "manifest.json").is_file())
            self.assertTrue(validation["manifest_exists"])
            self.assertEqual(validation["artifact_count"], 3)
            self.assertEqual(validation["readable_artifact_count"], 3)
            self.assertEqual(validation["missing_artifact_count"], 0)
            self.assertTrue(validation["all_ok"])
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(sample["current_latent"].shape), (16,))
            self.assertEqual(tuple(sample["future_latent"].shape), (16,))
            self.assertEqual(tuple(sample["history_latent"].shape), (16,))
            self.assertEqual(sample["metadata"]["current"]["window_role"], "current")
            self.assertEqual(sample["metadata"]["future"]["window_role"], "future")
            self.assertEqual(sample["metadata"]["history"]["window_role"], "history")

        self.assertEqual(report["written_artifact_count"], 3)
        self.assertEqual(report["planned_artifact_count"], 3)

    def test_dry_run_plans_without_writing_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            report = build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=2,
                    history_window_steps=3,
                    dry_run=True,
                )
            ).to_dict()

            self.assertFalse((cache_root / "manifest.json").is_file())

        self.assertEqual(report["dry_run"], True)
        self.assertEqual(report["written_artifact_count"], 0)
        self.assertEqual(report["planned_artifact_count"], 3)


if __name__ == "__main__":
    unittest.main()
