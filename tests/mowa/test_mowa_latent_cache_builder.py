"""Tests for the fake MoWA latent cache builder / loader loop."""

from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from starVLA.dataloader.mowa.latent_cache_builder import (
    MoWALatentCacheBuildConfig,
    MoWALatentCacheDataset,
    MoWAWanVaeLatentEncoderAdapter,
    build_mowa_latent_cache,
    _build_latent_encoder_adapter,
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
            self.assertEqual(tuple(sample["history_latent"].shape), (3, 16))
            self.assertFalse(torch.allclose(sample["history_latent"][0], sample["history_latent"][1]))
            self.assertEqual(sample["metadata"]["current"]["window_role"], "current")
            self.assertEqual(sample["metadata"]["future"]["window_role"], "future")
            self.assertEqual(sample["metadata"]["history"]["window_role"], "history")

        self.assertEqual(report["written_artifact_count"], 3)
        self.assertEqual(report["planned_artifact_count"], 3)
        self.assertEqual(
            report["go_no_go"],
            "TBD: latent cache write completed; validate cache before integration",
        )

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

    def test_allow_partial_windows_enables_boundary_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=2)

            strict_report = build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=3,
                    history_window_steps=3,
                    anchor_mode="all",
                    allow_partial_windows=False,
                    dry_run=False,
                )
            ).to_dict()

            self.assertEqual(strict_report["written_artifact_count"], 0)
            self.assertEqual(strict_report["planned_artifact_count"], 0)

            partial_cache_root = root / "partial_cache"
            partial_report = build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=partial_cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=3,
                    history_window_steps=3,
                    anchor_mode="all",
                    allow_partial_windows=True,
                    dry_run=False,
                )
            ).to_dict()

            validation = validate_mowa_latent_cache(partial_cache_root).to_dict()
            dataset = MoWALatentCacheDataset(partial_cache_root)

            self.assertEqual(partial_report["written_artifact_count"], 5)
            self.assertEqual(partial_report["planned_artifact_count"], 5)
            self.assertTrue((partial_cache_root / "manifest.json").is_file())
            self.assertTrue(validation["all_ok"])
            self.assertEqual(len(dataset), 2)

    def test_existing_cache_skips_encode_before_encoder_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            class _CountingEncoder:
                encoder_name = "wan-fake-encoder"
                encoder_version = "fake-v1"
                latent_dim = 8

                def __init__(self) -> None:
                    self.encode_calls = 0

                def encode(self, *, window_role: str = "current", frame_indices: tuple[int, ...] = (), **_):
                    self.encode_calls += 1
                    if window_role == "history":
                        return torch.arange(len(frame_indices) * self.latent_dim, dtype=torch.float32).reshape(
                            len(frame_indices), self.latent_dim
                        )
                    return torch.arange(self.latent_dim, dtype=torch.float32)

            encoder = _CountingEncoder()
            build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=2,
                    history_window_steps=3,
                    dry_run=False,
                ),
                encoder=encoder,
            )

            second_report = build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=2,
                    history_window_steps=3,
                    dry_run=False,
                ),
                encoder=encoder,
            ).to_dict()

            self.assertEqual(encoder.encode_calls, 3)
            self.assertEqual(second_report["skipped_existing_count"], 3)
            self.assertEqual(second_report["planned_artifact_count"], 0)
            self.assertEqual(
                second_report["go_no_go"],
                "No-Go: no latent cache artifacts were planned",
            )

    def test_dataset_reuses_loaded_payloads_for_repeated_sample_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            build_mowa_latent_cache(
                MoWALatentCacheBuildConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    latent_dim=8,
                    video_keys=("observation.images.robot0_agentview_left",),
                    current_window_steps=1,
                    future_window_steps=2,
                    history_window_steps=3,
                    dry_run=False,
                )
            )

            dataset = MoWALatentCacheDataset(cache_root)
            import torch

            original_load = torch.load
            load_calls = 0

            def _counting_load(*args, **kwargs):
                nonlocal load_calls
                load_calls += 1
                return original_load(*args, **kwargs)

            try:
                torch.load = _counting_load  # type: ignore[assignment]
                sample_1 = dataset.get_sample(
                    episode_index=0,
                    anchor_index=2,
                    video_key="observation.images.robot0_agentview_left",
                )
                sample_2 = dataset.get_sample(
                    episode_index=0,
                    anchor_index=2,
                    video_key="observation.images.robot0_agentview_left",
                )
            finally:
                torch.load = original_load  # type: ignore[assignment]

            self.assertEqual(load_calls, 3)
            self.assertEqual(tuple(sample_1["current_latent"].shape), (8,))
            self.assertEqual(tuple(sample_2["future_latent"].shape), (8,))

    def test_dataset_payload_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            cache_root.mkdir(parents=True)

            import torch

            artifacts = []
            for index in range(129):
                cache_path = cache_root / f"artifact_{index:03d}.pt"
                torch.save({"latent": torch.tensor([float(index)]), "metadata": {"index": index}}, cache_path)
                artifacts.append(
                    {
                        "episode_index": 0,
                        "anchor_index": 0,
                        "video_key": "observation.images.robot0_agentview_left",
                        "window_role": f"role_{index:03d}",
                        "cache_path": str(cache_path),
                    }
                )

            dataset = MoWALatentCacheDataset.__new__(MoWALatentCacheDataset)
            dataset.cache_root = cache_root
            dataset.manifest_path = cache_root / "manifest.json"
            dataset._artifacts = tuple(artifacts)
            dataset._payload_cache = OrderedDict()
            dataset._sample_keys = ((0, 0, "observation.images.robot0_agentview_left"),)
            sample = dataset.get_sample(
                episode_index=0,
                anchor_index=0,
                video_key="observation.images.robot0_agentview_left",
            )

            self.assertLessEqual(len(dataset._payload_cache), dataset._payload_cache_maxsize)
            self.assertEqual(len(sample["latents"]), 129)

    def test_encoder_factory_defaults_to_fake_and_requires_model_path_for_real_encoder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = MoWALatentCacheBuildConfig(
                dataset_path=root / "dataset",
                cache_root=root / "cache",
            )
            encoder = _build_latent_encoder_adapter(cfg)
            self.assertEqual(encoder.encoder_name, "wan-fake-encoder")

            real_cfg = MoWALatentCacheBuildConfig(
                dataset_path=root / "dataset",
                cache_root=root / "cache",
                encoder_kind="wan2.2-vae",
            )
            with self.assertRaisesRegex(ValueError, "encoder_model_path"):
                real_cfg.validate()

    def test_wan_encoder_uses_decord_backend_helper(self):
        encoder = MoWAWanVaeLatentEncoderAdapter(
            model_path=Path("/tmp/nonexistent-wan-model"),
            latent_dim=8,
            video_backend="decord",
        )
        helper_frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "video.mp4"
            source_path.write_bytes(b"fake-video-bytes")
            from unittest.mock import patch

            with patch(
                "starVLA.dataloader.gr00t_lerobot.video.get_frames_by_indices",
                return_value=helper_frames,
            ) as mocked_helper:
                frames = encoder._load_frames(source_path, (1, 3))

        mocked_helper.assert_called_once()
        self.assertEqual(len(frames), 2)
        self.assertEqual(tuple(frames[0].shape), (4, 4, 3))

    def test_wan_encoder_rejects_unknown_video_backend(self):
        encoder = MoWAWanVaeLatentEncoderAdapter(
            model_path=Path("/tmp/nonexistent-wan-model"),
            latent_dim=8,
            video_backend="unknown",
        )
        with self.assertRaisesRegex(NotImplementedError, "video_backend"):
            encoder.encode(
                dataset_path=Path("/tmp/dataset"),
                episode_index=0,
                anchor_index=0,
                video_key="observation.images.robot0_agentview_left",
                window_role="current",
                frame_indices=(0,),
                row_count=1,
                source_path=Path("/tmp/fake.mp4"),
                source_identity="identity",
            )


if __name__ == "__main__":
    unittest.main()
