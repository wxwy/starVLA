"""Tests for episode-level latent cache + window manifest + sample assembly."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAEpisodeLatentStore,
    MoWAEpisodeLatentStoreConfig,
    MoWAFakeLatentEncoderAdapter,
    MoWAWanVaeEpisodeEncoderAdapter,
    build_mowa_episode_latent_store,
    validate_mowa_episode_latent_store,
)
from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_latent_sample import (
    MoWAWindowLatentSampleDataset,
    assert_no_future_leakage,
    validate_window_indices,
)
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    MoWAWindowManifestEntry,
    build_mowa_window_manifest,
    load_mowa_window_manifest,
)


class EpisodeLevelLatentCacheTest(unittest.TestCase):
    def _write_minimal_episode(
        self,
        dataset_path: Path,
        *,
        episode_index: int = 0,
        length: int = 16,
        state_dim: int = 16,
        action_dim: int = 12,
        video_keys: tuple[str, ...] = ("observation.images.robot0_agentview_left",),
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

        for video_key in video_keys:
            video_dir = dataset_path / "videos" / "chunk-000" / video_key
            video_dir.mkdir(parents=True)
            (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"fake-video-bytes")

    def test_build_and_validate_episode_latent_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=16)

            report = build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=16,
                    dry_run=False,
                )
            ).to_dict()

            validation = validate_mowa_episode_latent_store(cache_root).to_dict()
            store = MoWAEpisodeLatentStore(cache_root / "ep_000000.h5")

            self.assertEqual(report["written_count"], 1)
            self.assertEqual(report["failed_count"], 0)
            self.assertTrue((cache_root / "ep_000000.h5").is_file())
            self.assertEqual(validation["valid_count"], 1)

            self.assertEqual(store.attrs["episode_id"], "ep_000000")
            self.assertEqual(store.attrs["frame_count"], 16)
            self.assertEqual(store.attrs["latent_type"], "pooled_vector")
            self.assertEqual(store.list_video_keys(), ("observation.images.robot0_agentview_left",))
            self.assertEqual(store.num_frames("observation.images.robot0_agentview_left"), 16)

            full_latents = store.get_latents("observation.images.robot0_agentview_left")
            self.assertEqual(tuple(full_latents.shape), (16, 16))

            current = store.get_latents("observation.images.robot0_agentview_left", (8,))
            self.assertEqual(tuple(current.shape), (1, 16))

    def test_build_and_load_window_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            self._write_minimal_episode(dataset_path, length=16)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=16,
                    dry_run=False,
                )
            )

            report = build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=4,
                        future_steps=4,
                        action_chunk_steps=4,
                    ),
                    video_keys=("observation.images.robot0_agentview_left",),
                    history_stride=1,
                    wam_hz=4.0,
                )
            ).to_dict()

            entries = load_mowa_window_manifest(manifest_path)
            self.assertEqual(report["window_count"], 9)
            self.assertEqual(len(entries), 9)

            first = entries[0]
            self.assertEqual(first.anchor_index, 3)
            self.assertEqual(first.history_indices, (0, 1, 2, 3))
            self.assertEqual(first.current_index, 3)
            self.assertEqual(first.future_indices, (4, 5, 6, 7))
            self.assertEqual(first.action_chunk_indices, (3, 4, 5, 6))

            last = entries[-1]
            self.assertEqual(last.anchor_index, 11)
            self.assertEqual(last.future_indices, (12, 13, 14, 15))

    def test_dataset_assembles_window_latent_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            self._write_minimal_episode(dataset_path, length=16)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=16,
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

            dataset = MoWAWindowLatentSampleDataset(manifest_path)
            sample = dataset[0]

            self.assertEqual(sample.episode_id, "ep_000000")
            self.assertEqual(sample.anchor_index, 3)
            self.assertEqual(tuple(sample.current_latent.shape), (16,))
            self.assertEqual(tuple(sample.history_latents.shape), (4, 16))
            self.assertEqual(tuple(sample.future_latents.shape), (4, 16))
            self.assertEqual(tuple(sample.action_chunk.shape), (4, 12))
            self.assertIsInstance(sample.language, str)

    def test_index_validation_enforces_boundaries(self):
        valid = MoWAWindowManifestEntry(
            sample_id="s",
            episode_id="ep_000000",
            episode_latent_path="/tmp/ep_000000.h5",
            task_name="t",
            anchor_index=8,
            anchor_timestamp=2.0,
            video_key="v",
            history_indices=(4, 5, 6, 7, 8),
            current_index=8,
            future_indices=(9, 10, 11, 12),
            robot_state_indices=(4, 5, 6, 7, 8),
            history_action_indices=(4, 5, 6, 7),
            action_chunk_indices=(8, 9, 10, 11),
            label_sidecar_path=None,
            label_index=8,
            history_seconds=1.0,
            future_seconds=1.0,
            wam_hz=4.0,
            history_stride=1,
            split="train",
            status="target",
        )
        validate_window_indices(valid)

        invalid_history = self._replace_entry(valid, history_indices=(4, 5, 6, 7, 9))
        with self.assertRaisesRegex(ValueError, "history_indices cannot include future"):
            validate_window_indices(invalid_history)

        invalid_current = self._replace_entry(valid, current_index=7)
        with self.assertRaisesRegex(ValueError, "current_index must equal anchor_index"):
            validate_window_indices(invalid_current)

        invalid_future = self._replace_entry(valid, future_indices=(8, 9, 10, 11))
        with self.assertRaisesRegex(ValueError, "future_indices must be strictly after"):
            validate_window_indices(invalid_future)

    def test_history_can_exclude_anchor_when_configured(self):
        # This test documents the alternative boundary convention.
        entry = MoWAWindowManifestEntry(
            sample_id="s",
            episode_id="ep_000000",
            episode_latent_path="/tmp/ep_000000.h5",
            task_name="t",
            anchor_index=8,
            anchor_timestamp=2.0,
            video_key="v",
            history_indices=(4, 5, 6, 7),
            current_index=8,
            future_indices=(9, 10, 11, 12),
            robot_state_indices=(4, 5, 6, 7, 8),
            history_action_indices=(4, 5, 6, 7),
            action_chunk_indices=(8, 9, 10, 11),
            label_sidecar_path=None,
            label_index=8,
            history_seconds=1.0,
            future_seconds=1.0,
            wam_hz=4.0,
            history_stride=1,
            split="train",
            status="target",
        )
        validate_window_indices(entry)
        self.assertLess(max(entry.history_indices), entry.anchor_index)

    def test_action_chunk_and_labels_not_in_wam_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            self._write_minimal_episode(dataset_path, length=16)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=16,
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

            dataset = MoWAWindowLatentSampleDataset(manifest_path)
            sample = dataset[0]

            # action_chunk and labels are fields on the sample, not part of inputs.
            self.assertEqual(tuple(sample.action_chunk.shape), (4, 12))
            self.assertIn("labels", sample.__dict__)
            assert_no_future_leakage(sample)

    def test_multi_view_video_keys_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            video_keys = (
                "observation.images.robot0_agentview_left",
                "observation.images.robot0_agentview_right",
            )
            self._write_minimal_episode(dataset_path, length=8, video_keys=video_keys)

            report = build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=video_keys,
                    latent_dim=8,
                    dry_run=False,
                )
            ).to_dict()

            self.assertEqual(report["written_count"], 1)
            store = MoWAEpisodeLatentStore(cache_root / "ep_000000.h5")
            self.assertEqual(set(store.list_video_keys()), set(video_keys))

            for video_key in video_keys:
                latents = store.get_latents(video_key)
                self.assertEqual(tuple(latents.shape), (8, 8))

    def test_latent_shape_matches_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=32,
                    latent_shape_per_frame=("D",),
                    dry_run=False,
                )
            )

            store = MoWAEpisodeLatentStore(cache_root / "ep_000000.h5")
            self.assertEqual(store.attrs["latent_shape_per_frame"], "[32]")
            self.assertEqual(store.attrs["latent_type"], "pooled_vector")
            latents = store.get_latents("observation.images.robot0_agentview_left")
            self.assertEqual(tuple(latents.shape), (8, 32))

    def test_valid_mask_matches_frame_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=8)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=8,
                    dry_run=False,
                )
            )

            store = MoWAEpisodeLatentStore(cache_root / "ep_000000.h5")
            valid = store.get_valid_mask("observation.images.robot0_agentview_left")
            self.assertEqual(valid.shape[0], 8)
            self.assertTrue(valid.all())

    def test_label_sidecar_jsonl_loading_legacy_compat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            sidecar_root = root / "labels"
            self._write_minimal_episode(dataset_path, length=8)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=8,
                    dry_run=False,
                )
            )

            sidecar_root.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_root / "ep_000000.jsonl"
            with sidecar_path.open("w", encoding="utf-8") as file:
                for idx in range(8):
                    file.write(json.dumps({"row": idx, "failure_risk": 0.0}) + "\n")

            build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=3,
                        future_steps=2,
                        action_chunk_steps=2,
                    ),
                    video_keys=("observation.images.robot0_agentview_left",),
                    label_sidecar_root=sidecar_root,
                )
            )

            dataset = MoWAWindowLatentSampleDataset(manifest_path, label_sidecar_root=sidecar_root)
            sample = dataset[0]
            self.assertIn("row", sample.labels)
            self.assertEqual(sample.labels["row"], 2)

    def test_label_sidecar_parquet_loading_matches_current_writer_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = root / "manifest.parquet"
            sidecar_root = root / "labels"
            self._write_minimal_episode(dataset_path, length=8)

            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    latent_dim=8,
                    dry_run=False,
                )
            )

            sidecar_root.mkdir(parents=True, exist_ok=True)
            sidecar_path = sidecar_root / "episode_000000.parquet"
            pq.write_table(
                pa.table(
                    {
                        "row": list(range(8)),
                        "failure_risk": [0.0] * 8,
                        "failure_risk_mask": [True] * 8,
                    }
                ),
                sidecar_path,
            )

            report = build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=3,
                        future_steps=2,
                        action_chunk_steps=2,
                    ),
                    video_keys=("observation.images.robot0_agentview_left",),
                    label_sidecar_root=sidecar_root,
                )
            )

            self.assertTrue(report.window_count > 0)
            entries = load_mowa_window_manifest(manifest_path)
            self.assertEqual(
                Path(entries[0].label_sidecar_path).name,
                "episode_000000.parquet",
            )
            dataset = MoWAWindowLatentSampleDataset(manifest_path, label_sidecar_root=sidecar_root)
            sample = dataset[0]
            self.assertEqual(sample.labels["row"], 2)
            self.assertEqual(sample.labels["failure_risk"], 0.0)
            self.assertTrue(sample.labels["failure_risk_mask"])

    @staticmethod
    def _replace_entry(entry: MoWAWindowManifestEntry, **kwargs) -> MoWAWindowManifestEntry:
        data = entry.to_dict()
        data.update(kwargs)
        return MoWAWindowManifestEntry(
            sample_id=data["sample_id"],
            episode_id=data["episode_id"],
            episode_latent_path=data["episode_latent_path"],
            task_name=data["task_name"],
            anchor_index=data["anchor_index"],
            anchor_timestamp=data["anchor_timestamp"],
            video_key=data["video_key"],
            history_indices=tuple(data["history_indices"]),
            current_index=data["current_index"],
            future_indices=tuple(data["future_indices"]),
            robot_state_indices=tuple(data["robot_state_indices"]),
            history_action_indices=tuple(data["history_action_indices"]),
            action_chunk_indices=tuple(data["action_chunk_indices"]),
            label_sidecar_path=data["label_sidecar_path"],
            label_index=data["label_index"],
            history_seconds=data["history_seconds"],
            future_seconds=data["future_seconds"],
            wam_hz=data["wam_hz"],
            history_stride=data["history_stride"],
            split=data["split"],
            status=data["status"],
        )

    def test_wan_config_requires_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = MoWAEpisodeLatentStoreConfig(
                dataset_path=root / "dataset",
                cache_root=root / "cache",
                encoder_kind="wan2.2-vae",
            )
            with self.assertRaisesRegex(ValueError, "encoder_model_path"):
                cfg.validate()

    def test_wan_adapter_rejects_missing_model_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "Wan2.2 model path not found"):
            MoWAWanVaeEpisodeEncoderAdapter(
                model_path=Path("/tmp/nonexistent-wan-model"),
                latent_type="vae_spatial",
            )

    def test_wan_adapter_rejects_unsupported_latent_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported latent_type"):
            MoWAWanVaeEpisodeEncoderAdapter(
                model_path=Path("/tmp/nonexistent-wan-model"),
                latent_type="patch_token",
            )


if __name__ == "__main__":
    unittest.main()
