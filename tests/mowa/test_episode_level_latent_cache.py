"""Tests for episode-level latent cache + window manifest + sample assembly."""

from __future__ import annotations

from collections import OrderedDict
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import h5py
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
from starVLA.dataloader.mowa.latent_cache_dataset import MoWALatentCacheDataset
from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_latent_sample import (
    MoWAWindowLatentSampleDataset,
    assert_no_future_leakage,
    validate_window_indices,
)
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    default_mowa_window_manifest_path,
    MoWAWindowManifestEntry,
    _manifest_entry_from_row,
    build_mowa_window_manifest,
    load_mowa_window_manifest,
    load_mowa_window_manifest_table,
    slice_mowa_window_manifest_entry,
)
from starVLA.dataloader.mowa.sampler import _manifest_anchor_category


class EpisodeLevelLatentCacheTest(unittest.TestCase):
    def test_raw_episode_cache_is_bounded_lru(self):
        dataset = object.__new__(MoWAWindowLatentSampleDataset)
        dataset._raw_episode_cache_size = 2
        dataset._raw_episode_cache = OrderedDict()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / f"episode_{index}.parquet" for index in range(3)]
            for path in paths:
                path.touch()
            entries = [SimpleNamespace(source_episode_path=str(path)) for path in paths]
            payloads = [object() for _ in paths]

            with patch(
                "starVLA.dataloader.mowa.window_latent_sample.pd.read_parquet",
                side_effect=payloads,
            ) as read_parquet:
                self.assertIs(dataset._get_raw_episode_data(entries[0]), payloads[0])
                self.assertIs(dataset._get_raw_episode_data(entries[1]), payloads[1])
                self.assertIs(dataset._get_raw_episode_data(entries[0]), payloads[0])
                self.assertIs(dataset._get_raw_episode_data(entries[2]), payloads[2])

            self.assertEqual(read_parquet.call_count, 3)
            self.assertEqual(
                list(dataset._raw_episode_cache),
                [str(paths[0]), str(paths[2])],
            )

    def test_manifest_table_parses_entries_lazily(self):
        entry = MoWAWindowManifestEntry(
            sample_id="sample", episode_id="ep_000000", episode_latent_path="ep_000000.h5",
            task_name="task", anchor_index=1, anchor_timestamp=0.1,
            anchor_video_key="observation.images.robot0_agentview_left",
            history_indices=(0,), current_index=1, future_indices=(2,),
            history_state_indices=(0,), current_state_index=1, future_state_indices=(2,),
            history_action_indices=(0,), action_chunk_indices=(1,), label_sidecar_path=None,
            label_index=1, history_seconds=0.2, future_seconds=0.2, wam_hz=5.0,
            history_stride=1, split="train", status="regular",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.parquet"
            pq.write_table(pa.Table.from_pylist([entry.to_dict(), entry.to_dict()]), manifest_path)
            with patch(
                "starVLA.dataloader.mowa.window_manifest._manifest_entry_from_row",
                wraps=_manifest_entry_from_row,
            ) as parser:
                table = load_mowa_window_manifest_table(manifest_path)
                self.assertEqual(len(table), 2)
                parser.assert_not_called()
                self.assertEqual(table[0].sample_id, "sample")
                self.assertEqual(parser.call_count, 1)

    def test_maximum_manifest_entry_can_serve_shorter_e003_window(self):
        entry = MoWAWindowManifestEntry(
            sample_id="sample",
            episode_id="ep_000000",
            episode_latent_path="/tmp/ep_000000.h5",
            task_name="task",
            anchor_index=10,
            anchor_timestamp=2.0,
            anchor_video_key="observation.images.robot0_agentview_left",
            history_indices=tuple(range(1, 11)),
            current_index=10,
            future_indices=tuple(range(11, 19)),
            history_state_indices=tuple(range(1, 11)),
            current_state_index=10,
            future_state_indices=tuple(range(11, 19)),
            history_action_indices=tuple(range(0, 40)),
            action_chunk_indices=tuple(range(11, 43)),
            label_sidecar_path=None,
            label_index=10,
            history_seconds=2.0,
            future_seconds=1.6,
            wam_hz=5.0,
            history_stride=1,
            split="train",
            status="ok",
            history_valid_mask=(False,) * 10,
            future_valid_mask=(True,) * 8,
            action_valid_mask=(True,) * 32,
            history_action_valid_mask=(False,) * 40,
            history_state_valid_mask=(False,) * 10,
            future_state_valid_mask=(True,) * 8,
        )

        e003_entry = slice_mowa_window_manifest_entry(
            entry, history_steps=0, future_steps=8, action_chunk_steps=32
        )

        self.assertEqual(e003_entry.history_indices, ())
        self.assertEqual(e003_entry.history_action_indices, ())
        self.assertEqual(e003_entry.future_indices, tuple(range(11, 19)))
        self.assertEqual(e003_entry.action_chunk_indices, tuple(range(11, 43)))
        self.assertEqual(_manifest_anchor_category(e003_entry), "regular")
        self.assertEqual(
            _manifest_anchor_category(entry, history_steps=0, future_steps=8, action_chunk_steps=32),
            "regular",
        )

    def test_action_chunk_defaults_to_four_raw_steps_per_future_latent(self):
        config = MoWAWindowConfig(history_steps=0, future_steps=8)
        config.validate()
        self.assertEqual(config.resolved_action_chunk_steps, 32)

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

    def test_wan_episode_encoder_preserves_temporal_video_and_source_mapping(self):
        class FakeVideoProcessor:
            def preprocess_video(self, frames, *, height, width):
                self.frame_count = len(frames)
                self.frame_sizes = [frame.size for frame in frames]
                return torch.zeros((1, 3, len(frames), height, width), dtype=torch.float32)

        class FakeVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])
                self.input_shape = None

            def encode(self, video):
                self.input_shape = tuple(video.shape)
                temporal = (video.shape[2] - 1) // 4 + 1
                values = torch.arange(temporal, dtype=video.dtype, device=video.device).view(1, 1, temporal, 1, 1)
                latent = values.repeat(1, 2, 1, 1, 1) + torch.tensor([1.0, 2.0], device=video.device).view(1, 2, 1, 1, 1)
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_minimal_episode(dataset_path, length=6)

            encoder = MoWAWanVaeEpisodeEncoderAdapter(
                model_path=root,
                latent_type="vae_spatial",
                video_backend="decord",
            )
            processor = FakeVideoProcessor()
            vae = FakeVae()
            encoder._video_processor = processor
            encoder._vae = vae
            encoder._torch_dtype = torch.float32
            encoder._load_all_frames_decord = lambda *_: [np.zeros((4, 8, 3), dtype=np.uint8) for _ in range(6)]

            report = build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=("observation.images.robot0_agentview_left",),
                    encoder_kind="wan2.2-vae",
                    encoder_model_path=root,
                    latent_type="vae_spatial",
                    dry_run=False,
                ),
                encoder=encoder,
            )
            self.assertEqual(report.failed_count, 0)
            self.assertEqual(processor.frame_count, 9)  # 6 source frames -> 4n+1 padded clip.
            self.assertEqual(processor.frame_sizes, [(256, 256)] * 9)
            self.assertEqual(vae.input_shape, (1, 3, 9, 256, 256))

            store = MoWAEpisodeLatentStore(cache_root / "ep_000000.h5")
            self.assertEqual(store.num_frames("observation.images.robot0_agentview_left"), 6)
            self.assertEqual(store.num_latent_frames("observation.images.robot0_agentview_left"), 3)
            self.assertEqual(store.attrs["temporal_compression_factor"], 4)
            self.assertEqual(store.attrs["vae_preprocess_policy"], "center_crop_square_then_resize_256")
            with h5py.File(cache_root / "ep_000000.h5", "r") as file:
                self.assertEqual(
                    file["/indices/latent_source_frame_indices/observation.images.robot0_agentview_left"][()].tolist(),
                    [0, 4, 5],
                )

            gathered = store.get_latents("observation.images.robot0_agentview_left", (1, 4, 5))
            self.assertEqual(tuple(gathered.shape), (3, 2, 1, 1))
            self.assertEqual(gathered[:, 0, 0, 0].tolist(), [0.0, 0.5, 1.0])

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

    def test_window_latent_sample_uses_raw_episode_for_low_dim_and_sibling_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            manifest_path = default_mowa_window_manifest_path(cache_root, "window_manifest.parquet")
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
            with h5py.File(cache_root / "ep_000000.h5", "r+") as file:
                file["/robot/state"][...] = np.full((16, 16), -999.0, dtype=np.float32)
                file["/robot/action"][...] = np.full((16, 12), -777.0, dtype=np.float32)

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

            latent_cache_dataset = MoWALatentCacheDataset(cache_root)
            self.assertEqual(latent_cache_dataset.manifest_path, manifest_path)

            dataset = MoWAWindowLatentSampleDataset(manifest_path)
            sample = dataset[0]

            self.assertEqual(tuple(sample.robot_state_history.shape), (4, 16))
            self.assertEqual(tuple(sample.history_actions.shape), (4, 12))
            self.assertEqual(tuple(sample.action_chunk.shape), (4, 12))
            self.assertEqual(float(sample.robot_state_history[0, 0]), 0.0)
            self.assertEqual(float(sample.history_actions[0, 0]), 0.0)
            self.assertEqual(float(sample.action_chunk[0, 0]), 36.0)

    def test_index_validation_enforces_boundaries(self):
        valid = MoWAWindowManifestEntry(
            sample_id="s",
            episode_id="ep_000000",
            episode_latent_path="/tmp/ep_000000.h5",
            task_name="t",
            anchor_index=8,
            anchor_timestamp=2.0,
            anchor_video_key="v",
            history_indices=(4, 5, 6, 7, 8),
            current_index=8,
            future_indices=(9, 10, 11, 12),
            history_state_indices=(4, 5, 6, 7),
            current_state_index=8,
            future_state_indices=(9, 10, 11, 12),
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
            anchor_video_key="v",
            history_indices=(4, 5, 6, 7),
            current_index=8,
            future_indices=(9, 10, 11, 12),
            history_state_indices=(4, 5, 6, 7),
            current_state_index=8,
            future_state_indices=(9, 10, 11, 12),
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

            manifest_path = root / "window_manifest.parquet"
            manifest_report = build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=0,
                        future_steps=1,
                        action_chunk_steps=1,
                    ),
                    anchor_video_key=video_keys[0],
                )
            )
            entries = load_mowa_window_manifest(manifest_path)

            self.assertEqual(manifest_report.window_count, 7)
            self.assertEqual(len(entries), 7)
            self.assertTrue(all(entry.anchor_video_key == video_keys[0] for entry in entries))
            self.assertTrue(all(video_keys[1] not in entry.sample_id for entry in entries))
            sample = MoWAWindowLatentSampleDataset(
                manifest_path,
                video_keys=video_keys,
            )[0]
            self.assertEqual(sample.multi_view_video_keys, video_keys)
            self.assertEqual(tuple(sample.multi_view_history_latents.shape), (2, 0, 8))
            self.assertEqual(tuple(sample.multi_view_current_latents.shape), (2, 8))
            self.assertEqual(tuple(sample.multi_view_future_latents.shape), (2, 1, 8))

    def test_recursive_global_manifest_keeps_episode_ids_unique_and_loader_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            global_cache_root = root / "atomic"
            cache_roots = []
            for task_name in ("task_a", "task_b"):
                dataset_path = root / f"dataset_{task_name}"
                cache_root = global_cache_root / task_name / "20250822" / "lerobot"
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
                cache_roots.append(cache_root)

            manifest_path = global_cache_root / "window_manifests" / "global.parquet"
            report = build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=global_cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=0,
                        future_steps=1,
                        action_chunk_steps=1,
                    ),
                    recursive_cache_search=True,
                )
            )
            entries = load_mowa_window_manifest(manifest_path)

            self.assertEqual(report.window_count, 14)
            self.assertEqual(len({entry.sample_id for entry in entries}), 14)
            self.assertEqual(len(MoWALatentCacheDataset(cache_roots[0], manifest_path)), 7)

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

    def test_wan_manifest_drops_causal_first_chunk_and_pads_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            cache_root.mkdir()
            store_path = cache_root / "ep_000000.h5"
            video_key = "observation.images.robot0_agentview_left"
            with h5py.File(store_path, "w") as file:
                file.attrs["episode_id"] = "ep_000000"
                file.attrs["episode_index"] = 0
                file.attrs["frame_count"] = 17
                file.attrs["temporal_compression_factor"] = 4
                file.create_dataset(f"/latents/{video_key}", data=np.arange(10, dtype=np.float32).reshape(5, 2))
                file.create_dataset(
                    f"/indices/latent_source_frame_indices/{video_key}",
                    data=np.asarray([0, 4, 8, 12, 16], dtype=np.int64),
                )
                file.create_dataset("/robot/state", data=np.zeros((17, 2), dtype=np.float32))
                file.create_dataset(
                    "/robot/action",
                    data=np.repeat(np.arange(17, dtype=np.float32)[:, None], 2, axis=1),
                )
                file.create_dataset("/language/instruction", data="test")

            manifest_path = root / "wan_manifest.parquet"
            build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(history_steps=2, future_steps=2, action_chunk_steps=3),
                    video_keys=(video_key,),
                    wam_hz=4.0,
                )
            )
            entries = load_mowa_window_manifest(manifest_path)
            self.assertEqual([entry.current_index for entry in entries], [1, 2, 3, 4])
            self.assertEqual(entries[0].anchor_index, 4)
            self.assertEqual(entries[0].history_indices, (1, 1))
            self.assertEqual(entries[0].history_valid_mask, (False, False))
            self.assertEqual(entries[0].future_indices, (2, 3))
            self.assertEqual(entries[0].action_chunk_indices, (5, 6, 7))
            self.assertEqual(entries[0].history_action_indices, (0, 0, 0, 0, 1, 2, 3, 4))
            self.assertEqual(entries[0].history_action_valid_mask, (False, False, False, True, True, True, True, True))
            self.assertEqual(entries[-1].future_indices, (4, 4))
            self.assertEqual(entries[-1].future_valid_mask, (False, False))
            self.assertEqual(entries[-1].action_chunk_indices, (16, 16, 16))
            self.assertEqual(entries[-1].action_valid_mask, (False, False, False))

            sample = MoWAWindowLatentSampleDataset(manifest_path)[0]
            self.assertEqual(sample.current_latent.tolist(), [2.0, 3.0])
            self.assertEqual(sample.history_valid_mask.tolist(), [False, False])
            self.assertEqual(sample.future_done_target.tolist(), [0.0, 0.0])
            self.assertEqual(
                MoWAWindowLatentSampleDataset(manifest_path)[1].future_done_target.tolist(),
                [0.0, 1.0],
            )
            self.assertEqual(sample.history_actions[:, 0].tolist(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
            self.assertEqual(
                MoWAWindowLatentSampleDataset(manifest_path)[-1].action_chunk[:, 0].tolist(),
                [16.0, 16.0, 16.0],
            )
            self.assertEqual(
                MoWAWindowLatentSampleDataset(manifest_path)[-1].future_done_target.tolist(),
                [1.0, 1.0],
            )

            delta_manifest_path = root / "wan_delta_manifest.parquet"
            build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=delta_manifest_path,
                    window_config=MoWAWindowConfig(history_steps=2, future_steps=2, action_chunk_steps=3),
                    video_keys=(video_key,),
                    action_representation="delta",
                    wam_hz=4.0,
                )
            )
            self.assertEqual(
                MoWAWindowLatentSampleDataset(delta_manifest_path)[-1].action_chunk[:, 0].tolist(),
                [0.0, 0.0, 0.0],
            )

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
            anchor_video_key=data["anchor_video_key"],
            history_indices=tuple(data["history_indices"]),
            current_index=data["current_index"],
            future_indices=tuple(data["future_indices"]),
            history_state_indices=tuple(data["history_state_indices"]),
            current_state_index=data["current_state_index"],
            future_state_indices=tuple(data["future_state_indices"]),
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
