"""Integration test for E-003/E-004 window-latent data path in train_starvla.

This test verifies that the unified ``MoWALatentCacheDataset`` can serve the
episode-level store + window manifest layout through the same API used by
``gr00t_lerobot/datasets.py``, and that ``LeRobotSingleDataset`` attaches
``mowa_current_latent``, ``mowa_future_latent_target`` and
``mowa_history_latent`` to each sample without loading the full StarFlowVLA
model.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    ModalityConfig,
    _attach_mowa_latent_cache,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAEpisodeLatentStoreConfig,
    build_mowa_episode_latent_store,
)
from starVLA.dataloader.mowa.latent_cache_dataset import MoWALatentCacheDataset
from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    build_mowa_window_manifest,
)


class TrainStarVLAWindowLatentIntegrationTest(unittest.TestCase):
    _STATE_DIM = 16
    _ACTION_DIM = 12
    _LATENT_DIM = 16
    _HISTORY_STEPS = 4
    _FUTURE_STEPS = 4
    _EPISODE_LENGTH = 16
    _VIDEO_KEY = "observation.images.robot0_agentview_left"

    def _write_parquet(self, dataset_path: Path, episode_index: int = 0) -> None:
        data_dir = dataset_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        length = self._EPISODE_LENGTH

        state_values = [float(i) for i in range(length * self._STATE_DIM)]
        action_values = [float(i) for i in range(length * self._ACTION_DIM)]

        table = pa.table(
            {
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "task_index": pa.array([0] * length, type=pa.int64()),
                "timestamp": pa.array([0.05 * i for i in range(length)], type=pa.float32()),
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(state_values), self._STATE_DIM
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action_values), self._ACTION_DIM
                ),
                "next.reward": pa.array([0.0] * (length - 1) + [1.0], type=pa.float32()),
                "next.done": pa.array([False] * (length - 1) + [True]),
            }
        )
        pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")

    def _write_meta(self, dataset_path: Path, episode_index: int = 0) -> None:
        meta_dir = dataset_path / "meta"
        meta_dir.mkdir(parents=True)

        modality = {
            "state": {
                "state": {
                    "start": 0,
                    "end": self._STATE_DIM,
                    "dtype": "float32",
                    "absolute": True,
                    "rotation_type": None,
                    "original_key": "observation.state",
                }
            },
            "action": {
                "action": {
                    "start": 0,
                    "end": self._ACTION_DIM,
                    "dtype": "float32",
                    "absolute": True,
                    "rotation_type": None,
                    "original_key": "action",
                }
            },
            "video": {
                "primary": {
                    "original_key": self._VIDEO_KEY,
                }
            },
        }
        (meta_dir / "modality.json").write_text(
            json.dumps(modality, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        info = {
            "codebase_version": "v2.1",
            "robot_type": "franka",
            "total_episodes": 1,
            "total_frames": self._EPISODE_LENGTH,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 20,
            "splits": {"train": f"{episode_index}:{episode_index + 1}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                self._VIDEO_KEY: {
                    "dtype": "video",
                    "shape": [224, 224, 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.height": 224,
                        "video.width": 224,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": 20,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
            },
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": episode_index, "length": self._EPISODE_LENGTH})
            + "\n",
            encoding="utf-8",
        )
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps(
                {
                    "task_index": 0,
                    "task": "open the drawer",
                    "episode_index": episode_index,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        # The episode store builder checks that the source video exists before
        # encoding, but the fake encoder does not read its content.
        video_dir = dataset_path / "videos" / "chunk-000" / self._VIDEO_KEY
        video_dir.mkdir(parents=True)
        (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"fake-video")

    def _build_latent_cache(self, dataset_path: Path, cache_root: Path) -> Path:
        build_mowa_episode_latent_store(
            MoWAEpisodeLatentStoreConfig(
                dataset_path=dataset_path,
                cache_root=cache_root,
                video_keys=(self._VIDEO_KEY,),
                latent_dim=self._LATENT_DIM,
                latent_type="pooled_vector",
                dry_run=False,
            )
        )
        manifest_path = cache_root / "window_manifest.parquet"
        build_mowa_window_manifest(
            MoWAWindowManifestConfig(
                cache_root=cache_root,
                output_path=manifest_path,
                window_config=MoWAWindowConfig(
                    history_steps=self._HISTORY_STEPS,
                    future_steps=self._FUTURE_STEPS,
                    action_chunk_steps=self._FUTURE_STEPS,
                ),
                video_keys=(self._VIDEO_KEY,),
                allow_partial_windows=False,
            )
        )
        return manifest_path

    def _make_dataset(
        self,
        dataset_path: Path,
        cache_root: Path,
        manifest_path: Path,
    ) -> LeRobotSingleDataset:
        modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["video.primary"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["state.state"]),
            "action": ModalityConfig(
                delta_indices=list(range(self._FUTURE_STEPS)),
                modality_keys=["action.action"],
            ),
        }
        data_cfg = {
            "mowa_latent_cache": {
                "cache_root": str(cache_root),
                "manifest_path": str(manifest_path),
                "video_keys": [self._VIDEO_KEY],
            },
            "action_mode": "abs",
            "video_backend": "torchvision_av",
        }
        return LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="torchvision_av",
            transforms=ComposedModalityTransform(transforms=[]),
            data_cfg=data_cfg,
        )

    def test_unified_cache_dataset_exposes_same_api_as_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_parquet(dataset_path)
            self._write_meta(dataset_path)
            manifest_path = self._build_latent_cache(dataset_path, cache_root)

            cache_dataset = MoWALatentCacheDataset(
                cache_root=cache_root, manifest_path=manifest_path
            )
            self.assertGreater(len(cache_dataset), 0)
            self.assertTrue(cache_dataset.sample_keys)

            first_key = cache_dataset.sample_keys[0]
            sample = cache_dataset.get_sample(
                episode_index=first_key[0],
                anchor_index=first_key[1],
                video_key=first_key[2],
            )
            self.assertIn("current_latent", sample)
            self.assertIn("future_latent", sample)
            self.assertIn("history_latent", sample)
            self.assertEqual(tuple(sample["current_latent"].shape), (self._LATENT_DIM,))
            self.assertEqual(
                tuple(sample["future_latent"].shape), (self._LATENT_DIM,)
            )
            self.assertEqual(
                tuple(sample["history_latent"].shape),
                (self._HISTORY_STEPS, self._LATENT_DIM),
            )

    def test_filter_steps_and_attach_latents_through_lerobot_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_parquet(dataset_path)
            self._write_meta(dataset_path)
            manifest_path = self._build_latent_cache(dataset_path, cache_root)

            dataset = self._make_dataset(dataset_path, cache_root, manifest_path)

            # Without the cache filter there would be 16 steps; the window manifest
            # only contains valid windows (anchor 3..11 -> 9 samples).
            self.assertEqual(len(dataset.all_steps), 9)

            trajectory_id, base_index = dataset.all_steps[0]
            sample = _attach_mowa_latent_cache({}, dataset, trajectory_id, base_index)

            self.assertIn("mowa_current_latent", sample)
            self.assertIn("mowa_future_latent_target", sample)
            self.assertIn("mowa_history_latent", sample)

            current = sample["mowa_current_latent"]
            future = sample["mowa_future_latent_target"]
            history = sample["mowa_history_latent"]

            self.assertIsInstance(current, torch.Tensor)
            self.assertIsInstance(future, torch.Tensor)
            self.assertIsInstance(history, torch.Tensor)

            self.assertEqual(tuple(current.shape), (self._LATENT_DIM,))
            self.assertEqual(tuple(future.shape), (self._LATENT_DIM,))
            self.assertEqual(tuple(history.shape), (self._HISTORY_STEPS, self._LATENT_DIM))
            self.assertNotIn("mowa_action_valid_mask", sample)

    def test_spatial_latent_store_returns_visual_latent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset"
            cache_root = root / "cache"
            self._write_parquet(dataset_path)
            self._write_meta(dataset_path)

            # Build an episode store that advertises a spatial latent type.
            build_mowa_episode_latent_store(
                MoWAEpisodeLatentStoreConfig(
                    dataset_path=dataset_path,
                    cache_root=cache_root,
                    video_keys=(self._VIDEO_KEY,),
                    latent_dim=self._LATENT_DIM,
                    latent_type="vae_spatial",
                    dry_run=False,
                )
            )
            manifest_path = cache_root / "window_manifest.parquet"
            build_mowa_window_manifest(
                MoWAWindowManifestConfig(
                    cache_root=cache_root,
                    output_path=manifest_path,
                    window_config=MoWAWindowConfig(
                        history_steps=self._HISTORY_STEPS,
                        future_steps=self._FUTURE_STEPS,
                        action_chunk_steps=self._FUTURE_STEPS,
                    ),
                    video_keys=(self._VIDEO_KEY,),
                    allow_partial_windows=False,
                )
            )

            cache_dataset = MoWALatentCacheDataset(
                cache_root=cache_root, manifest_path=manifest_path
            )
            sample = cache_dataset[0]
            self.assertIn("visual_latent", sample)
            self.assertTrue(torch.is_tensor(sample["visual_latent"]))
            self.assertIn("lang", sample)


class _DummyDataset:
    def __init__(self):
        self._data = [{}]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def save_dataset_statistics(self, _path):
        pass


class BuildDataloaderManifestWiringTest(unittest.TestCase):
    def test_instruction_text_cache_does_not_enable_visual_latent_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = OmegaConf.create(
                {
                    "output_dir": str(root / "out"),
                    "run_root_dir": str(root / "run"),
                    "run_id": "smoke",
                    "datasets": {
                        "vla_data": {
                            "dataset_py": "lerobot_datasets",
                            "per_device_batch_size": 1,
                            "num_workers": 0,
                        }
                    },
                    "latent_cache": {
                        "instruction_text_latent": str(root / "instruction_text_latents.pt"),
                    },
                }
            )

            with patch(
                "starVLA.dataloader.lerobot_datasets.get_vla_dataset"
            ) as mock_get:
                mock_get.return_value = _DummyDataset()
                build_dataloader(cfg, dataset_py="lerobot_datasets")
                passed_cfg = mock_get.call_args.kwargs["data_cfg"]
                self.assertNotIn("mowa_latent_cache", passed_cfg)

    def test_build_dataloader_defaults_manifest_path_from_window_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            cache_root.mkdir()
            manifest_path = cache_root / "window_manifest.parquet"
            manifest_path.write_bytes(b"dummy")

            cfg = OmegaConf.create(
                {
                    "output_dir": str(root / "out"),
                    "run_root_dir": str(root / "run"),
                    "run_id": "smoke",
                    "datasets": {
                        "vla_data": {
                            "dataset_py": "lerobot_datasets",
                            "per_device_batch_size": 1,
                            "num_workers": 0,
                        }
                    },
                    "latent_cache": {
                        "cache_root": str(cache_root),
                        "video_keys": ["observation.images.robot0_agentview_left"],
                    },
                }
            )

            with patch(
                "starVLA.dataloader.lerobot_datasets.get_vla_dataset"
            ) as mock_get:
                mock_get.return_value = _DummyDataset()
                build_dataloader(cfg, dataset_py="lerobot_datasets")

                passed_cfg = mock_get.call_args.kwargs["data_cfg"]
                self.assertEqual(
                    passed_cfg["mowa_latent_cache"]["manifest_path"],
                    str(manifest_path),
                )
                self.assertEqual(
                    passed_cfg["mowa_latent_cache"]["cache_root"],
                    str(cache_root),
                )

    def test_build_dataloader_prefers_manifest_matching_history_window_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_root = root / "cache"
            cache_root.mkdir()
            default_manifest = cache_root / "window_manifest.parquet"
            history_manifest = cache_root / "window_manifest_h10.parquet"
            default_manifest.write_bytes(b"default")
            history_manifest.write_bytes(b"h10")

            cfg = OmegaConf.create(
                {
                    "output_dir": str(root / "out"),
                    "run_root_dir": str(root / "run"),
                    "run_id": "smoke",
                    "datasets": {
                        "vla_data": {
                            "dataset_py": "lerobot_datasets",
                            "per_device_batch_size": 1,
                            "num_workers": 0,
                        }
                    },
                    "latent_cache": {
                        "cache_root": str(cache_root),
                        "history_window_steps": 10,
                        "video_keys": ["observation.images.robot0_agentview_left"],
                    },
                }
            )

            with patch(
                "starVLA.dataloader.lerobot_datasets.get_vla_dataset"
            ) as mock_get:
                mock_get.return_value = _DummyDataset()
                build_dataloader(cfg, dataset_py="lerobot_datasets")

                passed_cfg = mock_get.call_args.kwargs["data_cfg"]
                self.assertEqual(
                    passed_cfg["mowa_latent_cache"]["manifest_path"],
                    str(history_manifest),
                )


if __name__ == "__main__":
    unittest.main()
