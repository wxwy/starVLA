"""Tests for raw video/state/action temporal alignment validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from starVLA.dataloader.mowa.temporal_alignment import (
    validate_mowa_episode_temporal_alignment,
)


class TemporalAlignmentTest(unittest.TestCase):
    def _write_episode(self, root: Path, *, length: int = 4) -> None:
        data_dir = root / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array([[0.0]] * length),
                    "action": pa.array([[0.0]] * length),
                }
            ),
            data_dir / "episode_000000.parquet",
        )
        for video_key in ("global", "wrist"):
            video_path = root / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"placeholder")

    def test_all_configured_views_share_parquet_time_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_episode(root)
            report = validate_mowa_episode_temporal_alignment(
                root,
                video_keys=("global", "wrist"),
                video_frame_counter=lambda _: 4,
            )

            self.assertEqual(report.checked_episode_count, 1)
            self.assertEqual(report.valid_episode_count, 1)
            self.assertEqual(report.invalid_episode_count, 0)
            self.assertEqual(report.episodes[0].video_frame_counts, {"global": 4, "wrist": 4})

    def test_reports_per_view_frame_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_episode(root)
            frame_counts = {"global": 4, "wrist": 3}
            report = validate_mowa_episode_temporal_alignment(
                root,
                video_keys=("global", "wrist"),
                video_frame_counter=lambda path: frame_counts[path.parent.name],
            )

            self.assertEqual(report.invalid_episode_count, 1)
            self.assertIn("video=3, state/action=4", report.episodes[0].errors[0])

    def test_reports_missing_configured_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_episode(root)
            report = validate_mowa_episode_temporal_alignment(
                root,
                video_keys=("global", "missing"),
                video_frame_counter=lambda _: 4,
            )

            self.assertEqual(report.invalid_episode_count, 1)
            self.assertIsNone(report.episodes[0].video_frame_counts["missing"])
            self.assertIn("missing video for missing", report.episodes[0].errors[0])
