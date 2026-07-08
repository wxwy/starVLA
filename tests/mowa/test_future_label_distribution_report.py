"""Tests for the future-label distribution report tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.mowa.report_future_label_distribution import (
    _summarize_binary_head,
    _summarize_task,
    discover_task_sidecar_dirs,
)


class MoWAFutureLabelDistributionReportTest(unittest.TestCase):
    def test_summarize_binary_head_candidate(self):
        values = np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        masks = np.array([True, True, True, True, True], dtype=bool)
        summary = _summarize_binary_head(values, masks, "test_head")
        self.assertEqual(summary["status"], "candidate")
        self.assertEqual(summary["positive_count"], 2)
        self.assertEqual(summary["negative_count"], 3)

    def test_summarize_binary_head_single_class(self):
        values = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        masks = np.array([True, True, True], dtype=bool)
        summary = _summarize_binary_head(values, masks, "test_head")
        self.assertEqual(summary["status"], "blocked_single_class")

    def test_summarize_binary_head_low_minority(self):
        values = np.array([1.0] + [0.0] * 50, dtype=np.float64)
        masks = np.array([True] * 51, dtype=bool)
        summary = _summarize_binary_head(values, masks, "test_head")
        self.assertEqual(summary["status"], "review_low_minority_rate")

    def test_summarize_binary_head_high_majority(self):
        values = np.array([1.0] * 50 + [0.0], dtype=np.float64)
        masks = np.array([True] * 51, dtype=bool)
        summary = _summarize_binary_head(values, masks, "test_head")
        self.assertEqual(summary["status"], "review_high_majority_rate")

    def test_summarize_task_from_synthetic_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_dir = Path(tmpdir) / "mowa_future_labels" / "FakeTask"
            sidecar_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "frame_index": pa.array([0, 1, 2, 3], type=pa.int64()),
                    "failure_risk": pa.array([0.0, 0.0, 0.0, 0.0], type=pa.float64()),
                    "failure_risk_mask": pa.array([False, False, False, True], type=pa.bool_()),
                    "subgoal_feasibility": pa.array([0.0, 1.0, 0.0, 1.0], type=pa.float64()),
                    "subgoal_feasibility_mask": pa.array([False, False, False, True], type=pa.bool_()),
                    "manipulation_readiness": pa.array([1.0, 1.0, 0.0, 0.0], type=pa.float64()),
                    "manipulation_readiness_mask": pa.array([False, False, False, True], type=pa.bool_()),
                    "object_visibility_future": pa.array([1.0, 0.0, 1.0, 0.0], type=pa.float64()),
                    "object_visibility_future_mask": pa.array([False, False, False, True], type=pa.bool_()),
                    "next_best_view_score": pa.array([0.5, 0.0, 0.25, 0.0], type=pa.float64()),
                    "next_best_view_score_mask": pa.array([False, False, False, True], type=pa.bool_()),
                }
            )
            pq.write_table(table, sidecar_dir / "episode_000000.parquet")

            report = _summarize_task("FakeTask", sidecar_dir)
            self.assertEqual(report["task"], "FakeTask")
            self.assertEqual(report["episode_count"], 1)
            self.assertEqual(report["heads"]["failure_risk"]["status"], "blocked_single_class")
            self.assertEqual(report["heads"]["subgoal_feasibility"]["status"], "candidate")
            self.assertEqual(report["heads"]["manipulation_readiness"]["status"], "candidate")
            self.assertEqual(report["heads"]["object_visibility_future"]["status"], "candidate")
            self.assertEqual(report["heads"]["next_best_view_score"]["status"], "candidate")
            self.assertEqual(report["heads"]["object_visibility_future"]["positive_count"], 2)
            self.assertGreater(report["heads"]["next_best_view_score"]["std"], 0.0)

    def test_discover_task_sidecar_dirs_finds_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            sidecar_dir = (
                data_root
                / "v1.0"
                / "target"
                / "atomic"
                / "OpenDrawer"
                / "20250816"
                / "lerobot"
                / "mowa_future_labels"
                / "OpenDrawer"
            )
            sidecar_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "frame_index": pa.array([0], type=pa.int64()),
                    "failure_risk": pa.array([0.0], type=pa.float64()),
                    "failure_risk_mask": pa.array([True], type=pa.bool_()),
                    "subgoal_feasibility": pa.array([0.0], type=pa.float64()),
                    "subgoal_feasibility_mask": pa.array([True], type=pa.bool_()),
                    "manipulation_readiness": pa.array([0.0], type=pa.float64()),
                    "manipulation_readiness_mask": pa.array([True], type=pa.bool_()),
                    "object_visibility_future": pa.array([0.0], type=pa.float64()),
                    "object_visibility_future_mask": pa.array([True], type=pa.bool_()),
                    "next_best_view_score": pa.array([0.0], type=pa.float64()),
                    "next_best_view_score_mask": pa.array([True], type=pa.bool_()),
                }
            )
            pq.write_table(table, sidecar_dir / "episode_000000.parquet")

            found = discover_task_sidecar_dirs(data_root)
            self.assertIn("OpenDrawer", found)
            self.assertEqual(found["OpenDrawer"], sidecar_dir)


if __name__ == "__main__":
    unittest.main()
