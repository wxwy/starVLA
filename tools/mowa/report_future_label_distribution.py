#!/usr/bin/env python
"""Report per-task future-label distributions from pre-computed sidecars.

This tool reads the sidecar parquets written by
``precompute_all_atomic_task_future_labels.py`` and produces a quantitative
summary of ``subgoal_feasibility``, ``manipulation_readiness`` and
``failure_risk`` labels/masks.  The report is meant to support the data-gate
decision of whether to unmask each head in production training.

Examples
--------
    # Report distributions for all tasks with sidecars.
    python tools/mowa/report_future_label_distribution.py

    # Custom data root and output path.
    python tools/mowa/report_future_label_distribution.py \
        --data-root /path/to/robocasa365 \
        --output /tmp/mowa_label_distribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_ROOT = Path("playground/Datasets/robocasa365")
DEFAULT_OUTPUT = Path("docs_zh/mowa/mowa_all_atomic_task_future_label_distribution.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report MoWA future-label distributions from sidecars."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path for the distribution report.",
    )
    return parser.parse_args()


def discover_task_sidecar_dirs(data_root: Path) -> dict[str, Path]:
    """Return a mapping from task name to sidecar directory path."""
    atomic_root = data_root / "v1.0" / "target" / "atomic"
    sidecar_dirs: dict[str, Path] = {}
    if not atomic_root.is_dir():
        return sidecar_dirs

    for task_dir in sorted(atomic_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for date_dir in sorted(task_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            candidate = date_dir / "lerobot" / "mowa_future_labels" / task_name
            if candidate.is_dir() and any(candidate.glob("episode_*.parquet")):
                sidecar_dirs[task_name] = candidate
                break
    return sidecar_dirs


def _summarize_binary_head(
    values: np.ndarray,
    masks: np.ndarray,
    head_name: str,
) -> dict[str, Any]:
    """Summarize a binary head.  ``masks`` are True for active (unmasked) samples."""
    total = int(values.shape[0])
    active = int(masks.sum())
    inactive = total - active
    mask_rate = inactive / total if total else 0.0

    active_values = values[masks]
    positive = int((active_values > 0.5).sum())
    negative = active - positive
    positive_rate = positive / active if active else 0.0

    return {
        "head": head_name,
        "total_samples": total,
        "active_samples": active,
        "masked_samples": inactive,
        "mask_rate": round(mask_rate, 6),
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": round(positive_rate, 6),
        "status": _head_status(active, positive, negative, mask_rate),
    }


def _head_status(active: int, positive: int, negative: int, mask_rate: float) -> str:
    """Return a coarse status label for a binary head distribution."""
    if active == 0:
        return "blocked_all_masked"
    if positive == 0 or negative == 0:
        return "blocked_single_class"
    if mask_rate > 0.95:
        return "review_high_mask_rate"
    positive_rate = positive / active
    if positive_rate > 0.95:
        return "review_high_majority_rate"
    if positive_rate < 0.05:
        return "review_low_minority_rate"
    return "candidate"


def _summarize_continuous_head(
    values: np.ndarray,
    masks: np.ndarray,
    head_name: str,
) -> dict[str, Any]:
    """Summarize a continuous head.  ``masks`` are True for active samples."""
    total = int(values.shape[0])
    active = int(masks.sum())
    inactive = total - active
    mask_rate = inactive / total if total else 0.0

    active_values = values[masks]
    if active > 0:
        mean = float(active_values.mean())
        std = float(active_values.std())
        min_val = float(active_values.min())
        max_val = float(active_values.max())
    else:
        mean = std = min_val = max_val = 0.0

    status = "candidate"
    if active == 0:
        status = "blocked_all_masked"
    elif mask_rate > 0.95:
        status = "review_high_mask_rate"
    elif std < 1e-6:
        status = "review_low_variance"

    return {
        "head": head_name,
        "total_samples": total,
        "active_samples": active,
        "masked_samples": inactive,
        "mask_rate": round(mask_rate, 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min_val, 6),
        "max": round(max_val, 6),
        "status": status,
    }


def _summarize_task(task_name: str, sidecar_dir: Path) -> dict[str, Any]:
    """Aggregate sidecar statistics for one task."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Distribution report requires pyarrow.") from exc

    parquet_paths = sorted(sidecar_dir.glob("episode_*.parquet"))
    if not parquet_paths:
        return {
            "task": task_name,
            "sidecar_dir": str(sidecar_dir),
            "episode_count": 0,
            "error": "no sidecar files found",
        }

    all_subgoal_values: list[np.ndarray] = []
    all_subgoal_masks: list[np.ndarray] = []
    all_readiness_values: list[np.ndarray] = []
    all_readiness_masks: list[np.ndarray] = []
    all_failure_risk_values: list[np.ndarray] = []
    all_failure_risk_masks: list[np.ndarray] = []
    all_object_visibility_values: list[np.ndarray] = []
    all_object_visibility_masks: list[np.ndarray] = []
    all_next_best_view_values: list[np.ndarray] = []
    all_next_best_view_masks: list[np.ndarray] = []

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        data = table.to_pydict()

        all_subgoal_values.append(np.asarray(data["subgoal_feasibility"], dtype=np.float64))
        all_subgoal_masks.append(~np.asarray(data["subgoal_feasibility_mask"], dtype=bool))
        all_readiness_values.append(np.asarray(data["manipulation_readiness"], dtype=np.float64))
        all_readiness_masks.append(~np.asarray(data["manipulation_readiness_mask"], dtype=bool))
        all_failure_risk_values.append(np.asarray(data["failure_risk"], dtype=np.float64))
        all_failure_risk_masks.append(~np.asarray(data["failure_risk_mask"], dtype=bool))
        all_object_visibility_values.append(np.asarray(data["object_visibility_future"], dtype=np.float64))
        all_object_visibility_masks.append(~np.asarray(data["object_visibility_future_mask"], dtype=bool))
        all_next_best_view_values.append(np.asarray(data["next_best_view_score"], dtype=np.float64))
        all_next_best_view_masks.append(~np.asarray(data["next_best_view_score_mask"], dtype=bool))

    subgoal = _summarize_binary_head(
        np.concatenate(all_subgoal_values),
        np.concatenate(all_subgoal_masks),
        "subgoal_feasibility",
    )
    readiness = _summarize_binary_head(
        np.concatenate(all_readiness_values),
        np.concatenate(all_readiness_masks),
        "manipulation_readiness",
    )
    failure_risk = _summarize_binary_head(
        np.concatenate(all_failure_risk_values),
        np.concatenate(all_failure_risk_masks),
        "failure_risk",
    )
    object_visibility = _summarize_binary_head(
        np.concatenate(all_object_visibility_values),
        np.concatenate(all_object_visibility_masks),
        "object_visibility_future",
    )
    next_best_view = _summarize_continuous_head(
        np.concatenate(all_next_best_view_values),
        np.concatenate(all_next_best_view_masks),
        "next_best_view_score",
    )

    return {
        "task": task_name,
        "sidecar_dir": str(sidecar_dir),
        "episode_count": len(parquet_paths),
        "total_samples": int(subgoal["total_samples"]),
        "heads": {
            "subgoal_feasibility": subgoal,
            "manipulation_readiness": readiness,
            "failure_risk": failure_risk,
            "object_visibility_future": object_visibility,
            "next_best_view_score": next_best_view,
        },
    }


def main() -> None:
    args = parse_args()
    sidecar_dirs = discover_task_sidecar_dirs(args.data_root)

    task_reports: list[dict[str, Any]] = []
    for task_name in sorted(sidecar_dirs.keys()):
        print(f"Reporting {task_name} ...")
        task_reports.append(_summarize_task(task_name, sidecar_dirs[task_name]))

    supervision_heads = (
        "subgoal_feasibility",
        "manipulation_readiness",
        "object_visibility_future",
        "next_best_view_score",
    )
    candidate_tasks = []
    review_tasks = []
    blocked_tasks = []
    for report in task_reports:
        heads = report.get("heads", {})
        supervision_statuses = [heads.get(h, {}).get("status") for h in supervision_heads]
        if all(s == "candidate" for s in supervision_statuses):
            candidate_tasks.append(report["task"])
        elif any(str(s).startswith("review_") for s in supervision_statuses):
            review_tasks.append(report["task"])
        else:
            blocked_tasks.append(report["task"])

    summary = {
        "data_root": str(args.data_root),
        "task_count": len(task_reports),
        "tasks": task_reports,
        "candidate_tasks": candidate_tasks,
        "review_tasks": review_tasks,
        "blocked_tasks": blocked_tasks,
        "go_no_go": (
            "TBD: review per-task distributions before unmasking"
            if task_reports else "No-Go: no sidecars found"
        ),
        "notes": [
            "Status 'candidate' means active samples have both classes and minority class >= 5% (binary) "
            "or non-trivial variance (continuous).",
            "Status 'blocked_single_class' means all active labels are the same value.",
            "Status 'review_low_minority_rate' means minority class < 5% of active samples.",
            "Status 'review_high_majority_rate' means majority class > 95% of active samples.",
            "Status 'review_high_mask_rate' means > 95% of samples are masked.",
            "Status 'review_low_variance' means continuous head has near-zero variance.",
            "failure_risk is expected to be blocked_single_class on pure human demo data.",
            "object_visibility_future / next_best_view_score are MuJoCo projection proxies and ignore occlusion.",
        ],
    }

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote distribution report to {args.output}")


if __name__ == "__main__":
    main()
