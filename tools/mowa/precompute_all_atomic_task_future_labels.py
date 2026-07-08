#!/usr/bin/env python
"""Pre-compute future-label sidecars for all registered MoWA atomic tasks.

This tool walks the RoboCasa365 atomic-task directory, matches each available
 dataset to a registered ``AtomicTaskLabelBuilder``, and writes per-episode
sidecar parquets under ``<dataset>/mowa_future_labels/<task_name>/``.

Examples
--------
    # Pre-compute sidecars for all available atomic tasks.
    python tools/mowa/precompute_all_atomic_task_future_labels.py

    # Limit to a subset of tasks and episodes for a quick dry run.
    python tools/mowa/precompute_all_atomic_task_future_labels.py \
        --tasks OpenDrawer OpenCabinet --max-episodes 10

    # Use a custom data root.
    python tools/mowa/precompute_all_atomic_task_future_labels.py \
        --data-root /path/to/robocasa365
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import re
from pathlib import Path
from typing import Any

import numpy as np

# Import all task builders so they register themselves.
import starVLA.dataloader.mowa.tasks  # noqa: F401
from starVLA.dataloader.mowa.atomic_task_label_builder import get_registered_atomic_task_names

DEFAULT_DATA_ROOT = Path("playground/Datasets/robocasa365")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute MoWA future-label sidecars for all atomic tasks."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of task names to process. Default: all registered builders.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional per-task episode limit for dry runs.",
    )
    parser.add_argument(
        "--failure-risk-horizon",
        type=int,
        default=10,
        help="Horizon for failure-risk proxy.",
    )
    parser.add_argument(
        "--subgoal-horizon",
        type=int,
        default=20,
        help="Horizon for subgoal-feasibility proxy.",
    )
    parser.add_argument(
        "--readiness-horizon",
        type=int,
        default=10,
        help="Horizon for manipulation-readiness fallback proxy.",
    )
    parser.add_argument(
        "--readiness-distance-threshold",
        type=float,
        default=0.05,
        help="EEF-to-handle distance threshold in meters when kinematics are available.",
    )
    parser.add_argument(
        "--readiness-progress-delta",
        type=float,
        default=0.05,
        help="Future progress-gain threshold for readiness fallback.",
    )
    parser.add_argument(
        "--visibility-horizon",
        type=int,
        default=10,
        help="Horizon for object_visibility_future / next_best_view_score proxy.",
    )
    parser.add_argument(
        "--disable-kinematics",
        action="store_true",
        help="Disable MuJoCo forward-kinematics and use progress-only proxies.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2),
        help="Number of parallel workers for task-level precomputation. Default: half of CPU cores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs_zh/mowa/mowa_all_atomic_task_future_label_cache_manifest.json"),
        help="JSON output path for the summary manifest.",
    )
    return parser.parse_args()


def discover_atomic_dataset_paths(data_root: Path) -> dict[str, Path]:
    """Return a mapping from task name to the latest dated LeRobot dataset path."""
    atomic_root = data_root / "v1.0" / "target" / "atomic"
    if not atomic_root.is_dir():
        raise FileNotFoundError(f"Atomic task root not found: {atomic_root}")

    discovered: dict[str, Path] = {}
    for task_dir in sorted(atomic_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        date_dirs = [
            d for d in sorted(task_dir.iterdir())
            if d.is_dir() and re.fullmatch(r"\d{8}", d.name)
        ]
        if not date_dirs:
            continue
        # Pick the latest dated run that contains a lerobot directory.
        for date_dir in reversed(date_dirs):
            lerobot_dir = date_dir / "lerobot"
            if lerobot_dir.is_dir():
                discovered[task_name] = lerobot_dir
                break
    return discovered


def _maybe_shorten(value: float | None, default: Any = None) -> float | Any:
    """Convert numpy scalar to Python scalar for JSON serialization."""
    if value is None:
        return default
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def _process_one_task(
    task_name: str,
    dataset_path: Path,
    *,
    failure_risk_horizon: int,
    subgoal_horizon: int,
    readiness_horizon: int | None,
    readiness_distance_threshold: float,
    readiness_progress_delta: float,
    enable_kinematics: bool,
    max_episodes: int | None,
) -> dict[str, Any]:
    """Process a single task in a worker process."""
    # Import inside worker to avoid pickling the builder and to ensure MuJoCo is
    # initialized cleanly in the spawned process.
    from starVLA.dataloader.mowa.atomic_task_label_builder import get_builder_for_task
    from starVLA.dataloader.mowa.label_cache import write_label_cache

    builder = get_builder_for_task(task_name)
    try:
        manifest = write_label_cache(
            builder=builder,
            dataset_path=dataset_path,
            failure_risk_horizon=failure_risk_horizon,
            subgoal_horizon=subgoal_horizon,
            readiness_horizon=readiness_horizon,
            readiness_distance_threshold=readiness_distance_threshold,
            readiness_progress_delta=readiness_progress_delta,
            enable_kinematics=enable_kinematics,
            max_episodes=max_episodes,
        )
    except Exception as exc:  # pragma: no cover - runtime resilience
        return {
            "status": "error",
            "task": task_name,
            "dataset_path": str(dataset_path),
            "reason": "builder_error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "status": "ok",
        "task": task_name,
        "manifest": manifest,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    registered_tasks = set(get_registered_atomic_task_names())
    target_tasks = set(args.tasks) if args.tasks else registered_tasks
    unknown = target_tasks - registered_tasks
    if unknown:
        raise ValueError(f"Unknown task names (no registered builder): {sorted(unknown)}")

    dataset_paths = discover_atomic_dataset_paths(data_root)
    task_order = sorted(target_tasks)

    task_manifests: list[dict[str, Any]] = []
    skipped_tasks: list[dict[str, Any]] = []
    total_subgoal_positive = 0
    total_readiness_positive = 0
    total_failure_risk_labeled = 0
    total_object_visibility_positive = 0
    total_next_best_view_score_sum = 0.0

    # Use spawn to avoid fork-safety issues with MuJoCo and torch in workers.
    multiprocessing.set_start_method("spawn", force=True)

    work_items = [
        (task_name, dataset_paths[task_name])
        for task_name in task_order
        if task_name in dataset_paths
    ]
    for task_name in task_order:
        if task_name not in dataset_paths:
            skipped_tasks.append({
                "task": task_name,
                "reason": "dataset_path_not_found",
            })

    print(f"Processing {len(work_items)} tasks with {args.num_workers} workers ...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_one_task,
                task_name,
                dataset_path,
                failure_risk_horizon=args.failure_risk_horizon,
                subgoal_horizon=args.subgoal_horizon,
                readiness_horizon=args.readiness_horizon,
                readiness_distance_threshold=args.readiness_distance_threshold,
                readiness_progress_delta=args.readiness_progress_delta,
                enable_kinematics=not args.disable_kinematics,
                max_episodes=args.max_episodes,
            ): task_name
            for task_name, dataset_path in work_items
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - runtime resilience
                skipped_tasks.append({
                    "task": task_name,
                    "reason": "worker_error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  -> {task_name} worker error: {exc}")
                continue

            if result["status"] == "error":
                skipped_tasks.append({
                    "task": result["task"],
                    "dataset_path": result.get("dataset_path"),
                    "reason": result["reason"],
                    "error": result["error"],
                })
                print(f"  -> {task_name} skipped due to builder error: {result['error']}")
                continue

            manifest = result["manifest"]
            task_manifests.append({
                "task": task_name,
                "dataset_path": str(manifest["dataset_path"]),
                "output_dir": str(manifest["output_dir"]),
                "processed_episode_count": int(manifest["processed_episode_count"]),
                "skipped_no_extras_count": int(manifest["skipped_no_extras_count"]),
                "skipped_empty_count": int(manifest["skipped_empty_count"]),
                "total_row_count": int(manifest["total_row_count"]),
                "subgoal_positive_count": int(manifest["subgoal_positive_count"]),
                "readiness_positive_count": int(manifest["readiness_positive_count"]),
                "failure_risk_labeled_count": int(manifest["failure_risk_labeled_count"]),
                "object_visibility_positive_count": int(manifest["object_visibility_positive_count"]),
                "next_best_view_score_sum": float(manifest["next_best_view_score_sum"]),
                "schema_version": manifest["schema_version"],
                "go_no_go": manifest["go_no_go"],
            })
            total_subgoal_positive += int(manifest["subgoal_positive_count"])
            total_readiness_positive += int(manifest["readiness_positive_count"])
            total_failure_risk_labeled += int(manifest["failure_risk_labeled_count"])
            total_object_visibility_positive += int(manifest["object_visibility_positive_count"])
            total_next_best_view_score_sum += float(manifest["next_best_view_score_sum"])
            print(f"  -> {task_name} done: {manifest['processed_episode_count']} episodes")

    summary = {
        "recipe_name": "mowa_robocasa365_target_human_atomic_core_v1",
        "data_root": str(data_root),
        "target_task_count": len(target_tasks),
        "processed_task_count": len(task_manifests),
        "skipped_task_count": len(skipped_tasks),
        "skipped_tasks": skipped_tasks,
        "total_subgoal_positive_count": total_subgoal_positive,
        "total_readiness_positive_count": total_readiness_positive,
        "total_failure_risk_labeled_count": total_failure_risk_labeled,
        "total_object_visibility_positive_count": total_object_visibility_positive,
        "total_next_best_view_score_sum": total_next_best_view_score_sum,
        "tasks": task_manifests,
        "parameters": {
            "failure_risk_horizon": args.failure_risk_horizon,
            "subgoal_horizon": args.subgoal_horizon,
            "readiness_horizon": args.readiness_horizon,
            "readiness_distance_threshold": args.readiness_distance_threshold,
            "readiness_progress_delta": args.readiness_progress_delta,
            "visibility_horizon": args.visibility_horizon,
            "enable_kinematics": not args.disable_kinematics,
            "max_episodes": args.max_episodes,
        },
        "go_no_go": (
            "TBD: sidecars built for all requested tasks; review distribution before unmasking"
            if task_manifests else "No-Go: no task sidecars built"
        ),
        "notes": [
            "Sidecars are written under <dataset>/mowa_future_labels/<task_name>/.",
            "failure_risk remains masked in production if the cached distribution is single-class.",
            "subgoal_feasibility / manipulation_readiness masks respect the per-episode builder logic.",
            "Run tools/mowa/report_future_label_distribution.py next to review per-task label distributions.",
        ],
    }

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
