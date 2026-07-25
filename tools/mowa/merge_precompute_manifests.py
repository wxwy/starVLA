#!/usr/bin/env python
"""Merge per-group MoWA precompute manifests into a single summary manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple MoWA precompute manifests into one."
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        type=Path,
        help="Input manifest JSON files to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_tasks: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []
    parameters: dict[str, Any] | None = None
    recipe_name: str | None = None
    data_root: str | None = None

    total_subgoal = 0
    total_readiness = 0
    total_failure_risk = 0
    total_ovf = 0
    total_nbv = 0.0

    for manifest_path in args.manifests:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        if recipe_name is None:
            recipe_name = manifest.get("recipe_name")
        if data_root is None:
            data_root = manifest.get("data_root")
        if parameters is None:
            parameters = manifest.get("parameters")

        all_tasks.extend(manifest.get("tasks", []))
        all_skipped.extend(manifest.get("skipped_tasks", []))
        total_subgoal += int(manifest.get("total_subgoal_positive_count", 0))
        total_readiness += int(manifest.get("total_readiness_positive_count", 0))
        total_failure_risk += int(manifest.get("total_failure_risk_labeled_count", 0))
        total_ovf += int(manifest.get("total_object_visibility_positive_count", 0))
        total_nbv += float(manifest.get("total_next_best_view_score_sum", 0.0))

    target_task_count = len(all_tasks) + len(all_skipped)
    summary = {
        "recipe_name": recipe_name,
        "data_root": data_root,
        "target_task_count": target_task_count,
        "processed_task_count": len(all_tasks),
        "skipped_task_count": len(all_skipped),
        "skipped_tasks": all_skipped,
        "total_subgoal_positive_count": total_subgoal,
        "total_readiness_positive_count": total_readiness,
        "total_failure_risk_labeled_count": total_failure_risk,
        "total_object_visibility_positive_count": total_ovf,
        "total_next_best_view_score_sum": total_nbv,
        "tasks": all_tasks,
        "parameters": parameters,
        "go_no_go": (
            "TBD: sidecars built for all requested tasks; review distribution before unmasking"
            if all_tasks else "No-Go: no task sidecars built"
        ),
        "notes": [
            "Sidecars are written under <dataset>/mowa_future_labels/<task_name>/.",
            "failure_risk remains masked in production if the cached distribution is single-class.",
            "subgoal_feasibility / manipulation_readiness masks respect the per-episode builder logic.",
            "Run tools/mowa/report_future_label_distribution.py next to review per-task label distributions.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote merged manifest to {args.output}")
    print(f"  processed tasks: {len(all_tasks)}")
    if all_skipped:
        print(f"  skipped tasks: {len(all_skipped)}")
        for skipped in all_skipped:
            print(f"    - {skipped['task']}: {skipped.get('error', skipped.get('reason'))}")


if __name__ == "__main__":
    main()
