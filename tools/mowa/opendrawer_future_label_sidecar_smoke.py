"""Build an audit-side OpenDrawer future-label sidecar smoke report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import build_mowa_opendrawer_future_label_sidecar_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an audit-side OpenDrawer future-label sidecar smoke report."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot"),
        help="OpenDrawer LeRobot dataset path.",
    )
    parser.add_argument(
        "--failure-risk-horizon",
        type=int,
        default=10,
        help="Failure-risk future horizon. Default: 10",
    )
    parser.add_argument(
        "--subgoal-horizon",
        type=int,
        default=20,
        help="Subgoal-feasibility future horizon. Default: 20",
    )
    parser.add_argument(
        "--readiness-horizon",
        type=int,
        default=5,
        help="Manipulation-readiness future horizon. Default: 5",
    )
    parser.add_argument(
        "--readiness-progress-delta",
        type=float,
        default=0.10,
        help="Minimum future drawer-progress gain to mark readiness positive. Default: 0.10",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=2,
        help="Maximum number of episodes included in the sidecar smoke. Default: 2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_mowa_opendrawer_future_label_sidecar_smoke(
        dataset_path=args.dataset_path,
        failure_risk_horizon=args.failure_risk_horizon,
        subgoal_horizon=args.subgoal_horizon,
        readiness_horizon=args.readiness_horizon,
        readiness_progress_delta=args.readiness_progress_delta,
        max_episodes=args.max_episodes,
    ).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
