"""Pre-compute OpenDrawer future labels for production dataloader consumption.

This is the bridge between the audit-sidecar logic in
``future_label_audit.py`` and the training dataloader.  It writes small
parquet sidecars that ``datasets.py:_attach_mowa_future_labels`` can load in
O(1) per episode, avoiding online gzip/XML parsing during training.

Example:
    python tools/mowa/precompute_opendrawer_future_labels.py \\
        --dataset-path \\
            playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot \\
        --output-dir \\
            playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot/mowa_future_labels/opendrawer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.opendrawer_label_cache import write_opendrawer_label_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute OpenDrawer future labels for production training."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the OpenDrawer Lerobot dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the default sidecar output directory.",
    )
    parser.add_argument(
        "--failure-risk-horizon",
        type=int,
        default=10,
        help="Horizon H for failure_risk proxy.",
    )
    parser.add_argument(
        "--subgoal-horizon",
        type=int,
        default=20,
        help="Horizon H for subgoal_feasibility proxy.",
    )
    parser.add_argument(
        "--readiness-horizon",
        type=int,
        default=5,
        help="Horizon H for manipulation_readiness proxy.",
    )
    parser.add_argument(
        "--readiness-progress-delta",
        type=float,
        default=0.1,
        help="Progress delta threshold for the fallback manipulation_readiness proxy.",
    )
    parser.add_argument(
        "--readiness-distance-threshold",
        type=float,
        default=0.05,
        help="EEF-to-handle distance threshold for kinematics-based readiness (meters).",
    )
    parser.add_argument(
        "--disable-kinematics",
        action="store_true",
        help="Disable MuJoCo forward kinematics and use the progress-imminence proxy only.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit the number of episodes to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest.",
    )

    args = parser.parse_args()

    manifest = write_opendrawer_label_cache(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        failure_risk_horizon=args.failure_risk_horizon,
        subgoal_horizon=args.subgoal_horizon,
        readiness_horizon=args.readiness_horizon,
        readiness_progress_delta=args.readiness_progress_delta,
        readiness_distance_threshold=args.readiness_distance_threshold,
        enable_kinematics=not args.disable_kinematics,
        max_episodes=args.max_episodes,
    )

    print(json.dumps(manifest, indent=2))
    if args.output_manifest is not None:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
