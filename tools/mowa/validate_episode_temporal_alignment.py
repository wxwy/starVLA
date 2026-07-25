"""校验 RoboCasa LeRobot episode 的视频、state 与 action 帧数是否对齐。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from starVLA.dataloader.mowa.temporal_alignment import (
    DEFAULT_MOWA_VIDEO_KEY,
    validate_mowa_episode_temporal_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate video/state/action temporal alignment for LeRobot episodes."
    )
    parser.add_argument("--dataset-path", type=Path, required=True, help="LeRobot dataset directory.")
    parser.add_argument(
        "--video-key",
        dest="video_keys",
        action="append",
        default=None,
        help=f"Video key to validate. Repeatable. Default: {DEFAULT_MOWA_VIDEO_KEY}",
    )
    parser.add_argument(
        "--episode-index",
        dest="episode_indices",
        type=int,
        action="append",
        default=None,
        help="Episode index to validate. Repeatable. Default validates all episodes.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_mowa_episode_temporal_alignment(
        args.dataset_path,
        video_keys=tuple(args.video_keys) if args.video_keys else (DEFAULT_MOWA_VIDEO_KEY,),
        episode_indices=tuple(args.episode_indices) if args.episode_indices else None,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if report.invalid_episode_count or report.checked_episode_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
