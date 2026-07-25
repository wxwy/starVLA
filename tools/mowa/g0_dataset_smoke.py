"""MoWA G0 RoboCasa365 dataset-level smoke check.

该工具只读取 Lerobot metadata 和少量 parquet schema，不读取视频内容，
不启动训练，不输出 fps/Hz/window 的正式数值结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import (
    MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH,
    MoWAWindowConfig,
    inspect_robocasa365_lerobot_dataset_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA G0 dataset-level smoke check.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument(
        "--relative-path",
        default=MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH,
        help="Dataset relative path under data root.",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=[0, 1, 4],
        help="Episode indices to sample for schema/window smoke.",
    )
    parser.add_argument("--history-steps", type=int, default=3)
    parser.add_argument("--future-steps", type=int, default=2)
    parser.add_argument("--action-chunk-steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. When omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data_root / args.relative_path
    smoke = inspect_robocasa365_lerobot_dataset_smoke(
        dataset_path,
        episode_indices=tuple(args.episode_indices),
        window_config=MoWAWindowConfig(
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            action_chunk_steps=args.action_chunk_steps,
        ),
    )
    payload = json.dumps(smoke.to_dict(), ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
