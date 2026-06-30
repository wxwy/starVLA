"""MoWA G0 full-recipe temporal profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import build_mowa_atomic_core_temporal_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA atomic core temporal profile.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional cap for quick checks. Omit for full recipe profile.",
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
    report = build_mowa_atomic_core_temporal_profile(
        args.data_root,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
