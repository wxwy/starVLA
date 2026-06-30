"""MoWA G0 P0 label coverage smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import (
    MOWA_ROBOCASA365_OPEN_DRAWER_RELATIVE_PATH,
    inspect_mowa_p0_label_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA P0 label coverage smoke check.")
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
        help="Episode indices to sample for field-level coverage.",
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
    report = inspect_mowa_p0_label_coverage(
        args.data_root / args.relative_path,
        episode_indices=tuple(args.episode_indices),
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
