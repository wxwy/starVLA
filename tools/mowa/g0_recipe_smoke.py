"""MoWA G0 fixed recipe availability smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import inspect_mowa_robocasa365_atomic_core_recipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MoWA fixed RoboCasa365 recipe availability.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
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
    report = inspect_mowa_robocasa365_atomic_core_recipe(args.data_root).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
