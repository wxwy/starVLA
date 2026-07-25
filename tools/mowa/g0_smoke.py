"""MoWA G0 local smoke check.

该工具只检查本地 RoboCasa365 最小闭环路径是否存在，不下载数据，
不读取 episode，不输出任何 fps/Hz/window 的实测结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import build_mowa_robocasa365_local_smoke_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA G0 local smoke check.")
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
    report = build_mowa_robocasa365_local_smoke_report(args.data_root).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
