"""Run MoWA failure_risk data-gate smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import build_mowa_failure_risk_data_gate_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit failure_risk proxy coverage and distribution.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Failure-risk future horizon. Default: 10",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional cap on episodes scanned per task.",
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
    report = build_mowa_failure_risk_data_gate_smoke(
        data_root=args.data_root,
        horizon=args.horizon,
        max_episodes_per_task=args.max_episodes_per_task,
    ).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
