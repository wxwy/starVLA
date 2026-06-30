"""MoWA G0 metadata-level leakage gate smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import (
    MoWAWindowConfig,
    build_mowa_atomic_core_leakage_gate_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA atomic core leakage gate smoke.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
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
    report = build_mowa_atomic_core_leakage_gate_smoke(
        args.data_root,
        window_config=MoWAWindowConfig(
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            action_chunk_steps=args.action_chunk_steps,
        ),
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
