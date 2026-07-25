"""MoWA G0 production-entry preflight smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa import (
    MoWAWindowConfig,
    build_mowa_atomic_core_production_preflight_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA atomic core production preflight smoke.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365"),
        help="RoboCasa365 dataset root. Default: playground/Datasets/robocasa365",
    )
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--rank-count", type=int, default=2)
    parser.add_argument("--max-worker-samples", type=int, default=16)
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
    report = build_mowa_atomic_core_production_preflight_smoke(
        args.data_root,
        val_every=args.val_every,
        worker_count=args.worker_count,
        rank_count=args.rank_count,
        max_worker_samples=args.max_worker_samples,
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
