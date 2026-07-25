"""CLI to build MoWA window manifest from episode latent stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.schema import MoWAWindowConfig
from starVLA.dataloader.mowa.window_manifest import (
    MoWAWindowManifestConfig,
    build_mowa_window_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA window manifest.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="latent_cache/ root containing ep_*.h5 files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output parquet path for the window manifest.",
    )
    parser.add_argument(
        "--history-steps",
        type=int,
        default=8,
        help="History window steps.",
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=8,
        help="Future window steps.",
    )
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=None,
        help="Raw action steps. Default: future-steps × 4 for Wan latent windows.",
    )
    parser.add_argument(
        "--anchor-video-key",
        default="observation.images.robot0_agentview_left",
        help="Video key used only to define the shared temporal latent grid.",
    )
    parser.add_argument(
        "--recursive-cache-search",
        action="store_true",
        help="Recursively discover ep_*.h5 under cache-root for one global manifest.",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Stride between consecutive history frames.",
    )
    parser.add_argument(
        "--wam-hz",
        type=float,
        default=None,
        help="Optional WAM frequency override. Default infers original fps / Wan temporal factor.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="Dataset split.",
    )
    parser.add_argument(
        "--task-name",
        default="TBD",
        help="Task name to record in manifest entries.",
    )
    parser.add_argument(
        "--label-sidecar-root",
        type=Path,
        default=None,
        help="Root directory for ep_*.jsonl label sidecars.",
    )
    parser.add_argument(
        "--episode-index",
        dest="episode_indices",
        type=int,
        action="append",
        default=None,
        help="Episode index to include. Can be repeated. Default scans all stores.",
    )
    parser.add_argument(
        "--allow-partial-windows",
        action="store_true",
        help="Allow incomplete windows at episode boundaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the build report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MoWAWindowManifestConfig(
        cache_root=args.cache_root,
        output_path=args.output_path,
        window_config=MoWAWindowConfig(
            history_steps=args.history_steps,
            future_steps=args.future_steps,
            action_chunk_steps=args.action_chunk_steps,
        ),
        anchor_video_key=args.anchor_video_key,
        history_stride=args.history_stride,
        wam_hz=args.wam_hz,
        split=args.split,
        task_name=args.task_name,
        label_sidecar_root=args.label_sidecar_root,
        episode_indices=tuple(args.episode_indices) if args.episode_indices is not None else None,
        allow_partial_windows=args.allow_partial_windows,
        recursive_cache_search=args.recursive_cache_search,
    )

    report = build_mowa_window_manifest(config).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
