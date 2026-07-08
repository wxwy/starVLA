"""Build MoWA future latent cache with a deterministic fake encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.latent_cache_builder import (
    MoWALatentCacheBuildConfig,
    build_mowa_latent_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA future latent cache.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Dataset root containing data/chunk-000/episode_*.parquet.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Output cache root.",
    )
    parser.add_argument(
        "--encoder-name",
        default="wan-fake-encoder",
        help="Encoder name recorded in the manifest.",
    )
    parser.add_argument(
        "--encoder-version",
        default="fake-v1",
        help="Encoder version recorded in the manifest.",
    )
    parser.add_argument(
        "--encoder-kind",
        choices=("fake", "wan2.2-vae"),
        default="fake",
        help="Encoder implementation to use. Default: fake",
    )
    parser.add_argument(
        "--encoder-model-path",
        type=Path,
        default=None,
        help="Local Wan2.2 model directory for wan2.2-vae encoder kind.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Latent dimension for the fake encoder.",
    )
    parser.add_argument(
        "--video-key",
        dest="video_keys",
        action="append",
        default=None,
        help="Video key to include. Can be repeated. Default uses three common RoboCasa views.",
    )
    parser.add_argument(
        "--episode-index",
        dest="episode_indices",
        type=int,
        action="append",
        default=None,
        help="Episode index to include. Can be repeated. Default scans all episodes.",
    )
    parser.add_argument(
        "--anchor-index",
        dest="anchor_indices",
        type=int,
        action="append",
        default=None,
        help="Explicit anchor index list. Can be repeated.",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=("smoke", "all", "custom"),
        default="smoke",
        help="Anchor selection mode. Default: smoke",
    )
    parser.add_argument(
        "--current-window-steps",
        type=int,
        default=1,
        help="Current window size. Default: 1",
    )
    parser.add_argument(
        "--future-window-steps",
        type=int,
        default=8,
        help="Future window size. Default: 8",
    )
    parser.add_argument(
        "--history-window-steps",
        type=int,
        default=8,
        help="History window size. Default: 8",
    )
    parser.add_argument(
        "--allow-partial-windows",
        action="store_true",
        help="Allow incomplete windows at episode boundaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache artifacts.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Materialize cache artifacts. Without this flag, the command is dry-run only.",
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
    report = build_mowa_latent_cache(
        MoWALatentCacheBuildConfig(
            dataset_path=args.dataset_path,
            cache_root=args.cache_root,
            encoder_name=args.encoder_name,
            encoder_version=args.encoder_version,
            encoder_kind=args.encoder_kind,
            encoder_model_path=args.encoder_model_path,
            latent_dim=args.latent_dim,
            video_keys=tuple(args.video_keys) if args.video_keys is not None else (
                "observation.images.robot0_eye_in_hand",
                "observation.images.robot0_agentview_left",
                "observation.images.robot0_agentview_right",
            ),
            current_window_steps=args.current_window_steps,
            future_window_steps=args.future_window_steps,
            history_window_steps=args.history_window_steps,
            episode_indices=tuple(args.episode_indices) if args.episode_indices is not None else None,
            anchor_indices=tuple(args.anchor_indices) if args.anchor_indices is not None else None,
            anchor_mode=args.anchor_mode,
            allow_partial_windows=args.allow_partial_windows,
            overwrite=args.overwrite,
            dry_run=not args.execute,
        )
    ).to_dict()

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
