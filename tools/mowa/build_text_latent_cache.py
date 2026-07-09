"""CLI to build MoWA episode-level text latent stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.text_latent_store import (
    MoWATextLatentStoreConfig,
    build_mowa_text_latent_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA episode-level text latent stores.")
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
        help="Output text_latent_cache/ root.",
    )
    parser.add_argument(
        "--encoder-kind",
        choices=("fake", "umt5"),
        default="fake",
        help="Encoder implementation. Default: fake",
    )
    parser.add_argument(
        "--encoder-name",
        default="google/umt5-xxl",
        help="Encoder name recorded in store attrs.",
    )
    parser.add_argument(
        "--encoder-version",
        default="TBD",
        help="Encoder version recorded in store attrs.",
    )
    parser.add_argument(
        "--encoder-model-path",
        type=Path,
        default=None,
        help="Local Wan2.2 model directory for umt5 encoder kind.",
    )
    parser.add_argument(
        "--text-hidden-dim",
        type=int,
        default=4096,
        help="UMT5 hidden dimension.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32", "bfloat16"),
        help="Stored latent dtype.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing text latent stores.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Materialize text latent stores. Without this flag, the command is dry-run only.",
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

    config = MoWATextLatentStoreConfig(
        dataset_path=args.dataset_path,
        cache_root=args.cache_root,
        encoder_kind=args.encoder_kind,
        encoder_name=args.encoder_name,
        encoder_version=args.encoder_version,
        encoder_model_path=args.encoder_model_path,
        text_hidden_dim=args.text_hidden_dim,
        max_length=args.max_length,
        dtype=args.dtype,
        dry_run=not args.execute,
        overwrite=args.overwrite,
        episode_indices=tuple(args.episode_indices) if args.episode_indices is not None else None,
    )

    report = build_mowa_text_latent_store(config).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
