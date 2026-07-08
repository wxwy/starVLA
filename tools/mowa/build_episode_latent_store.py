"""CLI to build MoWA episode-level latent stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAEpisodeLatentStoreConfig,
    build_mowa_episode_latent_store,
)
from starVLA.dataloader.mowa.schema import DATA_GATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MoWA episode-level latent stores.")
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
        help="Output latent_cache/ root.",
    )
    parser.add_argument(
        "--video-key",
        dest="video_keys",
        action="append",
        default=None,
        help="Video key to encode. Can be repeated.",
    )
    parser.add_argument(
        "--encoder-kind",
        choices=("fake", "wan2.2-vae"),
        default="fake",
        help="Encoder implementation. Default: fake",
    )
    parser.add_argument(
        "--encoder-name",
        default="mowa-fake-encoder",
        help="Encoder name recorded in store attrs.",
    )
    parser.add_argument(
        "--encoder-version",
        default="fake-v1",
        help="Encoder version recorded in store attrs.",
    )
    parser.add_argument(
        "--encoder-model-path",
        type=Path,
        default=None,
        help="Local Wan2.2 model directory for wan2.2-vae encoder kind.",
    )
    parser.add_argument(
        "--latent-model",
        default="Wan2.2-VAE",
        help="Latent model name recorded in store attrs.",
    )
    parser.add_argument(
        "--latent-model-version",
        default="TBD",
        help="Latent model version recorded in store attrs.",
    )
    parser.add_argument(
        "--latent-type",
        default="pooled_vector",
        choices=("pooled_vector", "vae_spatial"),
        help="Latent type recorded in store attrs. Wan VAE supports pooled_vector/vae_spatial.",
    )
    parser.add_argument(
        "--latent-shape-per-frame",
        default=None,
        help='JSON list for latent_shape_per_frame, e.g. "[\"D\"]" or "[\"C\", \"h\", \"w\"]".',
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Latent dimension for the fake encoder.",
    )
    parser.add_argument(
        "--flatten-policy",
        default="none",
        choices=("none", "flatten", "mean_pool"),
        help="Flatten policy recorded in store attrs.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32", "bfloat16"),
        help="Latent dtype.",
    )
    parser.add_argument(
        "--video-backend",
        default="opencv",
        choices=("opencv", "decord"),
        help="Video decoding backend for Wan VAE encoder.",
    )
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=1,
        help="Frames per VAE forward pass. Currently only 1 is supported.",
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
        help="Overwrite existing episode stores.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Materialize episode stores. Without this flag, the command is dry-run only.",
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

    latent_shape_per_frame: tuple[str | int, ...] = ("D",)
    if args.latent_shape_per_frame is not None:
        parsed = json.loads(args.latent_shape_per_frame)
        if not isinstance(parsed, list):
            raise ValueError("--latent-shape-per-frame must be a JSON list.")
        latent_shape_per_frame = tuple(parsed)

    video_keys = tuple(args.video_keys) if args.video_keys is not None else (
        "observation.images.robot0_eye_in_hand",
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
    )

    config = MoWAEpisodeLatentStoreConfig(
        dataset_path=args.dataset_path,
        cache_root=args.cache_root,
        video_keys=video_keys,
        encoder_kind=args.encoder_kind,
        encoder_name=args.encoder_name,
        encoder_version=args.encoder_version,
        encoder_model_path=args.encoder_model_path,
        latent_model=args.latent_model,
        latent_model_version=args.latent_model_version,
        latent_type=args.latent_type,
        latent_shape_per_frame=latent_shape_per_frame,
        latent_dim=args.latent_dim,
        flatten_policy=args.flatten_policy,
        dtype=args.dtype,
        video_backend=args.video_backend,
        vae_batch_size=args.vae_batch_size,
        obs_fps=DATA_GATE,
        action_hz=DATA_GATE,
        wam_hz=DATA_GATE,
        dry_run=not args.execute,
        overwrite=args.overwrite,
        episode_indices=tuple(args.episode_indices) if args.episode_indices is not None else None,
    )

    report = build_mowa_episode_latent_store(config).to_dict()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
