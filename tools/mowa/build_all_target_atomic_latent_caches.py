"""Build Wan2.2 visual latent caches for all target atomic tasks.

This driver iterates over every ``robocasa365/v1.0/target/atomic/<task>/<date>/lerobot``
dataset, builds an episode-level visual latent store (Wan2.2 temporal VAE), and
writes an aggregate report. UMT5 text latents are built separately as one cache
per unique instruction.

The driver is resumable: existing episode stores are skipped unless ``--overwrite``
is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from starVLA.dataloader.mowa.episode_latent_store import (
    MoWAEpisodeLatentStoreConfig,
    build_mowa_episode_latent_store,
)
from starVLA.dataloader.mowa.schema import DATA_GATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Wan2.2 visual latent caches for all target atomic tasks."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365/v1.0/target/atomic"),
        help="Root directory containing task subdirectories.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic"),
        help="Root directory for output latent caches.",
    )
    parser.add_argument(
        "--encoder-model-path",
        type=Path,
        default=Path("playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        help="Local Wan2.2-TI2V-5B-Diffusers directory.",
    )
    parser.add_argument(
        "--latent-type",
        default="pooled_vector",
        choices=("pooled_vector", "vae_spatial"),
        help="Visual latent type to cache.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=1024,
        help="Latent dimension for pooled_vector output.",
    )
    parser.add_argument(
        "--flatten-policy",
        default="mean_pool",
        choices=("none", "flatten", "mean_pool"),
        help="Flatten policy for visual latent.",
    )
    parser.add_argument(
        "--visual-dtype",
        default="float32",
        choices=("float16", "float32"),
        help="Stored visual latent dtype.",
    )
    parser.add_argument(
        "--video-backend",
        default="opencv",
        choices=("opencv", "decord"),
        help="Video decoding backend.",
    )
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=8,
        help="Legacy compatibility field. Wan2.2 VAE encodes each episode as a continuous clip.",
    )
    parser.add_argument("--num-workers", type=int, default=6, help="CPU video decode threads per episode.")
    parser.add_argument(
        "--video-keys",
        default=None,
        help="Comma-separated list of video keys to encode. Default: all agent cameras.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated list of task names to process. Default: all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode stores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playground/tmp/mowa_target_atomic_latent_cache_build_report.json"),
        help="Aggregate report output path.",
    )
    return parser.parse_args()


def discover_tasks(dataset_root: Path, task_filter: set[str] | None) -> list[tuple[Path, Path]]:
    """Return list of (dataset_path, relative_task_path) tuples."""
    tasks: list[tuple[Path, Path]] = []
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        for date_dir in sorted(task_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            lerobot = date_dir / "lerobot"
            if lerobot.is_dir():
                rel = Path(task_dir.name) / date_dir.name / "lerobot"
                tasks.append((lerobot, rel))
    return tasks


def build_visual_cache(
    dataset_path: Path,
    cache_root: Path,
    encoder_model_path: Path,
    latent_type: str,
    latent_dim: int,
    flatten_policy: str,
    dtype: str,
    video_backend: str,
    overwrite: bool,
    vae_batch_size: int = 8,
    num_workers: int = 6,
    video_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if video_keys is None:
        video_keys = (
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_agentview_right",
        )
    config = MoWAEpisodeLatentStoreConfig(
        dataset_path=dataset_path,
        cache_root=cache_root,
        video_keys=video_keys,
        encoder_kind="wan2.2-vae",
        encoder_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        encoder_version="wan2.2-vae-v1",
        encoder_model_path=encoder_model_path,
        latent_model="Wan2.2-VAE",
        latent_model_version="TBD",
        latent_type=latent_type,
        latent_shape_per_frame=("D",) if latent_type == "pooled_vector" else ("C", "h", "w"),
        latent_dim=latent_dim,
        flatten_policy=flatten_policy,
        dtype=dtype,
        video_backend=video_backend,
        vae_batch_size=vae_batch_size,
        num_workers=num_workers,
        obs_fps=DATA_GATE,
        action_hz=DATA_GATE,
        wam_hz=DATA_GATE,
        dry_run=False,
        overwrite=overwrite,
    )
    report = build_mowa_episode_latent_store(config)
    return report.to_dict()


def main() -> None:
    args = parse_args()

    task_filter = None
    if args.tasks:
        task_filter = {name.strip() for name in args.tasks.split(",") if name.strip()}

    tasks = discover_tasks(args.dataset_root, task_filter)
    if not tasks:
        print("No tasks discovered.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(tasks)} task/date dataset(s).")

    results: list[dict[str, Any]] = []
    for dataset_path, rel_path in tasks:
        print(f"\n=== {rel_path} ===")
        task_result: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "rel_path": str(rel_path),
        }

        visual_cache_root = args.cache_root / rel_path
        print("Building visual latent cache...")
        t0 = time.time()
        try:
            video_keys = None
            if args.video_keys:
                video_keys = tuple(key.strip() for key in args.video_keys.split(",") if key.strip())
            visual_report = build_visual_cache(
                dataset_path=dataset_path,
                cache_root=visual_cache_root,
                encoder_model_path=args.encoder_model_path,
                latent_type=args.latent_type,
                latent_dim=args.latent_dim,
                flatten_policy=args.flatten_policy,
                dtype=args.visual_dtype,
                video_backend=args.video_backend,
                overwrite=args.overwrite,
                vae_batch_size=args.vae_batch_size,
                num_workers=args.num_workers,
                video_keys=video_keys,
            )
            visual_report["elapsed_seconds"] = round(time.time() - t0, 2)
            task_result["visual"] = visual_report
            print(
                f"  visual: {visual_report['written_count']} written, "
                f"{visual_report['skipped_count']} skipped, "
                f"{visual_report['failed_count']} failed"
            )
        except Exception as exc:  # noqa: BLE001
            task_result["visual_error"] = str(exc)
            print(f"  visual failed: {exc}")

        results.append(task_result)

        # Write aggregate report after each task so progress is observable.
        aggregate = {
            "dataset_root": str(args.dataset_root),
            "cache_root": str(args.cache_root),
            "encoder_model_path": str(args.encoder_model_path),
            "latent_type": args.latent_type,
            "tasks_processed": len(results),
            "tasks_total": len(tasks),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAggregate report written to {args.output}")


if __name__ == "__main__":
    main()
