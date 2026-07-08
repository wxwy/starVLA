"""Run a MoWA E-003 real Wan2.2 latent cache smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starVLA.dataloader.mowa import MoWALatentCacheBuildConfig, build_mowa_latent_cache
from starVLA.dataloader.mowa import validate_mowa_latent_cache


DEFAULT_DATASET_PATH = Path(
    "playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot"
)
DEFAULT_CACHE_ROOT = Path("playground/mowa_latent_cache/robocasa365_open_drawer_target_human_wan2_2_latent_cache")
DEFAULT_MODEL_PATH = Path("playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MoWA E-003 Wan2.2 latent cache smoke.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        default=None,
        help="Episode index to include. Can be repeated. Default: 0.",
    )
    parser.add_argument(
        "--video-key",
        action="append",
        default=None,
        help="Video key to encode.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Materialize the Wan2.2 latent cache and validate it.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_e003_wan2_2_latent_cache_smoke(
        args.repo_root,
        dataset_path=args.dataset_path,
        cache_root=args.cache_root,
        model_path=args.model_path,
        episode_indices=tuple(args.episode_index) if args.episode_index is not None else None,
        video_keys=tuple(args.video_key) if args.video_key is not None else None,
        execute=args.execute,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_e003_wan2_2_latent_cache_smoke(
    repo_root: Path | str,
    *,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    episode_indices: tuple[int, ...] | None = None,
    video_keys: tuple[str, ...] | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root)
    dataset_path = Path(dataset_path)
    cache_root = Path(cache_root)
    model_path = Path(model_path)
    if episode_indices is None:
        episode_indices = (0,)
    if video_keys is None:
        video_keys = ("observation.images.robot0_agentview_left",)

    checks = {
        "dataset_path_exists": dataset_path.is_dir(),
        "model_path_exists": model_path.is_dir(),
        "launch_guard_closed": True,
    }
    if not checks["dataset_path_exists"]:
        return {
            "stage": "future_latent_prior",
            "task_id": "M3-004",
            "experiment_id": "E-003",
            "training_started": False,
            "checks": checks,
            "observed": {
                "repo_root": str(root),
                "dataset_path": str(dataset_path),
                "cache_root": str(cache_root),
                "model_path": str(model_path),
            },
            "unresolved_items": [
                "Dataset path does not exist; Wan2.2 smoke cannot start.",
                "This smoke only validates the real Wan2.2 adapter path when data is present.",
            ],
            "go_no_go": "No-Go: Wan2.2 latent cache smoke missing dataset path",
        }
    if not checks["model_path_exists"]:
        return {
            "stage": "future_latent_prior",
            "task_id": "M3-004",
            "experiment_id": "E-003",
            "training_started": False,
            "checks": checks,
            "observed": {
                "repo_root": str(root),
                "dataset_path": str(dataset_path),
                "cache_root": str(cache_root),
                "model_path": str(model_path),
            },
            "unresolved_items": [
                "Wan2.2-TI2V-5B-Diffusers model path does not exist yet.",
                "Real Wan2.2 smoke remains gated until the local download finishes.",
            ],
            "go_no_go": "No-Go: Wan2.2 latent cache smoke missing model path",
        }

    try:
        report = build_mowa_latent_cache(
            MoWALatentCacheBuildConfig(
                dataset_path=dataset_path,
                cache_root=cache_root,
                encoder_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                encoder_version="wan2.2-vae-v1",
                encoder_kind="wan2.2-vae",
                encoder_model_path=model_path,
                latent_dim=1024,
                video_keys=video_keys,
                episode_indices=episode_indices,
                anchor_mode="smoke",
                overwrite=execute,
                dry_run=not execute,
            )
        ).to_dict()
    except Exception as exc:
        checks["build_failed"] = True
        return {
            "stage": "future_latent_prior",
            "task_id": "M3-004",
            "experiment_id": "E-003",
            "training_started": False,
            "checks": checks,
            "observed": {
                "repo_root": str(root),
                "dataset_path": str(dataset_path),
                "cache_root": str(cache_root),
                "model_path": str(model_path),
                "error": f"{type(exc).__name__}: {exc}",
            },
            "unresolved_items": [
                "Real Wan2.2 encoder scaffold is present, but actual encode failed.",
            ],
            "go_no_go": "No-Go: Wan2.2 latent cache smoke failed during encode",
        }

    checks.update(
        {
            "config_valid": bool(report.get("planned_artifact_count", 0) > 0),
            "build_report_written": (
                report.get("written_artifact_count", 0) > 0 if execute else report.get("planned_artifact_count", 0) > 0
            ),
            "encoder_kind_wan2_2_vae": report.get("summary", {}).get("encoder_kind") == "wan2.2-vae",
        }
    )
    validation_report = None
    if execute:
        validation_report = validate_mowa_latent_cache(cache_root).to_dict()
        checks["validation_ok"] = bool(validation_report.get("all_ok"))
    return {
        "stage": "future_latent_prior",
        "task_id": "M3-004",
        "experiment_id": "E-003",
        "training_started": False,
        "checks": checks,
        "observed": {
            "repo_root": str(root),
            "dataset_path": str(dataset_path),
            "cache_root": str(cache_root),
            "model_path": str(model_path),
            "cache_build_report": report,
            "cache_validation_report": validation_report,
        },
        "unresolved_items": [
            "This smoke validates the real Wan2.2 adapter path only when the local model directory exists.",
            "The smoke is dry-run by default; pass --execute to materialize and validate cache artifacts.",
        ],
        "go_no_go": (
            "TBD: Wan2.2 latent cache write and validation passed; training remains gated"
            if execute and all(checks.values()) and validation_report is not None and validation_report.get("all_ok")
            else (
                "TBD: Wan2.2 latent cache smoke passed; real cache write still gated"
                if (not execute and all(checks.values()))
                else "No-Go: Wan2.2 latent cache smoke incomplete"
            )
        ),
    }


if __name__ == "__main__":
    main()
